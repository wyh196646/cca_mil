import argparse
import multiprocessing as mp
import os
import queue
import sys
import time
import traceback
from functools import partial
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


Image.MAX_IMAGE_PIXELS = None

OPENAI_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_STD = [0.26862954, 0.26130258, 0.27577711]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class WholeSlideBagFP:
    def __init__(self, h5_path, slide_path, img_transforms=None):
        self.h5_path = h5_path
        self.slide_path = slide_path
        self.img_transforms = img_transforms
        self._wsi = None  # opened lazily to allow pickling by DataLoader workers
        with h5py.File(h5_path, "r") as handle:
            coords = handle["coords"]
            self.coords = coords[:].astype(np.int64)
            self.patch_level = int(coords.attrs["patch_level"])
            self.patch_size = int(coords.attrs["patch_size"])

    @property
    def wsi(self):
        if self._wsi is None:
            import openslide
            self._wsi = openslide.open_slide(self.slide_path)
        return self._wsi

    # Exclude the unpicklable OpenSlide object when DataLoader forks workers
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_wsi"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        coord = self.coords[idx]
        img = self.wsi.read_region(
            tuple(coord),
            self.patch_level,
            (self.patch_size, self.patch_size),
        ).convert("RGB")
        if self.img_transforms is not None:
            img = self.img_transforms(img)
        return {"img": img, "coord": coord.astype(np.int32)}

    def close(self):
        if self._wsi is not None:
            self._wsi.close()
            self._wsi = None


def _normalise_ext(ext):
    if ext is None:
        return None
    ext = ext.strip()
    if not ext:
        return None
    return ext if ext.startswith(".") else "." + ext


def _slide_stem(slide_name, slide_ext=None):
    name = Path(str(slide_name)).name
    ext = _normalise_ext(slide_ext)
    if ext and name.lower().endswith(ext.lower()):
        return name[: -len(ext)]
    return Path(name).stem


def _row_text(row, key):
    value = row.get(key)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if text.lower() in {"", "none", "nan"}:
        return None
    return text


def _resolve_h5_path(data_h5_dir, slide_id):
    candidates = [
        Path(data_h5_dir) / "patches" / (slide_id + ".h5"),
        Path(data_h5_dir) / (slide_id + ".h5"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return str(candidates[0])


def _resolve_slide_path(data_slide_dir, raw_slide_id, slide_ext, source_path=None, source_slide_id=None):
    slide_dir = Path(data_slide_dir)
    ext = _normalise_ext(slide_ext)

    candidates = []
    lookup_names = [source_path, source_slide_id, raw_slide_id]
    seen = set()
    for lookup_name in lookup_names:
        if lookup_name is None:
            continue
        lookup_name = str(lookup_name).strip()
        if not lookup_name or lookup_name.lower() in {"none", "nan"}:
            continue
        raw = Path(lookup_name)
        keyed = str(raw)
        if keyed in seen:
            continue
        seen.add(keyed)

        if raw.is_absolute():
            candidates.append(raw)
        else:
            candidates.append(slide_dir / lookup_name)
        if ext:
            stem = _slide_stem(lookup_name, slide_ext)
            candidates.append(slide_dir / (stem + ext))

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    for lookup_name in lookup_names:
        if lookup_name is None:
            continue
        stem = _slide_stem(lookup_name, slide_ext)
        matches = sorted(slide_dir.glob(stem + ".*"))
        matches.extend(sorted(slide_dir.rglob(stem + ".*")))
        for match in matches:
            if match.is_file():
                return str(match)
    return str(candidates[0])


def _count_h5_coords(h5_path):
    if not os.path.isfile(h5_path) or os.path.getsize(h5_path) == 0:
        return 0
    try:
        with h5py.File(h5_path, "r") as handle:
            if "coords" not in handle:
                return 0
            return int(handle["coords"].shape[0])
    except OSError:
        return 0


def _format_seconds(value):
    return "{:.3f}s".format(float(value))


def _cuda_worker_info(torch, device):
    info = {"device": str(device)}
    if device.type == "cuda":
        info.update({
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(device),
            "memory_allocated_mb": round(torch.cuda.memory_allocated(device) / 1024 / 1024, 2),
            "memory_reserved_mb": round(torch.cuda.memory_reserved(device) / 1024 / 1024, 2),
        })
    return info


def _complete_pt(pt_path, verify=False):
    if not os.path.isfile(pt_path) or os.path.getsize(pt_path) == 0:
        return False
    if not verify:
        return True
    try:
        import torch

        features = torch.load(pt_path, map_location="cpu")
        return hasattr(features, "dim") and features.dim() == 2 and features.size(0) > 0
    except Exception:
        return False


def _get_eval_transforms(mean, std, target_img_size):
    from torchvision import transforms

    steps = []
    if target_img_size and target_img_size > 0:
        steps.append(transforms.Resize(target_img_size))
    steps.append(transforms.ToTensor())
    steps.append(transforms.Normalize(mean, std))
    return transforms.Compose(steps)


def _get_encoder(model_name, target_img_size, conch_ckpt_path=None, uni_ckpt_path=None):
    import torch

    if model_name == "conch_v1":
        if conch_ckpt_path:
            os.environ["CONCH_CKPT_PATH"] = conch_ckpt_path
        if "CONCH_CKPT_PATH" not in os.environ:
            raise ValueError("CONCH_CKPT_PATH is not set. Pass --conch_ckpt_path or export it.")
        from conch.open_clip_custom import create_model_from_pretrained

        model, _ = create_model_from_pretrained("conch_ViT-B-16", os.environ["CONCH_CKPT_PATH"])
        model.forward = partial(model.encode_image, proj_contrast=False, normalize=False)
        transforms = _get_eval_transforms(OPENAI_MEAN, OPENAI_STD, target_img_size)
        return model, transforms

    if model_name == "uni_v1":
        if uni_ckpt_path:
            os.environ["UNI_CKPT_PATH"] = uni_ckpt_path
        if "UNI_CKPT_PATH" not in os.environ:
            raise ValueError("UNI_CKPT_PATH is not set. Pass --uni_ckpt_path or export it.")
        import timm

        model = timm.create_model(
            "vit_large_patch16_224",
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )
        model.load_state_dict(torch.load(os.environ["UNI_CKPT_PATH"], map_location="cpu"), strict=True)
        transforms = _get_eval_transforms(IMAGENET_MEAN, IMAGENET_STD, target_img_size)
        return model, transforms

    if model_name == "resnet50_trunc":
        import timm
        import torch.nn as nn

        class TimmCNNEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = timm.create_model(
                    "resnet50.tv_in1k",
                    features_only=True,
                    out_indices=(3,),
                    pretrained=True,
                    num_classes=0,
                )
                self.pool = nn.AdaptiveAvgPool2d(1)

            def forward(self, x):
                out = self.model(x)
                if isinstance(out, list):
                    out = out[0]
                return self.pool(out).squeeze(-1).squeeze(-1)

        transforms = _get_eval_transforms(IMAGENET_MEAN, IMAGENET_STD, target_img_size)
        return TimmCNNEncoder(), transforms

    raise NotImplementedError("Unsupported model_name: {}".format(model_name))


def _append_hdf5(output_path, asset_dict, mode, chunk_size):
    with h5py.File(output_path, mode) as handle:
        for key, value in asset_dict.items():
            data_shape = value.shape
            if key not in handle:
                chunk_batch = min(max(1, data_shape[0]), max(1, chunk_size))
                chunks = (chunk_batch,) + data_shape[1:]
                maxshape = (None,) + data_shape[1:]
                dset = handle.create_dataset(
                    key,
                    shape=data_shape,
                    maxshape=maxshape,
                    chunks=chunks,
                    dtype=value.dtype,
                )
                dset[:] = value
            else:
                dset = handle[key]
                dset.resize(len(dset) + data_shape[0], axis=0)
                dset[-data_shape[0] :] = value


def _load_slides(csv_path, data_h5_dir):
    if csv_path:
        df = pd.read_csv(csv_path)
        if "slide_id" not in df.columns:
            raise KeyError("csv_path must contain a slide_id column")
        df["slide_id"] = df["slide_id"].astype(str)
        return df.to_dict("records")

    patch_dir = Path(data_h5_dir) / "patches"
    if not patch_dir.is_dir():
        patch_dir = Path(data_h5_dir)
    return [{"slide_id": path.stem} for path in sorted(patch_dir.glob("*.h5"))]


def _prepare_status_df(slides, process_csv):
    df = pd.DataFrame({"slide_id": slides})
    for key in ["status", "gpu", "num_patches", "time_sec", "h5_path", "pt_path", "error"]:
        df[key] = ""

    if os.path.isfile(process_csv):
        old = pd.read_csv(process_csv)
        if "slide_id" in old.columns:
            old = old.drop_duplicates("slide_id", keep="last").set_index("slide_id")
            for idx, slide_id in df["slide_id"].items():
                if slide_id in old.index:
                    for key in df.columns:
                        if key != "slide_id" and key in old.columns:
                            df.loc[idx, key] = old.loc[slide_id, key]
    return df


def _compute_slide(task, model, img_transforms, device, args, progress_queue=None, gpu_id=None):
    import torch
    from torch.utils.data import DataLoader

    slide_id = task["slide_id"]
    output_h5 = task["output_h5"]
    output_pt = task["output_pt"]
    tmp_h5 = output_h5 + ".partial.{}".format(os.getpid())
    tmp_pt = output_pt + ".partial.{}".format(os.getpid())

    for path in [tmp_h5, tmp_pt]:
        if os.path.exists(path):
            os.remove(path)

    dataset = None
    start = time.time()
    try:
        if not os.path.isfile(task["h5_path"]):
            raise FileNotFoundError("Patch h5 not found: {}".format(task["h5_path"]))
        if not os.path.isfile(task["slide_path"]):
            raise FileNotFoundError("Slide file not found: {}".format(task["slide_path"]))
        if _count_h5_coords(task["h5_path"]) <= 0:
            raise ValueError("Patch h5 has no coords: {}".format(task["h5_path"]))

        dataset = WholeSlideBagFP(task["h5_path"], task["slide_path"], img_transforms=img_transforms)
        total_patches = len(dataset)
        total_batches = int(np.ceil(total_patches / float(args["batch_size"])))
        if progress_queue is not None:
            progress_queue.put({
                "idx": task["idx"],
                "slide_id": slide_id,
                "status": "slide_start",
                "gpu": gpu_id if gpu_id is not None else "cpu",
                "num_patches": total_patches,
                "num_batches": total_batches,
                "h5_path": task["h5_path"],
                "slide_path": task["slide_path"],
            })

        loader_kwargs = {
            "batch_size": args["batch_size"],
            "shuffle": False,
            "num_workers": args["num_workers"],
            "pin_memory": args["pin_memory"] and device.type == "cuda",
            "timeout": args["loader_timeout"],
        }
        if args["num_workers"] > 0:
            loader_kwargs["prefetch_factor"] = args["prefetch_factor"]
            loader_kwargs["multiprocessing_context"] = args["loader_start_method"]

        loader = DataLoader(dataset=dataset, **loader_kwargs)
        mode = "w"
        amp_dtype = torch.float16 if args["amp_dtype"] == "float16" else torch.bfloat16
        processed_patches = 0
        processed_batches = 0
        last_reported_patches = 0
        loader_iter = iter(loader)

        while True:
            load_start = time.time()
            try:
                batch_data = next(loader_iter)
            except StopIteration:
                break
            load_time = time.time() - load_start

            with torch.inference_mode():
                h2d_start = time.time()
                batch = batch_data["img"].to(device, non_blocking=True)
                if args["channels_last"] and device.type == "cuda":
                    batch = batch.to(memory_format=torch.channels_last)
                coords = batch_data["coord"].numpy().astype(np.int32)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                h2d_time = time.time() - h2d_start

                infer_start = time.time()
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                    enabled=args["amp"] and device.type == "cuda",
                ):
                    features = model(batch)
                if isinstance(features, (tuple, list)):
                    features = features[0]
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                infer_time = time.time() - infer_start

                save_start = time.time()
                features = features.detach().float().cpu().numpy().astype(np.float32)
                _append_hdf5(
                    tmp_h5,
                    {"features": features, "coords": coords},
                    mode=mode,
                    chunk_size=args["h5_chunk_size"],
                )
                save_time = time.time() - save_start
                mode = "a"
                processed_batches += 1
                processed_patches += int(coords.shape[0])

                should_report = (
                    processed_batches % args["progress_interval"] == 0
                    or processed_batches == total_batches
                )
                if progress_queue is not None and should_report:
                    delta = processed_patches - last_reported_patches
                    last_reported_patches = processed_patches
                    progress_queue.put({
                        "idx": task["idx"],
                        "slide_id": slide_id,
                        "status": "slide_progress",
                        "gpu": gpu_id if gpu_id is not None else "cpu",
                        "delta_patches": delta,
                        "processed_patches": processed_patches,
                        "num_patches": total_patches,
                        "processed_batches": processed_batches,
                        "num_batches": total_batches,
                        "batch_size": int(coords.shape[0]),
                        "load_time": load_time,
                        "h2d_time": h2d_time,
                        "infer_time": infer_time,
                        "save_time": save_time,
                        "cuda_memory_allocated_mb": (
                            round(torch.cuda.memory_allocated(device) / 1024 / 1024, 2)
                            if device.type == "cuda" else 0
                        ),
                        "cuda_memory_reserved_mb": (
                            round(torch.cuda.memory_reserved(device) / 1024 / 1024, 2)
                            if device.type == "cuda" else 0
                        ),
                    })

                if args["debug_one_batch"]:
                    raise RuntimeError(
                        "debug_one_batch finished first batch for {} on gpu={}; "
                        "load={}, h2d={}, infer={}, save={}".format(
                            slide_id,
                            gpu_id if gpu_id is not None else "cpu",
                            _format_seconds(load_time),
                            _format_seconds(h2d_time),
                            _format_seconds(infer_time),
                            _format_seconds(save_time),
                        )
                    )

        with h5py.File(tmp_h5, "r") as handle:
            features = handle["features"][:]
            coords = handle["coords"][:]
            if features.shape[0] == 0 or features.shape[0] != coords.shape[0]:
                raise ValueError(
                    "Invalid feature h5 for {}: features={}, coords={}".format(
                        slide_id, features.shape, coords.shape
                    )
                )

        tensor = torch.from_numpy(features)
        torch.save(tensor, tmp_pt)
        os.replace(tmp_h5, output_h5)
        os.replace(tmp_pt, output_pt)

        return {
            "idx": task["idx"],
            "slide_id": slide_id,
            "status": "processed",
            "num_patches": int(tensor.shape[0]),
            "time_sec": time.time() - start,
            "h5_path": output_h5,
            "pt_path": output_pt,
            "error": "",
        }
    finally:
        if dataset is not None:
            dataset.close()
        for path in [tmp_h5, tmp_pt]:
            if os.path.exists(path):
                os.remove(path)


def _worker_loop(worker_id, gpu_id, task_queue, result_queue, args):
    try:
        import torch

        if gpu_id is None or not torch.cuda.is_available():
            device = torch.device("cpu")
        else:
            torch.cuda.set_device(gpu_id)
            device = torch.device("cuda:{}".format(gpu_id))
            torch.backends.cuda.matmul.allow_tf32 = args["allow_tf32"]
            if int(torch.cuda.current_device()) != int(gpu_id):
                raise RuntimeError(
                    "Worker {} expected cuda:{}, got cuda:{}".format(
                        worker_id, gpu_id, torch.cuda.current_device()
                    )
                )

        result_queue.put({
            "idx": None,
            "worker_id": worker_id,
            "gpu": gpu_id if gpu_id is not None else "cpu",
            "status": "worker_loading_model",
            "model_name": args["model_name"],
            "cuda": _cuda_worker_info(torch, device),
            "error": "",
        })
        model, img_transforms = _get_encoder(
            args["model_name"],
            args["target_patch_size"],
            conch_ckpt_path=args["conch_ckpt_path"],
            uni_ckpt_path=args["uni_ckpt_path"],
        )
        model = model.eval().to(device)
        if args["channels_last"] and device.type == "cuda":
            model = model.to(memory_format=torch.channels_last)
        first_param_device = str(next(model.parameters()).device)
        if str(device) not in first_param_device:
            raise RuntimeError(
                "Model is on {}, expected {}".format(first_param_device, device)
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        result_queue.put({
            "idx": None,
            "worker_id": worker_id,
            "gpu": gpu_id if gpu_id is not None else "cpu",
            "status": "worker_ready",
            "model_device": first_param_device,
            "cuda": _cuda_worker_info(torch, device),
            "error": "",
        })

        while True:
            task = task_queue.get()
            if task is None:
                break

            try:
                if (
                    args["auto_skip"]
                    and not args["overwrite"]
                    and _complete_pt(task["output_pt"], verify=args["verify_outputs"])
                ):
                    result = {
                        "idx": task["idx"],
                        "slide_id": task["slide_id"],
                        "status": "already_exist",
                        "num_patches": -1,
                        "time_sec": 0.0,
                        "h5_path": task["output_h5"],
                        "pt_path": task["output_pt"],
                        "error": "",
                    }
                else:
                    result = _compute_slide(
                        task,
                        model,
                        img_transforms,
                        device,
                        args,
                        progress_queue=result_queue,
                        gpu_id=gpu_id,
                    )
                result["gpu"] = gpu_id if gpu_id is not None else "cpu"
                result_queue.put(result)
            except Exception as exc:
                result_queue.put({
                    "idx": task["idx"],
                    "slide_id": task["slide_id"],
                    "status": "failed",
                    "gpu": gpu_id if gpu_id is not None else "cpu",
                    "num_patches": -1,
                    "time_sec": 0.0,
                    "h5_path": task["output_h5"],
                    "pt_path": task["output_pt"],
                    "error": "{}\n{}".format(exc, traceback.format_exc(limit=20)),
                })

    except Exception as exc:
        result_queue.put({
            "idx": None,
            "worker_id": worker_id,
            "gpu": gpu_id if gpu_id is not None else "cpu",
            "status": "worker_failed",
            "fatal": True,
            "error": "{}\n{}".format(exc, traceback.format_exc(limit=20)),
        })


def _parse_gpus(gpus):
    if gpus == "auto":
        import torch

        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
        return [None]
    ids = []
    for item in gpus.split(","):
        item = item.strip()
        if item:
            ids.append(int(item))
    return ids if ids else [None]


def _validate_and_describe_gpus(gpu_ids):
    import torch

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    if not torch.cuda.is_available():
        if any(gpu is not None for gpu in gpu_ids):
            raise RuntimeError("CUDA is not available, but GPU ids were requested: {}".format(gpu_ids))
        return ["CUDA_VISIBLE_DEVICES={}".format(visible), "CUDA unavailable; using CPU"]

    count = torch.cuda.device_count()
    lines = ["CUDA_VISIBLE_DEVICES={}".format(visible), "torch cuda device_count={}".format(count)]
    for gpu in gpu_ids:
        if gpu is None:
            continue
        if gpu < 0 or gpu >= count:
            raise ValueError(
                "GPU id {} is out of visible range [0, {}). "
                "If CUDA_VISIBLE_DEVICES is set, --gpus must use logical visible ids.".format(gpu, count)
            )
    for idx in range(count):
        lines.append("cuda:{} -> {}".format(idx, torch.cuda.get_device_name(idx)))
    return lines


def _update_status_df(df, result):
    idx = result["idx"]
    if idx is None:
        return
    for key in ["status", "gpu", "num_patches", "time_sec", "h5_path", "pt_path", "error"]:
        if key not in df.columns:
            df[key] = ""
        df.loc[idx, key] = result.get(key, "")


def build_parser():
    parser = argparse.ArgumentParser(description="Multi-GPU CLAM-style feature extraction")
    parser.add_argument("--data_h5_dir", type=str, required=True)
    parser.add_argument("--data_slide_dir", type=str, required=True)
    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument("--feat_dir", type=str, required=True)
    parser.add_argument("--slide_ext", type=str, default=".svs")
    parser.add_argument("--model_name", type=str, default="conch_v1", choices=["resnet50_trunc", "uni_v1", "conch_v1"])
    parser.add_argument("--conch_ckpt_path", type=str, default=None)
    parser.add_argument("--uni_ckpt_path", type=str, default=None)
    parser.add_argument("--target_patch_size", type=int, default=448)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers per GPU process")
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--loader_timeout", type=int, default=0, help="DataLoader timeout in seconds, 0 disables it")
    parser.add_argument(
        "--loader_start_method",
        type=str,
        default="fork",
        choices=["fork", "spawn", "forkserver"],
        help="Multiprocessing start method for DataLoader workers. fork avoids pickling OpenSlide internals on Linux.",
    )
    parser.add_argument("--progress_interval", type=int, default=1, help="Report progress every N batches")
    parser.add_argument("--h5_chunk_size", type=int, default=256)
    parser.add_argument("--gpus", type=str, default="auto", help="'auto', '0,1,2', or empty for CPU")
    parser.add_argument("--start_method", type=str, default="spawn", choices=["spawn", "fork", "forkserver"])
    parser.add_argument("--no_auto_skip", default=False, action="store_true")
    parser.add_argument("--overwrite", default=False, action="store_true")
    parser.add_argument("--verify_outputs", default=False, action="store_true")
    parser.add_argument("--limit_slides", type=int, default=0, help="Debug only: process at most N pending slides")
    parser.add_argument("--pin_memory", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--amp", default=False, action="store_true")
    parser.add_argument("--amp_dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--allow_tf32", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--channels_last", default=False, action="store_true")
    parser.add_argument("--debug_one_batch", default=False, action="store_true")
    parser.add_argument("--verbose_batches", default=False, action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    args.progress_interval = max(1, int(args.progress_interval))
    args.loader_timeout = max(0, int(args.loader_timeout))
    os.makedirs(args.feat_dir, exist_ok=True)
    h5_dir = os.path.join(args.feat_dir, "h5_files")
    pt_dir = os.path.join(args.feat_dir, "pt_files")
    os.makedirs(h5_dir, exist_ok=True)
    os.makedirs(pt_dir, exist_ok=True)

    slide_records = _load_slides(args.csv_path, args.data_h5_dir)
    slide_ids = [_slide_stem(record["slide_id"], args.slide_ext) for record in slide_records]
    process_csv = os.path.join(args.feat_dir, "process_list_features.csv")
    df = _prepare_status_df(slide_ids, process_csv)

    tasks = []
    for idx, (record, slide_id) in enumerate(zip(slide_records, slide_ids)):
        raw_slide = _row_text(record, "slide_id")
        output_h5 = os.path.join(h5_dir, slide_id + ".h5")
        output_pt = os.path.join(pt_dir, slide_id + ".pt")
        if (
            not args.no_auto_skip
            and not args.overwrite
            and _complete_pt(output_pt, verify=args.verify_outputs)
        ):
            result = {
                "idx": idx,
                "slide_id": slide_id,
                "status": "already_exist",
                "gpu": "",
                "num_patches": -1,
                "time_sec": 0.0,
                "h5_path": output_h5,
                "pt_path": output_pt,
                "error": "",
            }
            _update_status_df(df, result)
            continue

        h5_path = _resolve_h5_path(args.data_h5_dir, slide_id)
        slide_path = _resolve_slide_path(
            args.data_slide_dir,
            raw_slide,
            args.slide_ext,
            source_path=_row_text(record, "source_path"),
            source_slide_id=_row_text(record, "source_slide_id"),
        )
        num_patches = _count_h5_coords(h5_path)
        tasks.append({
            "idx": idx,
            "slide_id": slide_id,
            "raw_slide_id": raw_slide,
            "h5_path": h5_path,
            "slide_path": slide_path,
            "output_h5": output_h5,
            "output_pt": output_pt,
            "num_patches": num_patches,
        })

    df.to_csv(process_csv, index=False)
    if not tasks:
        print("No pending slides. All feature .pt files already exist.")
        print("process list: {}".format(process_csv))
        return

    if args.limit_slides and args.limit_slides > 0:
        tasks = tasks[: args.limit_slides]
        print("limit_slides enabled: processing first {} pending slides".format(len(tasks)))

    gpu_ids = _parse_gpus(args.gpus)
    for line in _validate_and_describe_gpus(gpu_ids):
        print(line)
    total_patches = sum(max(0, int(task.get("num_patches", 0))) for task in tasks)
    print("Pending slides: {}".format(len(tasks)))
    print("Pending patches: {}".format(total_patches))
    print("Workers: {}".format(["cpu" if gpu is None else "cuda:{}".format(gpu) for gpu in gpu_ids]))
    print("Feature dir: {}".format(args.feat_dir))
    print("Status csv: {}".format(process_csv))

    ctx = mp.get_context(args.start_method)
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    worker_args = vars(args).copy()
    worker_args["auto_skip"] = not args.no_auto_skip

    for task in tasks:
        task_queue.put(task)
    for _ in gpu_ids:
        task_queue.put(None)

    workers = []
    for worker_id, gpu_id in enumerate(gpu_ids):
        proc = ctx.Process(
            target=_worker_loop,
            args=(worker_id, gpu_id, task_queue, result_queue, worker_args),
        )
        proc.start()
        workers.append(proc)
        print("started worker {} pid={} gpu={}".format(worker_id, proc.pid, gpu_id if gpu_id is not None else "cpu"), flush=True)

    ready = 0
    completed = 0
    active_slides = {}
    with tqdm(total=len(tasks), desc="Slides", position=0) as slide_pbar, tqdm(
        total=total_patches if total_patches > 0 else None,
        desc="Patches",
        position=1,
        unit="patch",
    ) as patch_pbar:
        while completed < len(tasks):
            try:
                result = result_queue.get(timeout=5)
            except queue.Empty:
                dead = [proc for proc in workers if not proc.is_alive() and proc.exitcode not in (0, None)]
                if dead:
                    for proc in workers:
                        if proc.is_alive():
                            proc.terminate()
                    raise RuntimeError("A feature worker died before finishing all tasks.")
                continue

            if result.get("fatal"):
                for proc in workers:
                    if proc.is_alive():
                        proc.terminate()
                raise RuntimeError(result.get("error", "feature worker failed"))

            if result.get("status") == "worker_loading_model":
                cuda = result.get("cuda", {})
                tqdm.write(
                    "worker {} loading {} on {} | cuda={}".format(
                        result["worker_id"],
                        result.get("model_name", args.model_name),
                        result["gpu"],
                        cuda,
                    )
                )
                continue

            if result.get("status") == "worker_ready":
                ready += 1
                tqdm.write(
                    "worker {} ready on {} | model_device={} | cuda={}".format(
                        result["worker_id"],
                        result["gpu"],
                        result.get("model_device", "unknown"),
                        result.get("cuda", {}),
                    )
                )
                continue

            if result.get("status") == "slide_start":
                active_slides[result["idx"]] = {
                    "slide_id": result["slide_id"],
                    "gpu": result["gpu"],
                    "num_patches": int(result.get("num_patches", 0)),
                    "processed_patches": 0,
                }
                tqdm.write(
                    "[start] {} gpu={} patches={} batches={}".format(
                        result["slide_id"],
                        result["gpu"],
                        result.get("num_patches", 0),
                        result.get("num_batches", 0),
                    )
                )
                continue

            if result.get("status") == "slide_progress":
                delta = int(result.get("delta_patches", 0))
                if delta > 0:
                    patch_pbar.update(delta)
                active_slides[result["idx"]] = {
                    "slide_id": result["slide_id"],
                    "gpu": result["gpu"],
                    "num_patches": int(result.get("num_patches", 0)),
                    "processed_patches": int(result.get("processed_patches", 0)),
                }
                patch_pbar.set_postfix_str(
                    "{} gpu={} {}/{} load={} h2d={} infer={} save={} mem={}/{}MB".format(
                        result["slide_id"][:24],
                        result["gpu"],
                        result.get("processed_batches", 0),
                        result.get("num_batches", 0),
                        _format_seconds(result.get("load_time", 0.0)),
                        _format_seconds(result.get("h2d_time", 0.0)),
                        _format_seconds(result.get("infer_time", 0.0)),
                        _format_seconds(result.get("save_time", 0.0)),
                        result.get("cuda_memory_allocated_mb", 0),
                        result.get("cuda_memory_reserved_mb", 0),
                    )
                )
                if args.verbose_batches:
                    tqdm.write(
                        "[batch] {} gpu={} batch={}/{} patches={}/{} "
                        "load={} h2d={} infer={} save={} mem={}/{}MB".format(
                            result["slide_id"],
                            result["gpu"],
                            result.get("processed_batches", 0),
                            result.get("num_batches", 0),
                            result.get("processed_patches", 0),
                            result.get("num_patches", 0),
                            _format_seconds(result.get("load_time", 0.0)),
                            _format_seconds(result.get("h2d_time", 0.0)),
                            _format_seconds(result.get("infer_time", 0.0)),
                            _format_seconds(result.get("save_time", 0.0)),
                            result.get("cuda_memory_allocated_mb", 0),
                            result.get("cuda_memory_reserved_mb", 0),
                        )
                    )
                continue

            _update_status_df(df, result)
            df.to_csv(process_csv, index=False)
            completed += 1
            slide_pbar.update(1)

            active = active_slides.pop(result["idx"], None)
            if result["status"] in {"processed", "already_exist"} and active is not None:
                missing = active["num_patches"] - active["processed_patches"]
                if missing > 0:
                    patch_pbar.update(missing)

            tqdm.write(
                "[{}] {} gpu={} patches={} time={:.2f}s".format(
                    result["status"],
                    result["slide_id"],
                    result.get("gpu", ""),
                    result.get("num_patches", -1),
                    result.get("time_sec", 0.0),
                )
            )
            if result.get("error"):
                tqdm.write(result["error"])

    for proc in workers:
        proc.join()
        if proc.exitcode != 0:
            raise RuntimeError("Feature worker exited with code {}".format(proc.exitcode))

    df.to_csv(process_csv, index=False)
    n_failed = int((df["status"] == "failed").sum())
    n_done = int(df["status"].isin(["processed", "already_exist"]).sum())
    print("Done. done={}, failed={}, process_list={}".format(n_done, n_failed, process_csv))


if __name__ == "__main__":
    main()
