import argparse
import concurrent.futures as futures
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.batch_process_utils import initialize_df
from wsi_core.wsi_utils import StitchCoords


Image.MAX_IMAGE_PIXELS = None

COMPLETED_STATUSES = {"processed", "already_exist", "no_patches"}


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, np.integer)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def _as_int(value, default):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return int(default)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _as_float(value, default):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _normalise_ext(ext):
    if ext is None:
        return None
    ext = ext.strip()
    if not ext:
        return None
    return ext if ext.startswith(".") else "." + ext


def _slide_stem(slide_name):
    return Path(str(slide_name)).stem


def _row_text(row, key):
    value = row.get(key)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if text.lower() in {"", "none", "nan"}:
        return None
    return text


def _parse_id_list(value):
    if value is None:
        return []
    text = str(value).strip()
    if text.lower() in {"", "none", "nan"}:
        return []
    return np.array(text.split(",")).astype(int)


def _count_coords(h5_path):
    if not os.path.isfile(h5_path) or os.path.getsize(h5_path) == 0:
        return 0
    try:
        with h5py.File(h5_path, "r") as handle:
            if "coords" not in handle:
                return 0
            return int(handle["coords"].shape[0])
    except OSError:
        return 0


def _list_source_slides(args):
    slides = sorted([
        slide for slide in os.listdir(args.source)
        if os.path.isfile(os.path.join(args.source, slide))
    ])
    if args.slide_exts:
        exts = tuple(
            ext for ext in (_normalise_ext(ext) for ext in args.slide_exts.split(",")) if ext
        )
        exts = tuple(ext.lower() for ext in exts)
        slides = [slide for slide in slides if slide.lower().endswith(exts)]
    return slides


def _append_missing_source_slides(df, args, params):
    if not args.append_new_slides:
        return df

    existing_stems = {
        _slide_stem(slide)
        for slide in df["slide_id"].dropna().astype(str)
    }
    missing_slides = [
        slide for slide in _list_source_slides(args)
        if _slide_stem(slide) not in existing_stems
    ]
    if not missing_slides:
        return df

    missing_df = initialize_df(
        missing_slides,
        params["seg_params"],
        params["filter_params"],
        params["vis_params"],
        params["patch_params"],
    )
    print("Appended {} source slides missing from process list".format(len(missing_df)))
    return pd.concat([df, missing_df], ignore_index=True, sort=False)


def _refresh_pending_params(df, args, params):
    if not args.refresh_pending_params:
        return df

    pending_mask = df["process"].fillna(1).astype(int) == 1
    if not pending_mask.any():
        return df

    for group_name in ["seg_params", "filter_params", "vis_params", "patch_params"]:
        for key, value in params[group_name].items():
            if key in df.columns:
                df.loc[pending_mask, key] = value
    if "mag" in df.columns:
        df["mag"] = df["mag"].astype(object)
        df.loc[pending_mask, "mag"] = str(args.mag)
    print("Refreshed parameters for {} pending slides".format(int(pending_mask.sum())))
    return df


def _output_complete(slide_id, dirs, want_patch, want_mask, want_stitch):
    checks = []
    if want_patch:
        checks.append(_count_coords(os.path.join(dirs["patch_save_dir"], slide_id + ".h5")) > 0)
    if want_mask:
        checks.append(os.path.isfile(os.path.join(dirs["mask_save_dir"], slide_id + ".jpg")))
    if want_stitch:
        checks.append(os.path.isfile(os.path.join(dirs["stitch_save_dir"], slide_id + ".jpg")))
    return bool(checks) and all(checks)


def _resolve_slide_path(source, slide_name, slide_ext=None, source_path=None, source_slide_id=None):
    slide_name = str(slide_name)
    source_root = Path(source)

    candidates = []
    ext = _normalise_ext(slide_ext)
    lookup_names = [source_path, source_slide_id, slide_name]
    seen = set()

    for lookup_name in lookup_names:
        if lookup_name is None:
            continue
        lookup_name = str(lookup_name).strip()
        if not lookup_name or lookup_name.lower() in {"none", "nan"}:
            continue
        raw_path = Path(lookup_name)
        keyed = str(raw_path)
        if keyed in seen:
            continue
        seen.add(keyed)

        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            candidates.append(source_root / lookup_name)

        if ext and not lookup_name.lower().endswith(ext.lower()):
            candidates.append(source_root / (lookup_name + ext))
            candidates.append(source_root / (_slide_stem(lookup_name) + ext))

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    for lookup_name in lookup_names:
        if lookup_name is None:
            continue
        stem = _slide_stem(lookup_name)
        matches = sorted(source_root.glob(stem + ".*"))
        matches.extend(sorted(source_root.rglob(stem + ".*")))
        for match in matches:
            if match.is_file():
                return str(match)

    return str(candidates[0])


def _load_preset(args):
    seg_params = {
        "seg_level": -1,
        "sthresh": 8,
        "mthresh": 7,
        "close": 4,
        "use_otsu": False,
        "keep_ids": "none",
        "exclude_ids": "none",
    }
    filter_params = {"a_t": 100, "a_h": 16, "max_n_holes": 8}
    vis_params = {"vis_level": -1, "line_thickness": 250}
    patch_params = {"use_padding": True, "contour_fn": "four_pt"}

    if args.preset:
        preset_path = Path(args.preset)
        candidates = [preset_path]
        if not preset_path.is_absolute():
            candidates.extend([
                Path("presets") / args.preset,
                Path("CLAM") / "presets" / args.preset,
            ])
        for candidate in candidates:
            if candidate.is_file():
                preset_df = pd.read_csv(candidate)
                break
        else:
            raise FileNotFoundError("Preset not found: {}".format(args.preset))

        for key in seg_params:
            if key in preset_df:
                seg_params[key] = preset_df.loc[0, key]
        for key in filter_params:
            if key in preset_df:
                filter_params[key] = preset_df.loc[0, key]
        for key in vis_params:
            if key in preset_df:
                vis_params[key] = preset_df.loc[0, key]
        for key in patch_params:
            if key in preset_df:
                patch_params[key] = preset_df.loc[0, key]

    return {
        "seg_params": seg_params,
        "filter_params": filter_params,
        "vis_params": vis_params,
        "patch_params": patch_params,
    }


def _read_or_init_process_df(args, params):
    autogen_path = os.path.join(args.save_dir, "process_list_autogen.csv")
    if args.process_list:
        process_list = Path(args.process_list)
        if not process_list.is_absolute():
            process_list = Path(args.save_dir) / process_list
        df = pd.read_csv(process_list)
        df = initialize_df(
            df,
            params["seg_params"],
            params["filter_params"],
            params["vis_params"],
            params["patch_params"],
        )
    elif args.resume_process_list and os.path.isfile(autogen_path):
        df = pd.read_csv(autogen_path)
        df = initialize_df(
            df,
            params["seg_params"],
            params["filter_params"],
            params["vis_params"],
            params["patch_params"],
        )
    else:
        slides = _list_source_slides(args)
        df = initialize_df(
            slides,
            params["seg_params"],
            params["filter_params"],
            params["vis_params"],
            params["patch_params"],
        )

    if "mag" not in df.columns:
        df["mag"] = args.mag
    if "num_patches" not in df.columns:
        df["num_patches"] = -1
    for key in ["seg_time", "patch_time", "stitch_time", "error"]:
        if key not in df.columns:
            df[key] = ""
    df = _append_missing_source_slides(df, args, params)
    df = _refresh_pending_params(df, args, params)
    return df


def _prepare_current_params(row, wsi_object, defaults, use_default_params, legacy_support):
    if use_default_params:
        current_vis_params = defaults["vis_params"].copy()
        current_filter_params = defaults["filter_params"].copy()
        current_seg_params = defaults["seg_params"].copy()
        current_patch_params = defaults["patch_params"].copy()
    else:
        current_vis_params = {
            key: row.get(key, defaults["vis_params"][key])
            for key in defaults["vis_params"]
        }
        current_filter_params = {
            key: row.get(key, defaults["filter_params"][key])
            for key in defaults["filter_params"]
        }
        current_seg_params = {
            key: row.get(key, defaults["seg_params"][key])
            for key in defaults["seg_params"]
        }
        current_patch_params = {
            key: row.get(key, defaults["patch_params"][key])
            for key in defaults["patch_params"]
        }

    current_vis_params["vis_level"] = _as_int(
        current_vis_params["vis_level"], defaults["vis_params"]["vis_level"]
    )
    current_vis_params["line_thickness"] = _as_int(
        current_vis_params["line_thickness"], defaults["vis_params"]["line_thickness"]
    )

    current_seg_params["seg_level"] = _as_int(
        current_seg_params["seg_level"], defaults["seg_params"]["seg_level"]
    )
    current_seg_params["sthresh"] = _as_int(
        current_seg_params["sthresh"], defaults["seg_params"]["sthresh"]
    )
    current_seg_params["mthresh"] = _as_int(
        current_seg_params["mthresh"], defaults["seg_params"]["mthresh"]
    )
    current_seg_params["close"] = _as_int(
        current_seg_params["close"], defaults["seg_params"]["close"]
    )
    current_seg_params["use_otsu"] = _as_bool(current_seg_params["use_otsu"])
    current_seg_params["keep_ids"] = _parse_id_list(current_seg_params["keep_ids"])
    current_seg_params["exclude_ids"] = _parse_id_list(current_seg_params["exclude_ids"])

    current_filter_params["a_t"] = _as_float(
        current_filter_params["a_t"], defaults["filter_params"]["a_t"]
    )
    current_filter_params["a_h"] = _as_float(
        current_filter_params["a_h"], defaults["filter_params"]["a_h"]
    )
    current_filter_params["max_n_holes"] = _as_int(
        current_filter_params["max_n_holes"], defaults["filter_params"]["max_n_holes"]
    )
    if legacy_support and "a" in row:
        old_area = _as_float(row.get("a"), current_filter_params["a_t"])
        seg_level = current_seg_params["seg_level"]
        scale = wsi_object.level_downsamples[seg_level]
        current_filter_params["a_t"] = int(old_area * (scale[0] * scale[1]) / (512 * 512))

    current_patch_params["use_padding"] = _as_bool(current_patch_params["use_padding"])
    current_patch_params["contour_fn"] = str(current_patch_params["contour_fn"])
    current_patch_params["mag"] = str(row.get("mag", "40"))

    if current_vis_params["vis_level"] < 0:
        if len(wsi_object.level_dim) == 1:
            current_vis_params["vis_level"] = 0
        else:
            current_vis_params["vis_level"] = wsi_object.getOpenSlide().get_best_level_for_downsample(64)

    if current_seg_params["seg_level"] < 0:
        if len(wsi_object.level_dim) == 1:
            current_seg_params["seg_level"] = 0
        else:
            current_seg_params["seg_level"] = wsi_object.getOpenSlide().get_best_level_for_downsample(64)

    updates = {
        "vis_level": current_vis_params["vis_level"],
        "seg_level": current_seg_params["seg_level"],
        "a_t": current_filter_params["a_t"],
    }
    return current_vis_params, current_filter_params, current_seg_params, current_patch_params, updates


def _segment(wsi_object, seg_params, filter_params):
    start = time.time()
    wsi_object.segmentTissue(**seg_params, filter_params=filter_params)
    return time.time() - start


def _patch(wsi_object, patch_params):
    start = time.time()
    wsi_object.process_contours(**patch_params)
    return time.time() - start


def _stitch(h5_path, wsi_object, output_path):
    start = time.time()
    heatmap = StitchCoords(h5_path, wsi_object, downscale=64, bg_color=(0, 0, 0), alpha=-1, draw_grid=False)
    heatmap.save(output_path)
    return time.time() - start


def _process_one_slide(task):
    idx = task["idx"]
    row = task["row"]
    dirs = task["dirs"]
    defaults = task["defaults"]
    args = task["args"]

    slide = str(row["slide_id"])
    slide_id = _slide_stem(slide)
    result = {
        "idx": idx,
        "slide_id": slide,
        "slide_stem": slide_id,
        "status": "failed",
        "seg_time": -1.0,
        "patch_time": -1.0,
        "stitch_time": -1.0,
        "num_patches": -1,
        "updates": {},
        "error": "",
    }

    try:
        os.environ["WSI_CONTOUR_WORKERS"] = str(args["contour_workers"])
        if args["auto_skip"] and _output_complete(
            slide_id,
            dirs,
            want_patch=args["patch"],
            want_mask=args["save_mask"],
            want_stitch=args["stitch"],
        ):
            result["status"] = "already_exist"
            result["num_patches"] = _count_coords(os.path.join(dirs["patch_save_dir"], slide_id + ".h5"))
            return result

        slide_path = _resolve_slide_path(
            dirs["source"],
            slide,
            args["slide_ext"],
            source_path=_row_text(row, "source_path"),
            source_slide_id=_row_text(row, "source_slide_id"),
        )
        if not os.path.isfile(slide_path):
            raise FileNotFoundError("Slide not found: {}".format(slide_path))

        wsi_object = WholeSlideImage(slide_path)
        current_vis, current_filter, current_seg, current_patch, updates = _prepare_current_params(
            row,
            wsi_object,
            defaults,
            args["use_default_params"],
            args["legacy_support"],
        )
        result["updates"] = updates

        w, h = wsi_object.level_dim[current_seg["seg_level"]]
        if w * h > 1e8 and len(wsi_object.level_dim) > 1:
            fallback_seg_level = wsi_object.getOpenSlide().get_best_level_for_downsample(64)
            fallback_w, fallback_h = wsi_object.level_dim[fallback_seg_level]
            if fallback_w * fallback_h <= 1e8:
                current_seg["seg_level"] = fallback_seg_level
                if current_vis["vis_level"] == 0:
                    current_vis["vis_level"] = fallback_seg_level
                result["updates"]["seg_level"] = fallback_seg_level
                result["updates"]["vis_level"] = current_vis["vis_level"]
                w, h = fallback_w, fallback_h

        if w * h > 1e8:
            result["status"] = "failed_seg"
            result["error"] = (
                "segmentation level is too large: {} x {}. "
                "For single-level PNG slides such as UBC-OCEAN, convert them to tiled pyramidal TIFF first "
                "(see tools/convert_ubc_png_to_pyramid_tiff.py), then rerun patch extraction on the .tif folder."
            ).format(w, h)
            return result

        if args["seg"]:
            result["seg_time"] = _segment(wsi_object, current_seg, current_filter)

        if args["save_mask"]:
            mask = wsi_object.visWSI(**current_vis)
            if isinstance(mask, tuple):
                mask = mask[0]
            mask.save(os.path.join(dirs["mask_save_dir"], slide_id + ".jpg"))

        if args["patch"]:
            current_patch.update({
                "patch_level": args["patch_level"],
                "patch_size": args["patch_size"],
                "step_size": args["step_size"],
                "save_path": dirs["patch_save_dir"],
            })
            result["patch_time"] = _patch(wsi_object, current_patch)

        h5_path = os.path.join(dirs["patch_save_dir"], slide_id + ".h5")
        result["num_patches"] = _count_coords(h5_path)

        if args["stitch"] and result["num_patches"] > 0:
            stitch_path = os.path.join(dirs["stitch_save_dir"], slide_id + ".jpg")
            result["stitch_time"] = _stitch(h5_path, wsi_object, stitch_path)

        if args["patch"] and result["num_patches"] <= 0:
            result["status"] = "no_patches"
        else:
            result["status"] = "processed"
        return result

    except Exception as exc:
        result["status"] = "failed"
        result["error"] = "{}\n{}".format(exc, traceback.format_exc(limit=20))
        return result


def _update_process_df(df, result):
    idx = result["idx"]
    for key, value in result.get("updates", {}).items():
        if key not in df.columns:
            df[key] = ""
        df.loc[idx, key] = value

    for key in ["status", "error"]:
        if key in df.columns:
            df[key] = df[key].astype(object)
    df.loc[idx, "status"] = result["status"]
    df.loc[idx, "process"] = 0 if result["status"] in COMPLETED_STATUSES else 1
    df.loc[idx, "num_patches"] = result["num_patches"]
    df.loc[idx, "seg_time"] = result["seg_time"]
    df.loc[idx, "patch_time"] = result["patch_time"]
    df.loc[idx, "stitch_time"] = result["stitch_time"]
    df.loc[idx, "error"] = result["error"]


def build_parser():
    parser = argparse.ArgumentParser(description="Parallel CLAM-style WSI segmentation and patching")
    parser.add_argument("--source", type=str, required=True, help="Folder containing raw WSI files")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save processed data")
    parser.add_argument("--step_size", type=int, default=512)
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--patch_level", type=int, default=0)
    parser.add_argument("--patch", default=False, action="store_true")
    parser.add_argument("--seg", default=False, action="store_true")
    parser.add_argument("--stitch", default=False, action="store_true")
    parser.add_argument("--no_save_mask", default=False, action="store_true")
    parser.add_argument("--no_auto_skip", default=False, action="store_true")
    parser.add_argument("--resume_process_list", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--num_workers", type=int, default=max(1, min(mp.cpu_count(), 8)))
    parser.add_argument(
        "--contour_workers",
        type=int,
        default=1,
        help="Coordinate-filtering workers inside each WSI worker. Increase this for a few very large slides.",
    )
    parser.add_argument("--start_method", type=str, default="fork", choices=["fork", "spawn", "forkserver"])
    parser.add_argument("--preset", default=None, type=str, help="Preset CSV, e.g. tcga.csv")
    parser.add_argument("--process_list", type=str, default=None, help="CSV under save_dir or absolute path")
    parser.add_argument("--slide_ext", type=str, default=None, help="Append this extension if slide_id has no extension")
    parser.add_argument("--slide_exts", type=str, default=None, help="Comma-separated extension filter for source scanning")
    parser.add_argument("--append_new_slides", default=True, action=argparse.BooleanOptionalAction,
                        help="Append files found under --source that are missing from an existing process list")
    parser.add_argument("--refresh_pending_params", default=False, action="store_true",
                        help="Overwrite segmentation/patch parameters for pending slides with the current preset/defaults")
    parser.add_argument("--limit_slides", type=int, default=0, help="Debug only: process at most N pending slides")
    parser.add_argument("--dry_run", default=False, action="store_true", help="Update process list and print pending count without processing slides")
    parser.add_argument("--mag", type=str, default="40", choices=["20", "40"])
    parser.add_argument("--use_default_params", default=False, action="store_true")
    return parser


def main():
    args = build_parser().parse_args()

    patch_save_dir = os.path.join(args.save_dir, "patches")
    mask_save_dir = os.path.join(args.save_dir, "masks")
    stitch_save_dir = os.path.join(args.save_dir, "stitches")
    dirs = {
        "source": args.source,
        "save_dir": args.save_dir,
        "patch_save_dir": patch_save_dir,
        "mask_save_dir": mask_save_dir,
        "stitch_save_dir": stitch_save_dir,
    }
    for key, value in dirs.items():
        print("{}: {}".format(key, value))
        if key != "source":
            os.makedirs(value, exist_ok=True)

    defaults = _load_preset(args)
    df = _read_or_init_process_df(args, defaults)
    legacy_support = "a" in df.columns
    autogen_path = os.path.join(args.save_dir, "process_list_autogen.csv")

    if not args.no_auto_skip:
        for idx, row in df.iterrows():
            status = str(row.get("status", "")).strip()
            if int(row.get("process", 1)) == 0 and status != "no_patches":
                slide_id = _slide_stem(row["slide_id"])
                complete = _output_complete(
                    slide_id,
                    dirs,
                    want_patch=args.patch,
                    want_mask=not args.no_save_mask,
                    want_stitch=args.stitch,
                )
                if not complete:
                    df.loc[idx, "process"] = 1
                    df.loc[idx, "status"] = "tbp"

    df.to_csv(autogen_path, index=False)

    process_mask = df["process"].fillna(1).astype(int) == 1
    process_stack = df[process_mask]
    if len(process_stack) == 0:
        print("No pending slides. Existing process list is already complete.")
        return
    if args.dry_run:
        print("Dry run: {} pending slides, process list={}".format(len(process_stack), autogen_path))
        if "status" in df.columns:
            print(df["status"].fillna("").value_counts(dropna=False).to_string())
        return

    runtime_args = {
        "patch_size": args.patch_size,
        "step_size": args.step_size,
        "patch_level": args.patch_level,
        "patch": args.patch,
        "seg": args.seg,
        "stitch": args.stitch,
        "save_mask": not args.no_save_mask,
        "auto_skip": not args.no_auto_skip,
        "slide_ext": args.slide_ext,
        "mag": args.mag,
        "contour_workers": args.contour_workers,
        "use_default_params": args.use_default_params,
        "legacy_support": legacy_support,
    }
    tasks = [
        {
            "idx": int(idx),
            "row": row.to_dict(),
            "dirs": dirs,
            "defaults": defaults,
            "args": runtime_args,
        }
        for idx, row in process_stack.iterrows()
    ]
    if args.limit_slides and args.limit_slides > 0:
        tasks = tasks[: args.limit_slides]
        print("limit_slides enabled: processing first {} pending slides".format(len(tasks)))

    print("Processing {} pending slides with {} workers".format(len(tasks), args.num_workers))
    ctx = mp.get_context(args.start_method)
    results = []
    with futures.ProcessPoolExecutor(max_workers=args.num_workers, mp_context=ctx) as executor:
        future_to_idx = {executor.submit(_process_one_slide, task): task["idx"] for task in tasks}
        for future in tqdm(futures.as_completed(future_to_idx), total=len(future_to_idx)):
            result = future.result()
            results.append(result)
            _update_process_df(df, result)
            df.to_csv(autogen_path, index=False)
            print(
                "[{}] {} patches={} seg={:.2f}s patch={:.2f}s stitch={:.2f}s".format(
                    result["status"],
                    result["slide_id"],
                    result["num_patches"],
                    result["seg_time"],
                    result["patch_time"],
                    result["stitch_time"],
                )
            )
            if result["error"]:
                print(result["error"])

    completed = sum(1 for result in results if result["status"] in COMPLETED_STATUSES)
    failed = len(results) - completed
    df.to_csv(autogen_path, index=False)
    print("Done. completed={}, failed={}, process_list={}".format(completed, failed, autogen_path))


if __name__ == "__main__":
    main()
