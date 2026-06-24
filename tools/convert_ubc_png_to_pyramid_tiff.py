#!/usr/bin/env python3
# coding=utf-8
"""Convert UBC-OCEAN PNG slides to tiled pyramidal TIFF for OpenSlide/CLAM."""

import argparse
import concurrent.futures as futures
import os
import shutil
import subprocess
from pathlib import Path

from tqdm import tqdm


def list_pngs(source):
    return sorted(path for path in Path(source).iterdir() if path.is_file() and path.suffix.lower() == ".png")


def convert_one(task):
    src, dst, args = task
    if dst.is_file() and dst.stat().st_size > 0 and not args.overwrite:
        return {"src": str(src), "dst": str(dst), "status": "skip", "error": ""}

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    cmd = [
        args.vips_bin,
        "tiffsave",
        str(src),
        str(tmp),
        "--tile",
        "--pyramid",
        "--bigtiff",
        "--compression", args.compression,
        "--tile-width", str(args.tile_size),
        "--tile-height", str(args.tile_size),
    ]
    if args.compression in {"jpeg", "webp"}:
        cmd.extend(["--Q", str(args.quality)])
    if args.subifd:
        cmd.append("--subifd")
    if args.strip:
        cmd.append("--strip")

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        if tmp.exists():
            tmp.unlink()
        return {
            "src": str(src),
            "dst": str(dst),
            "status": "failed",
            "error": proc.stderr.strip() or proc.stdout.strip(),
        }

    tmp.replace(dst)
    return {"src": str(src), "dst": str(dst), "status": "converted", "error": ""}


def build_parser():
    parser = argparse.ArgumentParser(description="Convert UBC-OCEAN PNG slides to pyramidal TIFF")
    parser.add_argument("--source", required=True, help="Folder containing UBC-OCEAN .png files")
    parser.add_argument("--dest", required=True, help="Output folder for pyramidal .tif files")
    parser.add_argument("--workers", type=int, default=2, help="Concurrent vips conversions")
    parser.add_argument("--limit", type=int, default=0, help="Debug only: convert at most N files")
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--vips-bin", default=shutil.which("vips") or "vips")
    parser.add_argument("--compression", default="jpeg", choices=["jpeg", "deflate", "lzw", "zstd", "webp", "none"])
    parser.add_argument("--quality", type=int, default=90, help="JPEG/WEBP quality")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--subifd", action="store_true", default=False, help="Store pyramid levels as SubIFDs")
    parser.add_argument("--strip", action="store_true", default=True, help="Strip metadata")
    parser.add_argument("--no-strip", dest="strip", action="store_false")
    return parser


def main():
    args = build_parser().parse_args()
    if shutil.which(args.vips_bin) is None and not os.path.isfile(args.vips_bin):
        raise FileNotFoundError("vips executable not found: {}".format(args.vips_bin))

    pngs = list_pngs(args.source)
    if args.limit and args.limit > 0:
        pngs = pngs[: args.limit]
    if not pngs:
        raise FileNotFoundError("No PNG files found under {}".format(args.source))

    tasks = [
        (src, Path(args.dest) / (src.stem + ".tif"), args)
        for src in pngs
    ]

    counts = {"converted": 0, "skip": 0, "failed": 0}
    with futures.ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_list = [executor.submit(convert_one, task) for task in tasks]
        for future in tqdm(futures.as_completed(future_list), total=len(future_list)):
            result = future.result()
            counts[result["status"]] = counts.get(result["status"], 0) + 1
            if result["status"] == "failed":
                print("[failed] {} -> {}\n{}".format(result["src"], result["dst"], result["error"]))

    print("Done. converted={converted}, skip={skip}, failed={failed}, dest={dest}".format(
        dest=args.dest,
        **counts,
    ))
    if counts.get("failed", 0) > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
