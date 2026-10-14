"""Build a manifest ``<root>/metadata/image_list_<split>.h5`` by scanning a data root.

Each dataset ``size_<S>`` lists the relative paths of every ``<N>.jpg`` found under
``<source>/<type>/<region>/<S>/`` for which the required feature files exist.

    # index everything under a root (e.g. a small local copy) as split "demo"
    python scripts/build_manifest.py --root /path/to/bigcell_data --split demo
    # only regions whose relative path starts with one of the given prefixes
    python scripts/build_manifest.py --root ... --split my_test --include cptac/ tcga_diagnostic/brca/
"""

import argparse
from pathlib import Path

import h5py

SIZES = [2048, 4096, 8192, 16384, 32768]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--split", required=True)
    ap.add_argument("--include", nargs="*", default=None, help="relative path prefixes to keep")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out = args.root / "metadata" / f"image_list_{args.split}.h5"
    if out.exists() and not args.overwrite:
        raise FileExistsError(f"{out} exists (use --overwrite)")

    entries = {s: [] for s in SIZES}
    for size in SIZES:
        for img in sorted(args.root.glob(f"*/*/*/{size}/*.jpg")):
            if img.stem.endswith("_full"):
                continue
            rel = img.relative_to(args.root).as_posix()
            if args.include and not any(rel.startswith(p) for p in args.include):
                continue
            needed = ["_uni.npy"] + (["_vae.npy"] if size == 2048 else [])
            if all(img.with_name(img.stem + s).exists() for s in needed):
                entries[size].append(rel)

    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as f:
        for size, paths in entries.items():
            if paths:
                f.create_dataset(f"size_{size}", data=paths, dtype=h5py.string_dtype())
            print(f"size_{size}: {len(paths)}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
