import argparse
from pathlib import Path


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _list_images(root: Path):
    return sorted([p.name for p in root.iterdir() if p.suffix.lower() in IMG_EXTS])


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate meta_info.txt from an image folder.")
    parser.add_argument("--img_dir", required=True, help="Image folder")
    parser.add_argument(
        "--out",
        default=None,
        help="Output meta_info.txt path (default: img_dir/meta_info.txt)",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    img_dir = Path(args.img_dir)
    if not img_dir.exists():
        raise SystemExit(f"Folder not found: {img_dir}")
    names = _list_images(img_dir)
    if not names:
        raise SystemExit(f"No images found in: {img_dir}")
    out_path = Path(args.out) if args.out else (img_dir / "meta_info.txt")
    out_path.write_text("\n".join(names) + "\n", encoding="utf-8")
    print(f"Done. Wrote {len(names)} lines to {out_path}")


if __name__ == "__main__":
    main()
