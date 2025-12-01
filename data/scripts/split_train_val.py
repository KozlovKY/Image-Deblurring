import shutil
from pathlib import Path
from typing import Dict, List, Tuple


def move_file(src: Path, dst: Path) -> None:
    """Move file from src to dst, creating parent directories if needed."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(src), str(dst))


def parse_split_file(txt_path: Path, datasets_root: Path) -> List[Tuple[Path, Path]]:
    """Parse RSBlur_real_*.txt into (blur, sharp) path pairs.

    Lines look like:
      RSBlur/.../gt/gt_sharp.png RSBlur/.../real_blur/real_blur.png

    Returns pairs even if files don't exist (for shard-based workflows).
    """
    pairs: List[Tuple[Path, Path]] = []

    if not txt_path.is_file():
        return pairs

    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            gt_rel, blur_rel = parts
            gt_path = datasets_root / gt_rel
            blur_path = datasets_root / blur_rel
            pairs.append((blur_path, gt_path))

    return pairs


def main() -> None:
    """Split RSBlur dataset into train/val sets."""
    project_root = Path(__file__).resolve().parents[2]
    datasets_root = project_root / "data" / "datasets"
    rsblur_root = datasets_root / "RSBlur"

    if not rsblur_root.is_dir():
        raise SystemExit(f"RSBlur root not found: {rsblur_root}")

    # Используем готовые сплиты из txt в data/datasets/RSblur_splits
    splits_root = datasets_root / "RSblur_splits"
    split_files: Dict[str, Path] = {
        "train": splits_root / "RSBlur_real_train.txt",
        "val": splits_root / "RSBlur_real_val.txt",
    }

    all_pairs: Dict[str, List[Tuple[Path, Path]]] = {}
    for split_name, txt_path in split_files.items():
        pairs = parse_split_file(txt_path, datasets_root)
        all_pairs[split_name] = pairs

    if not all_pairs.get("train"):
        raise SystemExit(
            "No train pairs found. Check RSBlur_real_train.txt and data/datasets layout."
        )

    # Создаём финальную папку для данных обучения
    out_root = datasets_root / "rsblur_train_val"

    def make_name(blur_path: Path) -> str:
        # RSBlur/<scene>/<case>/real_blur/real_blur.png -> 0001_000001
        case_root = blur_path.parent.parent  # RSBlur/<scene>/<case>
        rel = case_root.relative_to(rsblur_root)
        return str(rel).replace("/", "_").replace("\\", "_")

    for subset_name, subset_pairs in all_pairs.items():
        blur_out_dir = out_root / f"blur_{subset_name}"
        sharp_out_dir = out_root / f"sharp_{subset_name}"

        # Создаём выходные директории всегда (для DVC)
        blur_out_dir.mkdir(parents=True, exist_ok=True)
        sharp_out_dir.mkdir(parents=True, exist_ok=True)

        if not subset_pairs:
            continue

        for blur_path, sharp_path in subset_pairs:
            if not blur_path.is_file() or not sharp_path.is_file():
                continue

            base = make_name(blur_path)
            blur_dst = blur_out_dir / f"{base}_blur.png"
            sharp_dst = sharp_out_dir / f"{base}_sharp.png"

            move_file(blur_path, blur_dst)
            move_file(sharp_path, sharp_dst)


if __name__ == "__main__":
    main()
