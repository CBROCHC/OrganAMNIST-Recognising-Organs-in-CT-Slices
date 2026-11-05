"""Download OrganAMNIST (MedMNIST+, 64x64) into ./data using the official `medmnist` package.

The package fetches the file from the official Zenodo record and verifies its MD5 checksum.

Usage:
    python scripts/download_data.py            # -> data/organamnist_64.npz
    python scripts/download_data.py --root /some/other/dir
"""
import argparse
from pathlib import Path

from medmnist import OrganAMNIST


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1] / "data"),
                    help="directory where organamnist_64.npz will be stored")
    args = ap.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    # One .npz holds all three splits; instantiating any split triggers the download.
    ds = OrganAMNIST(split="train", download=True, size=64, root=str(root))
    print(f"OK: {root / 'organamnist_64.npz'}  (train images: {len(ds)})")


if __name__ == "__main__":
    main()
