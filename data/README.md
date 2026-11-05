# Data

The dataset file is **not stored in this repository**: `organamnist_64.npz` is ≈200 MB, above GitHub's 100 MB per-file limit.

## Download

```bash
python scripts/download_data.py
```

This uses the official [`medmnist`](https://pypi.org/project/medmnist/) package, which downloads the file from the
[MedMNIST+ Zenodo record](https://zenodo.org/records/10519652) and verifies its checksum.
The result is `data/organamnist_64.npz`, which is what the notebook expects (`../data/organamnist_64.npz`).

Manual alternative: download `organamnist_64.npz` from the Zenodo record above and place it in this folder.

## File contents

| Key | Shape | dtype |
|---|---|---|
| `train_images` | (34561, 64, 64) | uint8 (0–255) |
| `train_labels` | (34561, 1) | int |
| `val_images`   | (6491, 64, 64)  | uint8 |
| `val_labels`   | (6491, 1)       | int |
| `test_images`  | (17778, 64, 64) | uint8 |
| `test_labels`  | (17778, 1)      | int |

## License and attribution

MedMNIST data is released under **CC BY 4.0**. If you use it, please cite the MedMNIST papers
(see the [main README](../README.md#references)) and the source-data papers listed on <https://medmnist.com/>.
The dataset is **not intended for clinical use**.
