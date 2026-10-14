# Foreground-VesselNet (FVNet): 2D / 3D Blood Vessel Segmentation

![FVNet architecture](assets/architecture.png)

FVNet is a foreground-aware encoder–decoder for thin tubular structures (coronary arteries, cerebral/abdominal
vessels, retinal vessels). **One code base, two modes:** set `dim: 3` for volumes (CTA / MRA / MRI, NIfTI) or
`dim: 2` for images (fundus and other 2D vessel datasets, PNG / TIF / GIF / JPG).

## Architecture (matches the diagram)

| Stage | What it does |
|---|---|
| **Preprocessing** | (3D) RAS orientation → intensity window + normalisation → resample to a fixed grid → multi-scale Hessian **Frangi vesselness** → vessel-guided enhancement. (2D) green channel → optional CLAHE → normalisation → Frangi vesselness → enhancement. Output: 3-channel input `[enhanced, vesselness, normalised image]`. |
| **Foreground-Aware Encoder** (4 stages) | Conv Block → **FAM** → **MSFB** → `F1…F4` |
| **FAM** | Channel attention (SE) → spatial attention → feature re-weighting |
| **MSFB** | Parallel 3×3(×3) + 5×5(×5) + dilated conv → concat → 1×1 conv |
| **Bottleneck** | Dilated convs (r = 1, 2, 4) + context aggregation + channel & spatial attention |
| **Attention Decoder** (Up3 → Up1) | Upsampling → **Attention Gate** on the skip connection → concat → Conv Block |
| **Dual-branch heads** | Vessel segmentation head + foreground-confidence head (1×1 conv each) |
| **Adaptive fusion** | `P = α·P_v + (1−α)·P_f`, with `α = sigmoid(learnable scalar)` |
| **Compound loss** | `L = λ1·Dice + λ2·BCE + λ3·Tversky + λ4·Centerline (soft clDice)` + deep supervision on the heads / decoder stages |

Code map: `fvnet/models/` (blocks + network), `fvnet/losses.py`, `fvnet/data/` (cases, vesselness, preprocessing,
patch sampling), `fvnet/engine/` (trainer, sliding-window inference, metrics, tester), `fvnet/visualize.py`, `fvnet/cli.py`.

> Note on "isotropic resampling": as in the original notebook, 3D volumes are resampled to a fixed grid
> (`preprocess.target`, default 256×256×160) and the resulting voxel spacing is tracked for mm-based metrics.
> It is not guaranteed isotropic unless you choose `target` accordingly.

## Install

```bash
git clone https://github.com/<your-user>/FVNet.git && cd FVNet
pip install -r requirements.txt        # or: pip install -e ".[dev]"
```
A CUDA GPU is strongly recommended (CPU works for the tiny smoke test only).

## 3D datasets (volumes)

### ImageCAS (default config)
Folder layout expected by `layout: imagecas`:
```
<root>/1.img.nii/<file>.nii   <root>/1.label.nii/<file>.nii   <root>/2.img.nii/ ...
```
```bash
python scripts/run_all.py -c configs/imagecas_3d.yaml --set data.root=/path/to/imagecas
```
### Any other NIfTI dataset (CTA / MRA / MRI)
Flat layout (`layout: flat`): `images/<id>.nii.gz` and `labels/<id>.nii.gz` (matched by file name). See
`configs/mra_3d.yaml`; set `preprocess.window` to the intensity range to map to [0,1] (HU for CT).

## 2D datasets (images)
Flat layout: `<root>/images/*` and `<root>/labels/*`, matched by file name, by name prefix, or by leading id
(`21_training.tif` ↔ `21_manual1.gif`, `im0001.ppm` ↔ `im0001.ah.ppm`, `Image_01L.jpg` ↔ `Image_01L_1stHO.png`).
```bash
python scripts/run_all.py -c configs/fundus_2d.yaml --set data.root=/path/to/DRIVE_merged
```
Works with DRIVE, STARE, CHASE_DB1, HRF or any 2D vessel set. Notes:
* Fundus vessels are dark in the green channel → `preprocess.bright_vessels: false` (default for `dim: 2`).
  For bright-vessel 2D data (angiograms) set it to `true` and `preprocess.channel: gray`.
* Optional field-of-view masks: `data.fov_dir`. Otherwise a FOV is estimated automatically.
* Images/labels with an official train/test split: pass `data.split_file=split.json`
  (`{"train": [...], "val": [...], "test": [...]}` using the case ids = file stems).

## Pipeline steps
```bash
python scripts/preprocess.py -c <cfg>     # cache preprocessed data + dataset / preprocessing figures
python scripts/train.py      -c <cfg>     # train (resumable) -> best_model.pth, curves
python scripts/test.py       -c <cfg>     # tune threshold on val, evaluate on test, metrics + plots + comparisons
python scripts/run_all.py    -c <cfg>     # all of the above
python scripts/predict.py    -c <cfg> -i new_scan.nii.gz -o preds/   # segment new data (file or folder)
```
Override any config value from the command line: `--set train.epochs=50 train.batch=4 out=runs/exp1`.
Training/preprocessing are **resumable**: re-run the same command after an interruption (e.g. a Kaggle session
restart) and it continues from the last epoch / cached case.

### Kaggle
```python
!git clone https://github.com/<your-user>/FVNet.git
%cd FVNet
!pip install -q nibabel pyyaml
!python scripts/run_all.py -c configs/imagecas_3d.yaml --set out=/kaggle/working/outputs_fvnet data.cache=/kaggle/working/cache_3d
```

### Outputs (`out/`)
`best_model.pth`, `last_model.pth`, `training_history.csv`, `split.json`, `threshold.json`,
`test_metrics_per_case.csv`, `test_metrics_summary.csv`, `config_used.json`,
`figures/` (dataset, preprocessing, training curves, threshold sweep, 19 test plots – ROC/PR, confusion matrix,
Bland-Altman, radar …, plus every plot as an individual PNG in `figures/individual/`),
`comparisons/` (per test case: raw | enhanced | GT | prediction | probability | overlays | TP/FP/FN;
3D adds axial/coronal MIPs; individual PNGs per slice).

Metrics: Dice, IoU, precision, recall, specificity, NPV, accuracy, balanced accuracy, F2, MCC, kappa, volume
similarity, RVD, HD, HD95, ASSD, **clDice** (+ clPrecision/clRecall), connected components, ROC-AUC, AP.

## Key config options (`configs/*.yaml`)
| Key | Meaning |
|---|---|
| `dim` | `3` volumes, `2` images |
| `model.filters` | channels of the 4 encoder stages (default `[32,64,128,256]`) |
| `train.patch`, `infer.roi` | training patch / sliding-window size (any size; inputs are padded to a multiple of 16 internally) |
| `loss.lambdas` | `dice, bce, tversky, cl` weights; `loss.tv_alpha/tv_beta` Tversky FP/FN weights |
| `loss.ds_weights` | deep-supervision weights |
| `test.tta`, `test.min_cc` | flip test-time augmentation, small-component removal |

## Tests
```bash
pip install -e ".[dev]" && pytest -q        # shape / loss / metric tests + 2D and 3D end-to-end smoke runs on synthetic data
make smoke                                  # same smoke run via the scripts
```

## Citation / License
Please add your paper citation here. Released under the MIT License (see `LICENSE`; edit the copyright holder).
