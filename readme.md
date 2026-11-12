# imObj Labeler

An app for labeling objects in microscopy images for YOLO and other object detection models.

It runs a local Python server (FastAPI) and shows the interface in your browser. Python handles file reading (Nanonis `.sxm` through [SciFiReaders](https://github.com/gnganesh99/SciFiReaders)) and YOLO inference (ultralytics). The browser handles viewing and box editing.

## Run

```bash
pip install -e .[predict]          # or use the existing .venv
python -m imobj_labeler [DATA_FOLDER] [--models MODELS_FOLDER] [--port 8765]
```

The app opens in your browser at http://127.0.0.1:8765. It remembers the last data folder, recent folders and the models folder in `~/.imobj_labeler/settings.json`.

**Open folder…** and the model **Browse…** button open the native OS folder dialog, which the local server shows. When that isn't possible, the app uses its own picker with Places (Desktop, Documents, Downloads, …), Recent and Drives. Cases where the native dialog isn't possible:
- no display or no tkinter;
- `--no-native-dialog` is set;
- `--host` is not a local address, because the dialog would appear on the server machine.

## Modes

**Labeling.** Drag on the image to draw a box in the current class. Click a box to select it, drag it to move it, and drag its handles to resize it. Press **Register** (Ctrl+S or Enter) to save the image and its labels.

**Prediction.** Choose a models folder (it is scanned for `*.pt` files) and pick a model.
- Unregistered files are predicted automatically when opened. You can also press **Predict** (P) at any time.
- Unreviewed predictions are drawn as dashed boxes with their confidence.
- Editing a prediction marks it as `edited`. Manual boxes are kept when you predict again.
- While no prediction has been edited, moving the confidence or IoU sliders re-runs the prediction.
- Inference uses the model's own training image size unless you set another one.

## Output

Registration writes everything into `obj_det_labels/` inside the data folder:

```
<data_folder>/obj_det_labels/
  images/<stem>.png       the rendered 2D image the boxes were drawn on
  labels/<stem>.txt       YOLO labels: "cls cx cy w h" (normalized)
  classes.json            class definitions
  data.yaml               ultralytics dataset file (refreshed on each registration)
  train.txt / val.txt     image lists used by data.yaml (nothing is copied)
  val_aug.txt             augmented val copies only (data.yaml "test:"), if any
  split.json              dataset source + split + augmentation settings, re-applied on each registration
  aug/images, aug/labels  augmented copies + manifest.json (only when you create them)
  register_log.csv        provenance: source (manual/model/edited), model, render settings
```

Train on the output with `yolo detect train data=<data_folder>/obj_det_labels/data.yaml`.

The **Dataset…** dialog chooses which images go into `data.yaml` and writes a random train/val split:

- **Combined** (default): existing labels from `../labels` plus registered ones. A file that is registered replaces its existing image and labels.
- **Registered**: only `obj_det_labels/images` + `labels`.
- **Existing**: only the data folder's own images with `../labels` (the data folder must be named `images`).

The split is per source file, so all registered versions of one file land in the same set. Nothing is copied: `train.txt` / `val.txt` list the image paths, and YOLO finds each label by swapping `/images/` for `/labels/` in the path. A validation fraction of 0 uses every image for both train and val.

### Augmentation

The Dataset dialog's **Create augmented copies** button writes saved copies of the images in the current split: one copy per ticked transform (rotate 90° / 180° / 270°, Gaussian noise, blur), for train, val or both. They go to `obj_det_labels/aug/images` + `aug/labels` and are added to `train.txt` / `val.txt`. It runs only when you press it, not on every registration, and pressing it again replaces the previous copies. With **Rotate + noise, p** ticked, each rotated copy also gets Gaussian noise (same σ) with probability p; such copies are named `<stem>__rot90_noise.png`. The draw uses the split seed, so the same settings give the same copies.

- Boxes stay exact: only 90° rotations are offered, since an arbitrary rotation makes axis-aligned boxes too large. Leave flips, zoom, crops, mosaic and brightness to YOLO during training (`fliplr`, `flipud`, `scale`, `translate`, `mosaic`, `hsv_v`).
- Each copy follows its source image into train or val, so re-splitting never leaks a val copy into train.
- Copies whose source labels changed since they were made are left out of the lists (the dialog shows how many) until you create them again.
- When val copies exist, `data.yaml` gets `test: val_aug.txt`. `yolo val data=.../data.yaml split=test` then scores the model on the augmented (e.g. noisy) val images alone, while `val.txt` holds the clean and augmented val images together.
- **remove augmented copies** deletes `aug/` and takes the copies out of the lists.

## Training on a GPU machine

**Train…** exports a *training bundle*: a self-contained copy of the current split (set in **Dataset…**, including augmented copies) that trains anywhere with the common `imobj_labeler/train_yolo.py`.

```
<bundle>/
  images/train, images/val, labels/train, labels/val   standard YOLO layout, labels normalised to "0 cx cy w h"
  data.yaml           relative paths (no "path:" key)
  val_aug.txt         augmented val copies alone (data.yaml "test:"), if any
  train_config.json   model, epochs, imgsz, batch, patience, save_period, device, YOLO args
  manifest.json       source of every file, split/augmentation settings, versions
  weights_init/       the starting model, if you chose a local .pt
```

The script needs only `ultralytics` and CUDA `torch`; copy it to the server once (again after app updates):

```bash
python train_yolo.py BUNDLE --check      # validate bundle + CUDA, don't train
nohup python train_yolo.py BUNDLE > BUNDLE/train.log 2>&1 &
python train_yolo.py BUNDLE --resume     # continue the latest run from last.pt
python train_yolo.py BUNDLE --epochs 5   # any config value can be overridden
```

- Each run goes to `BUNDLE/runs/<name>_<time>/`: the usual ultralytics output plus a copy of the script, the config used, and `training_data.json` (settings, versions, GPU, timing, final metrics).
- It stops with an error rather than silently training on CPU when CUDA is missing (`--allow-cpu` overrides).
- Online loggers (wandb, comet, …) are switched off for the run, without touching ultralytics' global settings. Add `"loggers": ["wandb"]` to `train_config.json` to keep one.
- It checks the bundle's `format_version`, and warns when the server's ultralytics version differs from the app's: a model trained with a newer ultralytics may not load in the app.
- After the dialog exports a bundle, it shows the `scp` / `ssh` / `nohup` commands for that bundle, to do it by hand.

### Training from the app (SSH)

The **Train…** dialog can also run everything on the server for you. It uses this PC's own `ssh` with your keys and `~/.ssh/config`. Key-based login is required: the app never asks for or stores a password.

1. **Server:** enter the host alias, a remote folder (default `~/bundles`) and the Python to use (e.g. your venv: `~/venvs/yolo/bin/python`), then click **Save & test connection**. It shows the GPU, torch/CUDA and ultralytics versions, and any problem.
2. **Export & train on server:** exports the bundle, uploads `train_yolo.py` if the server's copy differs (checksum), uploads the bundle, runs `--check`, then starts training detached with `nohup`. The run carries on if the app closes, the PC sleeps or the network drops.
3. **Runs on server:** lists the bundles on the server and their state (running / finished / stopped). Selecting one shows epoch x / y, mAP50 and mAP50-95 (latest and best), a mAP chart and the end of `train.log`. The view refreshes every minute while a run is training and the dialog is open, and only reads files the run writes anyway. **stop** ends a run (its `last.pt` is kept), and **resume** continues it.
4. **Download results:** copies the run (weights, `results.csv`, plots; per-epoch checkpoints are skipped) into the models folder, where Prediction mode lists the new `best.pt`.

On the server:

```
~/bundles/train_yolo.py
~/bundles/<bundle>/train.pid, train.log, runs/<name>_<time>/...
```

## Testing models (Models…)

**Models…** lists the `.pt` files in the models folder (where **Download results** puts them). For each model it shows what its run folder records: when it was trained, epochs done / planned, image size, and the best mAP50-95 during training. Hovering over a name shows the starting model, bundle, GPU and ultralytics version; ⚠ marks a model trained with a different ultralytics version than the app's.

Tick models and image sets, then **Test selected models**. This runs YOLO validation here, on CPU, at each model's own image size, on sets built from the current split in **Dataset…**:

| Set | Images | Tells you |
|---|---|---|
| Validation (clean) | val images, no augmented copies | the honest score |
| Validation, augmented copies | only the augmented val copies (e.g. noisy) | robustness |
| Training (clean) | train images, no copies | compared with val: over-fitting |
| All labelled images | everything (also without a split) | a quick check |

Results (precision, recall, mAP50, mAP50-95) are added to a table, with the best mAP50-95 per set in bold. They are kept in `obj_det_labels/evaluations.json` together with the split and augmentation settings used. **use in Prediction** switches to Prediction mode with that model selected. Validation writes its lists to `obj_det_labels/eval/`. Ultralytics also saves a `labels.cache` next to each labels folder it reads.

## Classes (`classes.json`)

```json
{"task": "detect", "classes": [{"id": 0, "name": "CO", "color": "#e6194b"}]}
```

Edit the classes with **Classes…** in the app.

In Prediction mode, a project that has no `classes.json` yet takes its class names from the model. If the names differ from an existing `classes.json`, the app shows a warning.

## Input handling

- **`.sxm` files.** Only 2D channels are used. Every channel and direction is selectable, and the app falls back to the first one (usually `Z forward`). The image is flipped so the top of the scan is at the top.
- **Display options.** Flatten options are none, offset, line and plane. Contrast is set by percentile clipping, followed by a colormap.
- **Default rendering.** The default is plane flatten, 1–99 % clip and viridis, which matches the existing training JPEGs.
- **RGB and 8-bit images.** These are used as they are. 16-bit and float TIFFs are treated as data.
- **Registered files.** When you reopen a registered file, the render settings it was registered with are applied, so the image and its labels stay consistent.
- **Existing YOLO labels.** If you open a YOLO `images/` folder, labels in its sibling `labels/` folder are loaded as `existing`. Tab separators and float class IDs (`0.0`) are accepted, and labels are always written back as integers separated by spaces.

## Shortcuts

| Key | Action |
| --- | --- |
| 1–9 | Set the class (and change the selected box's class) |
| Del | Delete the selected box |
| Arrows | Nudge the selected box (Shift moves 10 px); with no box selected, go to the previous/next file |
| A / D | Previous / next file |
| Ctrl+Z / Ctrl+Y | Undo / redo |
| P | Predict |
| F | Fit the image to the view |
| H | Show or hide boxes |
| Wheel | Zoom |
| Right-drag or Space+drag | Pan |

## Layout

```
imobj_labeler/
  reader.py       file → 2D array → RGB (channels, flatten, contrast, colormap)
  annotations.py  annotation dicts and YOLO txt I/O
  classes.py      classes.json schema
  project.py      data folder + obj_det_labels output, registration, data.yaml, bundles
  augment.py      offline augmentation (90° rotations, noise, blur) with exact boxes
  train_yolo.py   standalone training script for bundles (copy to the GPU machine)
  remote.py       SSH: test server, upload, start detached, status, stop, download
  evaluate.py     training summary per model, model.val on clean / augmented / train sets
  predict.py      ultralytics model cache and inference
  server.py       FastAPI API and static UI
  web/            index.html, app.js (state/UI), canvas.js (viewer/box editor)
```

Annotations are stored as `{"type": "bbox", ...}` dicts, so a future segmentation mode can add `{"type": "polygon", "points": [...]}` and a `seg_labels/` writer without changing this layout.

## Tests

```bash
python -m pytest tests
```
