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
  project.py      data folder + obj_det_labels output, registration, data.yaml
  predict.py      ultralytics model cache and inference
  server.py       FastAPI API and static UI
  web/            index.html, app.js (state/UI), canvas.js (viewer/box editor)
```

Annotations are stored as `{"type": "bbox", ...}` dicts, so a future segmentation mode can add `{"type": "polygon", "points": [...]}` and a `seg_labels/` writer without changing this layout.

## Tests

```bash
python -m pytest tests
```
