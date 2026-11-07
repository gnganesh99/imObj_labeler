# Design decisions

Decisions and constraints to keep in mind when changing the app. Each entry records its status: open, decided or implemented.

---

## 1. The train/val split must keep copies of the same source image together

**Status:** partly implemented (as of 2026-11-07): grouping by source file and split settings that survive registration are done (see entry 3); stable per-file assignments are not.

### Context

When "Overwrite same filename" is off, registering the same source file again saves extra copies: `CO_on_Cu_5960`, `CO_on_Cu_5960_1`, `CO_on_Cu_5960_2`, and so on. These are useful for the same scan rendered with different colormaps, flattening or contrast, or with corrected labels.

These copies are near-duplicates: the same molecules sit at the same positions. The current split in `Project.write_data_yaml` shuffles individual images. One copy can therefore end up in `train` and another in `val`. Validation would then test on images the model has effectively already seen, and mAP, precision and recall would look better than they really are. This is data leakage.

### Decision

Split by **source file**, not by saved image:

- Group the saved images by their source file. `register_log.csv` maps each `stem` to its `file`. A stem with no log row is a group of its own.
- Shuffle and split the groups, using the validation fraction and seed. Every copy in a group goes to the same set.
- The validation fraction counts groups, not images. Report both counts in the Dataset dialog, for example "18 train / 4 val sources (31 / 6 images)".

### Related: the split must stay stable and survive registration

These are two current problems in the same code:

1. Every registration calls `write_data_yaml()` with its defaults. That resets `data.yaml` to `train: images` / `val: images`, which silently discards a split the user wrote.
2. Writing the split again reshuffles everything. Images can move between train and val from one training run to the next, so results can't be compared.

The planned fix:

- Save the split settings (validation fraction and seed) and the group assignments in `obj_det_labels/`, for example in `split.json`.
- Re-apply them on every registration.
- Give each new source file a set (train or val) once, the first time it is registered, and never move it afterwards. Existing groups keep their set.
- New copies of an already-assigned source file join that file's set.
- Offer an explicit "re-split" action for when a full reshuffle is wanted.

### Where to change

- `imobj_labeler/project.py`: `write_data_yaml`, `register`, and a new loader for the split state.
- `imobj_labeler/web/app.js` and `index.html`: the Dataset dialog (counts, re-split).
- `tests/test_core.py`: tests that all copies of one source land in the same set, and that the split survives registration.

---

## 2. Which labels are shown: registered labels take precedence over existing ones

**Status:** implemented (as of 2026-11-07)

When the data folder is a YOLO `images/` folder with a sibling `../labels/` folder (for example `base_images_from_tr17/images` and `base_images_from_tr17/labels`), the app reads those labels as **existing** labels and shows them straight away.

- Existing labels are only ever read; the app never writes to `../labels/`.
- Editing and registering saves a new set to `<data folder>/obj_det_labels/images/` and `obj_det_labels/labels/`.
- Once a file is registered, the registered labels take precedence for display (the latest version, if there are several). The existing labels are no longer shown for that file.
- **Revert to existing labels** removes every registered version of the file (image and labels in `obj_det_labels/`), so the display falls back to the untouched `../labels/` file. It only appears when such a file exists. "Unregister" still removes just the current version.
- Sibling labels are only looked for when the data folder is literally named `images`.

---

## 3. Dataset source: existing, registered or combined

**Status:** implemented (as of 2026-11-07)

The Dataset dialog picks which images go into `data.yaml`: **combined** (the default), **registered** or **existing**. In combined, a file that has registered versions uses those and drops its existing image and labels.

- Nothing is copied. `train.txt` / `val.txt` list image paths: `./images/<stem>.png` for registered images, absolute paths for existing ones, which live outside `obj_det_labels/`. Ultralytics finds each label by replacing the last `/images/` in the path with `/labels/`, which works for both.
- The split is grouped by source file, which implements the grouping half of entry 1.
- The settings (source, validation fraction, seed) are saved in `obj_det_labels/split.json` and re-applied after every registration, so a registration no longer resets the split.
- **Still open from entry 1:** assignments aren't stored per file. Adding a file and re-applying the split reshuffles with the same seed, so some files can move between train and val. Fixing that needs stored per-file assignments.
- Ultralytics writes its `labels.cache` next to the first label folder it sees, so train and val may share one cache file and rebuild it on each run. This only costs time, not correctness.
