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

---

## 4. Offline augmentation: exact boxes only, copies follow their source

**Status:** implemented (as of 2026-11-07)

The Dataset dialog can write saved augmented copies, created only when the user presses the button and never on registration.

- **Transforms:** 90° rotations, Gaussian noise and blur. Arbitrary-angle rotation is left out on purpose: YOLO boxes are axis-aligned, so after a rotation the box enclosing the rotated corners is too large (about 40% at 45°). Flips, zoom, crops, mosaic and brightness are better done on the fly by ultralytics, which gives new randomness every epoch.
- **Folders:** `obj_det_labels/aug/images` and `aug/labels`, not `aug_images/` and `aug_labels/`. Ultralytics finds a label by replacing the last `/images/` in the image path with `/labels/`. With `aug_images/` it would replace an earlier `/images/` in the path and look in the wrong place.
- **No leakage:** `aug/manifest.json` records each copy's source image and source-file group. When the lists are written, each copy goes to the set its source is currently in. A re-split therefore can't put a copy of a val image into train.
- **Staleness:** the manifest stores the source image and label modification times. Copies whose source changed, or left the dataset, are skipped and counted until augmentation is run again.
- **Val copies:** listed in `val.txt` together with the clean val images (what the user asked for), and on their own in `val_aug.txt` as `test:`, so the noisy-image score can be read separately with `split=test`. Note that `best.pt` is then chosen on clean and augmented val together.
- The noise RNG is seeded with the split seed, so the same settings give the same copies.

---

## 5. Training runs outside the app: data bundles plus one common train_yolo.py

**Status:** steps 1 and 2 implemented (as of 2026-11-09): export, plus upload, start, watch, stop/resume and download over SSH. Step 2 was tested against a local stand-in server, not yet on the real VM. "Import run" (for runs started by hand) is planned.

### In plain words

- **Step 1 (export) only prepares a folder.** It doesn't train. **Train… → Export bundle** copies everything training needs into a new folder:
  - `images/train`, `images/val`, `labels/train`, `labels/val`. Augmented copies sit next to the originals as `x__rot90.png`; there is no separate `aug/` folder.
  - `data.yaml`, `train_config.json` (the settings), `manifest.json` (where every file came from), `val_aug.txt`, and `weights_init/` when starting from a local `.pt`.
  - The original data and `obj_det_labels/` are not changed.
- **Where it goes:** `<data folder>/obj_det_labels/bundles/<name>/` by default, or wherever Save in points. The last location is remembered. It stays on this PC until copied to the VM, e.g. with `scp`.
- **Loggers:** online experiment trackers (wandb, Comet, ClearML, MLflow…) that upload metrics to a website when installed. `train_yolo.py` switches them off. Ultralytics' local logs stay on: `results.csv` (one line per epoch), `train.log` (the console output, via `nohup … > train.log`), `last.pt` / `best.pt` and the plots.
- **Today the training is started by hand:** scp, ssh, nohup.
- **Step 2 (implemented):** the app does the same over SSH:
  1. Upload the bundle.
  2. Start training detached, so it survives the app closing, the PC sleeping or the network dropping.
  3. Only read `results.csv` and `train.log` to show progress. This never talks to the training process, so it can't slow or interrupt it.
  4. Download `best.pt` and the run folder when it's done.

- **Separate process:** training takes hours, so it runs as a separate process on the GPU machine. Closing or restarting the app never touches a run, and `--resume` continues from `last.pt` after an interruption.
- **Bundles hold data and config, not code.** One versioned `imobj_labeler/train_yolo.py` trains any bundle. The bundle's `format_version` and the script's `SUPPORTED_FORMATS` must agree, and the script refuses mismatches. Each run copies the script and the config it used into the run folder, so every `best.pt` can be traced back to exactly what produced it.
- **Standard YOLO layout, files copied:** the app's `train.txt` uses absolute Windows paths for existing images, which can't work on a Linux server. Copying also gives train and val their own `labels.cache`.
- **No `path:` in the bundle's data.yaml:** ultralytics resolves a relative `path:` against its global datasets folder, not the yaml's folder. Without the key, it uses the yaml's own folder, provided the yaml is passed as an absolute path, which `train_yolo.py` does.
- **No silent CPU fallback:** without CUDA the script stops instead of training for days on CPU.
- **Online loggers off by default:** an installed wandb starts automatically, can wait for a login on a detached run, and rejects an absolute project folder. They are switched off in memory with `dict.__setitem__`, because `SETTINGS[...] = ...` would rewrite ultralytics' global settings file.

### Step 2 details (SSH)

- **System ssh only:** the app calls this PC's `ssh` with `BatchMode=yes`, so it uses the user's keys and `~/.ssh/config` and fails instead of prompting. No password is ever handled or stored, and no port forwarding or server software is needed. Host names are checked so they can't be read as ssh options, and remote paths are shell-quoted (a leading `~/` stays expandable).
- **One folder on the server:** `<root>/train_yolo.py` and one subfolder per bundle. Each run is started with `nohup … < /dev/null &`, its PID is saved in `train.pid` and its output appended to `train.log`. The `;` before `nohup` matters: `a && b &` would background a subshell, and `$!` would be the wrong process.
- **Progress is read-only:** each check is one short ssh call that reads `train.pid` (`kill -0`), `results.csv`, `training_data.json` and the tail of `train.log`. It never talks to the training process. Output sections are separated by `@@name` lines, each preceded by an `echo`, so a file without a final newline can't swallow the next marker.
- **Script updates by checksum, not version:** a changed `train_yolo.py` with the same version number was not re-uploaded during testing. The app now compares `sha256sum` and uploads whenever the contents differ.
- **Bundle files are written with LF line endings,** since they go to Linux.
- **Readable log view:** `train.log` comes from a file, so tqdm writes every progress-bar update as its own line. The app shows each bar once, at its latest state (consecutive updates of the same epoch or validation bar are collapsed), and strips colour codes. The file on the server is unchanged. A restart begins the `=== start` marker on a new line, even after a run was stopped mid-bar.
- **Testing:** `tests/fake_ssh.py` stands in for ssh (a local bash with its own home folder). `IMOBJ_SSH` (a JSON list) can point the app at it, or at another ssh client.

---

## 6. Settled small items

**Status:** decided (as of 2026-11-10)

- **No automatic re-prediction when display settings change.** Changing the colormap, flatten or contrast doesn't re-run the model. Prediction runs only on request: Predict, or auto-predict when a new, unlabelled file is opened in Prediction mode. This is deliberate: the user predicts explicitly after choosing the display.
- **The native Windows folder dialog works.** The user confirmed it (tkinter `askdirectory` on the machine running the app). The in-app picker remains the fallback for remote or headless servers and for `--no-native-dialog`.

---

## 7. Testing models: local evaluation on sets from the current split

**Status:** implemented (as of 2026-11-10)

- **Where:** models are tested in the app, on this PC, with ultralytics `model.val` on CPU. That's fast enough for a few hundred images (about 5 s per set for a YOLOv10s at 256 px). It needs no GPU or server, and it tests the downloaded `best.pt` in the same ultralytics version that Prediction mode uses, which also confirms that a server-trained model loads in the app.
- **Sets** come from the current split (Dataset…):
  - clean val, the honest score;
  - augmented val copies only, for robustness;
  - clean train, to compare against val for over-fitting;
  - all labelled images.

  Augmented copies are kept out of the clean sets, so the "clean val" number can be compared across augmentation settings.
- **Image size:** each model is evaluated at its own training image size, read from its checkpoint.
- **History:** results are appended to `obj_det_labels/evaluations.json`, together with the split and augmentation settings in force. Numbers measured on different splits can therefore be told apart.
- **Training summary:** read from the run folder around each model (`args.yaml`, `results.csv`, and `training_data.json` when trained with `train_yolo.py`). It works for older runs too. A different ultralytics major.minor version than the app's is flagged.
