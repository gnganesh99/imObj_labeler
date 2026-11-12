from pathlib import Path

import numpy as np
import pytest

from imobj_labeler import annotations as ann
from imobj_labeler import classes as cls_cfg
from imobj_labeler import reader
from imobj_labeler.project import Project

ROOT = Path(__file__).resolve().parents[1]
SXM_DIR = ROOT / "to_ignore" / "data" / "new_sxm_images"
JPG_DIR = ROOT / "to_ignore" / "data" / "base_images_from_tr17" / "images"


def test_yolo_roundtrip_and_tolerant_parse():
    text = "0\t0.5\t0.5\t0.2\t0.1\n0.0 0.25 0.25 0.1 0.1\nbad line\n2 0.9 0.9 0.1 0.1 0.77\n"
    a = ann.parse_yolo(text, 200, 100)
    assert [x["cls"] for x in a] == [0, 0, 2]
    assert a[0]["x1"] == pytest.approx(80) and a[0]["y2"] == pytest.approx(55)
    assert a[2]["conf"] == pytest.approx(0.77)
    out = ann.format_yolo(a, 200, 100)
    assert out.splitlines()[0] == "0 0.500000 0.500000 0.200000 0.100000"
    assert "\t" not in out and "0.0 " not in out.split("\n")[1][:4]


def test_clip_drops_degenerate_boxes():
    assert ann.clip_bbox(ann.bbox(0, -10, -10, 0.5, 5), 10, 10) is None
    c = ann.clip_bbox(ann.bbox(0, -5, 2, 5, 20), 10, 10)
    assert (c["x1"], c["y2"]) == (0, 10)


def test_classes_validation():
    cfg = cls_cfg.validate({"classes": [{"id": 1, "name": "b"}, {"id": 0, "name": "a"}]})
    assert [c["name"] for c in cfg["classes"]] == ["a", "b"]
    with pytest.raises(ValueError):
        cls_cfg.validate({"classes": [{"id": 0, "name": "a"}, {"id": 0, "name": "b"}]})


def test_flatten_plane_removes_tilt():
    yy, xx = np.mgrid[0:20, 0:30]
    out = reader.apply_flatten(3 * xx - 2 * yy + 5.0, "plane")
    assert np.abs(out).max() < 1e-8


@pytest.mark.skipif(not SXM_DIR.is_dir(), reason="sample sxm data not present")
def test_sxm_first_channel_fallback():
    name = reader.list_files(SXM_DIR)[0]
    rgb, info = reader.render(SXM_DIR / name, reader.RenderParams(channel="nope"))
    assert info["fallback"] and info["channel"] == "Z forward"
    assert rgb.dtype == np.uint8 and rgb.shape == (256, 256, 3)


@pytest.mark.skipif(not JPG_DIR.is_dir(), reason="sample jpg data not present")
def test_existing_sibling_labels_are_read():
    p = Project(JPG_DIR)
    name = p.files()[0]
    rgb, info = reader.render(p.path_of(name), reader.RenderParams())
    labels = p.labels_for(name, info["width"], info["height"])
    assert labels and all(a["source"] == "existing" for a in labels)


def test_register_writes_dataset(tmp_path):
    from PIL import Image

    Image.fromarray(np.zeros((40, 60, 3), np.uint8)).save(tmp_path / "a.png")
    p = Project(tmp_path)
    rgb, info = reader.render(p.path_of("a.png"), reader.RenderParams())
    res = p.register("a.png", rgb, [ann.bbox(0, 6, 4, 18, 12)], reader.RenderParams())
    assert res["n_objects"] == 1 and p.is_registered("a.png")
    out = tmp_path / "obj_det_labels"
    assert (out / "images" / "a.png").is_file()
    assert (out / "labels" / "a.txt").read_text().startswith("0 0.200000 0.200000")
    assert "names:" in (out / "data.yaml").read_text()
    back = p.labels_for("a.png", 60, 40)
    assert back[0]["x1"] == pytest.approx(6) and back[0]["source"] == "registered"


def test_register_versions_without_overwrite(tmp_path):
    from PIL import Image

    Image.fromarray(np.zeros((40, 60, 3), np.uint8)).save(tmp_path / "b.png")
    p = Project(tmp_path)
    rgb, _ = reader.render(p.path_of("b.png"), reader.RenderParams())
    rp = reader.RenderParams()
    one, two = [ann.bbox(0, 1, 1, 10, 10)], [ann.bbox(0, 1, 1, 10, 10)] * 2

    assert p.register("b.png", rgb, one, rp, overwrite=False)["stem"] == "b"
    r = p.register("b.png", rgb, two, rp, overwrite=False)
    assert r["stem"] == "b_1" and not r["replaced"] and r["versions"] == ["b", "b_1"]
    assert p.register("b.png", rgb, one, rp, overwrite=False)["stem"] == "b_2"
    assert p.current_stem("b.png") == "b_2" and len(p.labels_for("b.png", 60, 40)) == 1

    # overwrite replaces the current (latest) version only
    r = p.register("b.png", rgb, two, rp, overwrite=True)
    assert r["stem"] == "b_2" and r["replaced"] and len(r["versions"]) == 3
    assert (tmp_path / "obj_det_labels" / "labels" / "b.txt").read_text().count("\n") == 1

    # unregister removes the current version; the previous one becomes current
    assert p.unregister("b.png") and p.current_stem("b.png") == "b_1"
    assert p.status() == {"b.png": True}


def test_revert_to_existing_labels(tmp_path):
    from PIL import Image

    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    Image.fromarray(np.zeros((40, 60, 3), np.uint8)).save(tmp_path / "images" / "c.png")
    (tmp_path / "labels" / "c.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    p = Project(tmp_path / "images")
    rgb, _ = reader.render(p.path_of("c.png"), reader.RenderParams())
    rp = reader.RenderParams()
    assert p.existing_label_path("c.png") and p.labels_for("c.png", 60, 40)[0]["source"] == "existing"

    two = [ann.bbox(0, 1, 1, 10, 10)] * 2
    p.register("c.png", rgb, two, rp, overwrite=False)
    p.register("c.png", rgb, two, rp, overwrite=False)
    assert p.versions("c.png") == ["c", "c_1"]
    assert p.labels_for("c.png", 60, 40)[0]["source"] == "registered"  # registered wins

    assert p.unregister("c.png", all_versions=True)
    assert p.versions("c.png") == [] and not p.is_registered("c.png")
    back = p.labels_for("c.png", 60, 40)
    assert len(back) == 1 and back[0]["source"] == "existing"
    assert (tmp_path / "labels" / "c.txt").is_file()  # existing labels untouched


def test_dataset_sources_and_grouped_split(tmp_path):
    from PIL import Image

    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    for i in range(6):
        Image.fromarray(np.zeros((40, 60, 3), np.uint8)).save(tmp_path / "images" / f"e{i}.jpg")
        (tmp_path / "labels" / f"e{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    p = Project(tmp_path / "images")
    rp = reader.RenderParams()
    rgb, _ = reader.render(p.path_of("e0.jpg"), rp)
    box = [ann.bbox(0, 1, 1, 10, 10)]
    p.register("e0.jpg", rgb, box, rp, overwrite=False)
    p.register("e0.jpg", rgb, box, rp, overwrite=False)  # e0, e0_1: one source file

    out = tmp_path / "images" / "obj_det_labels"
    lines = lambda f: (out / f).read_text().split()

    r = p.write_data_yaml(0.5, seed=1, source="combined")
    assert (r["n_registered"], r["n_existing"], r["n_overridden"]) == (1, 6, 1)
    allx = lines("train.txt") + lines("val.txt")
    assert len(allx) == 7 and not any(x.endswith("e0.jpg") for x in allx)  # registered wins
    reg_sets = [f for f in ("train.txt", "val.txt") if any(x.startswith("./") for x in lines(f))]
    assert len(reg_sets) == 1  # both registered versions of e0 land in the same set
    assert r["groups_train"] + r["groups_val"] == 6

    assert p.write_data_yaml(0.0, source="existing")["n_train"] == 6
    assert lines("train.txt") == [(tmp_path / "images" / f"e{i}.jpg").as_posix() for i in range(6)]
    assert not (out / "val.txt").exists() and "val: train.txt" in (out / "data.yaml").read_text()
    assert p.write_data_yaml(0.0, source="registered")["n_train"] == 2

    # a registration re-applies the saved settings instead of resetting them
    p.write_data_yaml(0.5, seed=1, source="combined")
    p.register("e3.jpg", rgb, box, rp)
    assert p.split_settings() == {"source": "combined", "val_fraction": 0.5, "seed": 1}
    assert "val: val.txt" in (out / "data.yaml").read_text()
    with pytest.raises(ValueError):
        p.write_data_yaml(source="nope")


@pytest.mark.parametrize("t", ["rot90", "rot180", "rot270"])
def test_rotation_moves_boxes_with_the_image(t):
    from imobj_labeler import augment

    img = np.zeros((40, 60), np.uint8)
    img[5:15, 30:50] = 255  # bright block, box in pixels x 30..50, y 5..15
    box = (0, 40 / 60, 10 / 40, 20 / 60, 10 / 40)
    out, rows = augment.apply(img, [box], t, np.random.default_rng(0))
    _, cx, cy, w, h = rows[0]
    H, W = out.shape
    ys, xs = np.nonzero(out)
    assert cx == pytest.approx((xs.min() + xs.max() + 1) / 2 / W)
    assert cy == pytest.approx((ys.min() + ys.max() + 1) / 2 / H)
    assert w == pytest.approx((xs.max() - xs.min() + 1) / W)
    assert h == pytest.approx((ys.max() - ys.min() + 1) / H)


def test_augmented_copies_follow_their_source(tmp_path):
    from PIL import Image

    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    for i in range(10):
        Image.fromarray(np.full((40, 60), 100, np.uint8)).save(tmp_path / "images" / f"e{i}.png")
        (tmp_path / "labels" / f"e{i}.txt").write_text("0\t0.5\t0.5\t0.2\t0.2\n")
    p = Project(tmp_path / "images")
    out = p.out
    lines = lambda f: (out / f).read_text().split()

    p.write_data_yaml(0.3, seed=2, source="existing")
    r = p.augment(["rot90", "noise"], noise_sigma=5, train=True, val=True)
    assert r["n_aug_written"] == 20 and (r["n_train_aug"], r["n_val_aug"]) == (14, 6)
    assert (out / "aug" / "images" / "e0__rot90.png").is_file()
    assert (out / "aug" / "labels" / "e0__rot90.txt").is_file()
    val_src = {Path(x).stem for x in lines("val.txt") if "/aug/" not in x}
    val_aug = {Path(x).stem.split("__")[0] for x in lines("val.txt") if "/aug/" in x}
    assert val_aug == val_src and lines("val_aug.txt") == [x for x in lines("val.txt") if "/aug/" in x]
    assert "test: val_aug.txt" in (out / "data.yaml").read_text()

    # re-split with another seed: copies move with their source, never leak
    p.write_data_yaml(0.3, seed=5)
    val_src = {Path(x).stem for x in lines("val.txt") if "/aug/" not in x}
    train_aug = {Path(x).stem.split("__")[0] for x in lines("train.txt") if "/aug/" in x}
    assert val_src and not (val_src & train_aug)

    # editing a source label makes its copies stale (skipped) until augmentation is re-run
    import os, time
    lab = tmp_path / "labels" / "e1.txt"
    lab.write_text("0 0.4 0.4 0.2 0.2\n")
    os.utime(lab, ns=(time.time_ns() + 10**9,) * 2)
    assert p.write_data_yaml()["n_aug_stale"] == 2

    p.remove_augmentation()
    assert not (out / "aug").exists() and not any("/aug/" in x for x in lines("train.txt"))
    assert "test:" not in (out / "data.yaml").read_text()


def test_rotated_copies_get_noise_with_probability(tmp_path):
    from PIL import Image

    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    for i in range(20):
        Image.fromarray(np.full((40, 60), 100, np.uint8)).save(tmp_path / "images" / f"e{i}.png")
        (tmp_path / "labels" / f"e{i}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    p = Project(tmp_path / "images")
    p.write_data_yaml(0.0, seed=3, source="existing")

    def names(prob):
        p.augment(["rot90", "rot180"], noise_sigma=8, rot_noise_p=prob)
        return sorted(f.name for f in (p.aug_dir / "images").iterdir())

    plain = names(0.0)
    assert len(plain) == 40 and not any("noise" in n for n in plain)
    assert all("_noise" in n for n in names(1.0))
    half = names(0.5)
    noisy = [n for n in half if n.endswith("_noise.png")]
    assert len(half) == 40 and 8 <= len(noisy) <= 32
    assert names(0.5) == half  # same seed, same draw
    img = np.asarray(Image.open(p.aug_dir / "images" / noisy[0]))
    assert img.shape == (60, 40) and img.std() > 3  # rotated and noisy
    assert p.split_settings()["aug"]["rot_noise_p"] == 0.5


def _small_dataset(tmp_path, n=6):
    from PIL import Image

    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    for i in range(n):
        Image.fromarray(np.full((40, 60), 100, np.uint8)).save(tmp_path / "images" / f"e{i}.png")
        (tmp_path / "labels" / f"e{i}.txt").write_text("0.0\t0.5\t0.5\t0.2\t0.2\n")
    return Project(tmp_path / "images")


def test_export_bundle_layout(tmp_path):
    import json

    p = _small_dataset(tmp_path)
    p.write_data_yaml(0.34, seed=0, source="existing")
    p.augment(["rot90"], val=True, train=True)
    model = tmp_path / "start.pt"
    model.write_bytes(b"fake")
    r = p.export_bundle(tmp_path / "bundle", {"model": str(model), "epochs": 3})
    b = tmp_path / "bundle"
    assert (r["train"], r["val"], r["train_aug"], r["val_aug"]) == (8, 4, 4, 2)
    assert len(list((b / "images" / "train").iterdir())) == 8
    assert len(list((b / "labels" / "val").iterdir())) == 4
    assert (b / "labels" / "train" / f"{next((b / 'images' / 'train').iterdir()).stem}.txt").is_file()
    lab = next((b / "labels" / "val").iterdir()).read_text()
    assert lab.startswith("0 ") and "\t" not in lab  # normalised
    y = (b / "data.yaml").read_text()
    assert not any(line.startswith("path:") for line in y.splitlines())
    assert "train: images/train" in y and "test: val_aug.txt" in y
    assert all(x.startswith("./images/val/") for x in (b / "val_aug.txt").read_text().split())
    cfg = json.loads((b / "train_config.json").read_text())
    assert cfg["model"] == "weights_init/start.pt" and (b / "weights_init" / "start.pt").is_file()
    assert cfg["epochs"] == 3 and cfg["imgsz"] == 256 and cfg["format_version"] == 1
    man = json.loads((b / "manifest.json").read_text())
    assert len(man["files"]) == 12 and man["split"]["seed"] == 0
    with pytest.raises(FileExistsError):
        p.export_bundle(tmp_path / "bundle", {})


def test_train_script_checks_bundle(tmp_path):
    import subprocess, sys

    pytest.importorskip("ultralytics")
    p = _small_dataset(tmp_path)
    p.write_data_yaml(0.34, seed=0, source="existing")
    p.export_bundle(tmp_path / "bundle", {})
    script = Path(__file__).resolve().parents[1] / "imobj_labeler" / "train_yolo.py"
    run = lambda *a: subprocess.run([sys.executable, str(script), *a], capture_output=True, text=True)
    assert "1.0" in run("--version").stdout
    r = run(str(tmp_path / "bundle"), "--check", "--allow-cpu")
    assert r.returncode == 0, r.stderr
    assert "train: 4 images" in r.stdout and "val  : 2 images" in r.stdout
    cfg = tmp_path / "bundle" / "train_config.json"
    cfg.write_text(cfg.read_text().replace('"format_version": 1', '"format_version": 99'))
    r = run(str(tmp_path / "bundle"), "--check", "--allow-cpu")
    assert r.returncode == 2 and "not supported" in r.stderr


def test_remote_quoting_and_sections():
    from imobj_labeler import remote

    assert remote.rq("~/bundles") == '"$HOME"/bundles'
    assert remote.rq("~/my runs") == "\"$HOME\"/'my runs'"
    assert remote.rq("/opt/x;rm -rf") == "'/opt/x;rm -rf'"
    assert remote.sections("junk\n@@a\n1\n2\n@@b\n") == {"a": "1\n2\n", "b": ""}
    for bad in ("-oProxyCommand=x", "a b", ""):
        with pytest.raises(remote.RemoteError):
            remote.Server(bad)
    with pytest.raises(remote.RemoteError):
        remote.check_name("../x")


GIT_BASH = Path(r"C:\Program Files\Git\usr\bin\bash.exe")


@pytest.fixture
def fake_server(tmp_path, monkeypatch):
    """A local bash pretending to be the GPU server (home = tmp_path/home)."""
    import shutil, sys
    from imobj_labeler import remote

    bash = str(GIT_BASH) if GIT_BASH.is_file() else shutil.which("bash")
    if not bash:
        pytest.skip("bash not available")
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("FAKE_SSH_HOME", home.as_posix())
    monkeypatch.setenv("FAKE_SSH_BASH", bash)
    monkeypatch.setattr(remote, "SSH_COMMAND", [sys.executable, str(Path(__file__).with_name("fake_ssh.py"))])
    return remote.Server("gpuvm", "~/bundles", Path(sys.executable).as_posix()), home


def test_remote_upload_check_status(tmp_path, fake_server):
    pytest.importorskip("ultralytics")
    from imobj_labeler import remote

    srv, home = fake_server
    p = _small_dataset(tmp_path)
    p.write_data_yaml(0.34, seed=0, source="existing")
    b = p.export_bundle(tmp_path / "out" / "b1", {"device": "cpu"})

    info = remote.test(srv)
    assert info["script_version"] is None and info["ultralytics"] and not info["root_exists"]
    assert remote.ensure_script(srv).startswith("uploaded")
    assert "already" in remote.ensure_script(srv)
    assert (home / "bundles" / "train_yolo.py").read_bytes() == remote.SCRIPT.read_bytes()

    up = remote.upload(srv, Path(b["path"]))
    assert up["name"] == "b1" and (home / "bundles" / "b1" / "images" / "train").is_dir()
    with pytest.raises(remote.RemoteError, match="already exists"):
        remote.upload(srv, Path(b["path"]))
    assert "OK" in remote.check(srv, "b1")
    assert remote.list_bundles(srv) == [{"name": "b1", "state": "not started", "run": None,
                                         "time": remote.list_bundles(srv)[0]["time"]}]
    st = remote.status(srv, "b1")
    assert st["state"] == "not started" and st["epochs"] == 300 and not st["running"]

    # a finished run, as train_yolo.py leaves it: status parses results.csv
    run = home / "bundles" / "b1" / "runs" / "train_x"
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "best.pt").write_bytes(b"pt")
    (run / "weights" / "epoch0.pt").write_bytes(b"pt")
    (run / "results.csv").write_text(
        "  epoch,  metrics/mAP50(B),  metrics/mAP50-95(B),  val/box_loss\n"
        "      1,  0.5,  0.3,  1.2\n      2,  0.7,  0.45,  1.0\n      3,  0.6,  0.4,  1.1\n")
    (run / "training_data.json").write_text('{"ok": true}')
    st = remote.status(srv, "b1")
    assert st["state"] == "finished" and st["epoch"] == 3 and st["best"]["epoch"] == 2
    assert st["has_best"] and st["run"] == "train_x"
    got = remote.fetch(srv, "b1", tmp_path / "models")
    assert Path(got["best"]).is_file() and not (Path(got["path"]) / "weights" / "epoch0.pt").exists()

    with pytest.raises(remote.RemoteError, match="Cannot connect"):
        remote.test(remote.Server("unreachable"))


def test_eval_sets_and_yaml(tmp_path):
    p = _small_dataset(tmp_path, n=6)
    assert set(p.eval_sets()) == {"all"}  # no split yet
    p.write_data_yaml(0.34, seed=0, source="existing")
    p.augment(["rot90"], train=True, val=True)
    sets = p.eval_sets()
    assert {k: len(v) for k, v in sets.items()} == {"all": 6, "val": 2, "train": 4, "val_aug": 2}
    assert not any("__" in x.name for x in sets["val"] + sets["train"])  # clean sets: no copies
    y, n = p.eval_yaml("val_aug")
    assert n == 2 and "val: val_aug.txt" in y.read_text()
    assert all(Path(x).is_file() for x in (y.parent / "val_aug.txt").read_text().split())
    with pytest.raises(ValueError):
        p.eval_yaml("nope")


def test_run_info_reads_run_folder(tmp_path):
    from imobj_labeler.evaluate import run_info

    run = tmp_path / "train_x"
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "best.pt").write_bytes(b"x")
    (run / "args.yaml").write_text("model: weights_init/start.pt\nepochs: 50\nimgsz: 256\nbatch: 16\n")
    (run / "results.csv").write_text("  epoch,  metrics/mAP50(B),  metrics/mAP50-95(B)\n"
                                     "  1, 0.5, 0.3\n  2, 0.8, 0.6\n  3, 0.7, 0.5\n")
    (run / "training_data.json").write_text(
        '{"training_end_time": "2026-11-10_10-00-00", "bundle": "/b/x_bundle",'
        ' "environment": {"gpu": "A100", "ultralytics": "8.3.40"}}')
    i = run_info(run / "weights" / "best.pt")
    assert (i["run"], i["epochs"], i["epochs_done"], i["imgsz"]) == ("train_x", 50, 3, 256)
    assert i["best"]["epoch"] == 2 and i["start_model"] == "start.pt"
    assert (i["gpu"], i["ultralytics"], i["bundle"]) == ("A100", "8.3.40", "x_bundle")
    assert run_info(tmp_path / "train_x" / "weights" / "best.pt")["file"] == "best.pt"


MODEL68 = ROOT / "to_ignore" / "yolo_models" / "train68" / "weights" / "best.pt"


@pytest.mark.skipif(not MODEL68.is_file(), reason="sample model not present")
def test_evaluate_model_records_history(tmp_path):
    pytest.importorskip("ultralytics")
    p = _small_dataset(tmp_path, n=4)
    r = p.evaluate_model(str(MODEL68), "all")
    assert r["n_images"] == 4 and 0 <= r["map"] <= 1 and r["imgsz"] == 256
    assert p.eval_history()[-1]["model"] == str(MODEL68)
