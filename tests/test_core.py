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
