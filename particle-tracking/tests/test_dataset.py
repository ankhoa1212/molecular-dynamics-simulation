import json
import warnings
from pathlib import Path

import pytest

from dataset import DatasetSplits, split_by_experiment


def _make_coco(images, annotations):
    return {
        "info": {},
        "licenses": [],
        "categories": [{"id": 1, "name": "particle", "supercategory": ""}],
        "images": images,
        "annotations": annotations,
    }


@pytest.fixture
def sample_dataset(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    filenames = [
        "trial1_frame_000.png",
        "trial1_frame_001.png",
        "trial2_frame_000.png",
        "trial3_frame_000.png",
    ]
    for name in filenames:
        (images_dir / name).write_bytes(b"fake-image")

    coco = _make_coco(
        images=[
            {"id": 1, "file_name": "trial1_frame_000.png", "width": 640, "height": 480},
            {"id": 2, "file_name": "trial1_frame_001.png", "width": 640, "height": 480},
            {"id": 3, "file_name": "trial2_frame_000.png", "width": 640, "height": 480},
            {"id": 4, "file_name": "trial3_frame_000.png", "width": 640, "height": 480},
        ],
        annotations=[
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0},
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [30, 30, 20, 20], "area": 400, "iscrowd": 0},
            {"id": 3, "image_id": 3, "category_id": 1, "bbox": [50, 50, 20, 20], "area": 400, "iscrowd": 0},
            {"id": 4, "image_id": 4, "category_id": 1, "bbox": [70, 70, 20, 20], "area": 400, "iscrowd": 0},
        ],
    )
    (tmp_path / "annotations.json").write_text(json.dumps(coco))
    return tmp_path


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def test_returns_dataset_splits_namedtuple(sample_dataset, tmp_path):
    result = split_by_experiment(
        dataset_path=sample_dataset,
        train_experiments=["trial1"],
        val_experiments=["trial2"],
        test_experiments=["trial3"],
        split_dir=tmp_path / "split",
    )
    assert isinstance(result, DatasetSplits)
    assert result.train_dir.exists()
    assert result.valid_dir.exists()
    assert result.test_dir.exists()


def test_images_assigned_to_correct_split(sample_dataset, tmp_path):
    splits = split_by_experiment(
        dataset_path=sample_dataset,
        train_experiments=["trial1"],
        val_experiments=["trial2"],
        test_experiments=["trial3"],
        split_dir=tmp_path / "split",
    )
    train_coco = _load(splits.train_dir / "_annotations.coco.json")
    valid_coco = _load(splits.valid_dir / "_annotations.coco.json")
    test_coco = _load(splits.test_dir / "_annotations.coco.json")

    assert len(train_coco["images"]) == 2
    assert len(valid_coco["images"]) == 1
    assert len(test_coco["images"]) == 1
    assert all("trial1" in img["file_name"] for img in train_coco["images"])
    assert all("trial2" in img["file_name"] for img in valid_coco["images"])
    assert all("trial3" in img["file_name"] for img in test_coco["images"])


def test_annotations_follow_their_images(sample_dataset, tmp_path):
    splits = split_by_experiment(
        dataset_path=sample_dataset,
        train_experiments=["trial1"],
        val_experiments=["trial2"],
        test_experiments=["trial3"],
        split_dir=tmp_path / "split",
    )
    train_coco = _load(splits.train_dir / "_annotations.coco.json")
    assert len(train_coco["annotations"]) == 2
    image_ids = {img["id"] for img in train_coco["images"]}
    assert all(ann["image_id"] in image_ids for ann in train_coco["annotations"])


def test_categories_preserved_in_every_split(sample_dataset, tmp_path):
    splits = split_by_experiment(
        dataset_path=sample_dataset,
        train_experiments=["trial1"],
        val_experiments=["trial2"],
        test_experiments=["trial3"],
        split_dir=tmp_path / "split",
    )
    for split_dir in (splits.train_dir, splits.valid_dir, splits.test_dir):
        coco = _load(split_dir / "_annotations.coco.json")
        assert coco["categories"] == [{"id": 1, "name": "particle", "supercategory": ""}]


def test_image_symlinks_created_in_split_dirs(sample_dataset, tmp_path):
    splits = split_by_experiment(
        dataset_path=sample_dataset,
        train_experiments=["trial1"],
        val_experiments=["trial2"],
        test_experiments=["trial3"],
        split_dir=tmp_path / "split",
    )
    assert (splits.train_dir / "trial1_frame_000.png").is_symlink()
    assert (splits.train_dir / "trial1_frame_001.png").is_symlink()
    assert (splits.valid_dir / "trial2_frame_000.png").is_symlink()
    assert (splits.test_dir / "trial3_frame_000.png").is_symlink()


def test_warns_on_unmatched_image(sample_dataset, tmp_path):
    coco_path = sample_dataset / "annotations.json"
    coco = json.loads(coco_path.read_text())
    coco["images"].append(
        {"id": 99, "file_name": "unknown_frame.png", "width": 640, "height": 480}
    )
    (sample_dataset / "images" / "unknown_frame.png").write_bytes(b"fake")
    coco_path.write_text(json.dumps(coco))

    with pytest.warns(UserWarning, match="unknown_frame.png"):
        split_by_experiment(
            dataset_path=sample_dataset,
            train_experiments=["trial1"],
            val_experiments=["trial2"],
            test_experiments=["trial3"],
            split_dir=tmp_path / "split",
        )


def test_split_dir_defaults_to_dataset_path_split(sample_dataset):
    split_by_experiment(
        dataset_path=sample_dataset,
        train_experiments=["trial1"],
        val_experiments=["trial2"],
        test_experiments=["trial3"],
    )
    assert (sample_dataset / "split" / "train").exists()
    assert (sample_dataset / "split" / "valid").exists()
    assert (sample_dataset / "split" / "test").exists()
