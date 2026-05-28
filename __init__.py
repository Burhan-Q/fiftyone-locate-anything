"""LocateAnything-3B FiftyOne Zoo Model.

NVIDIA Eagle's open-vocabulary grounding VLM as a FiftyOne Zoo remote model.
Supports image (7 operations) and video (frame-sampled) inference.

License notice: LocateAnything-3B weights are released under the
NVIDIA License — non-commercial research only.

Usage:
    import fiftyone.zoo as foz

    model = foz.load_zoo_model(
        "nvidia/LocateAnything-3B",
        operation="detect",
        classes=["car", "person"],
    )
    dataset.apply_model(model, label_field="detections")

Eagle JSONL dataset import:
    from fo_locate_anything.dataset import load_eagle_jsonl
    ds = load_eagle_jsonl("annotations.jsonl", "images/")
"""

from typing import Any

from huggingface_hub import snapshot_download

from .zoo import (
    LocateAnythingConfig,
    LocateAnythingImageModel,
    LocateAnythingVideoModel,
)


def download_model(model_name: str, model_path: str) -> None:
    """Download the LocateAnything model weights from HuggingFace."""
    snapshot_download(repo_id=model_name, local_dir=model_path)


def load_model(
    model_name: str | None = None,
    model_path: str | None = None,
    **kwargs: Any,
):
    """Load LocateAnything-3B for use with FiftyOne."""
    if model_path is None:
        model_path = "nvidia/LocateAnything-3B"

    # The App's resolve_input form passes `classes` as a comma-separated string.
    # Programmatic users pass a list. Normalize to list[str] | None.
    classes = kwargs.get("classes")
    if isinstance(classes, str):
        kwargs["classes"] = [c.strip() for c in classes.split(",") if c.strip()]

    config_dict = {"model_path": model_path, "model_name": model_name, **kwargs}
    config = LocateAnythingConfig(config_dict)
    if config.media_type == "video":
        return LocateAnythingVideoModel(config)
    return LocateAnythingImageModel(config)


# resolve_input is defined at the bottom of this file (Task 11 in the plan).
