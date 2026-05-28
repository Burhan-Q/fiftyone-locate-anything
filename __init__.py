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

Eagle JSONL dataset import: see README for the importlib-based loader (the
zoo source's on-disk directory contains characters that aren't valid in a
Python identifier, so direct `import` won't parse).
"""

from typing import Any

from fiftyone.operators import types
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


def resolve_input(model_name: str, ctx: Any) -> types.Property:
    """FiftyOne App operator UI for LocateAnything."""
    inputs = types.Object()

    inputs.enum(
        "media_type",
        values=["image", "video"],
        default="image",
        label="Media Type",
    )

    inputs.enum(
        "operation",
        values=[
            "detect",
            "grounding",
            "point",
            "scene_text",
            "layout",
            "text_grounding",
            "gui_box",
        ],
        default="detect",
        label="Operation",
        description=(
            "detect: provide classes. grounding/point/text_grounding/gui_box: "
            "provide prompt. scene_text: no prompt. layout: optional classes "
            "(defaults to title/paragraph/figure/table)."
        ),
    )

    inputs.str(
        "classes",
        default=None,
        required=False,
        label="Classes (comma-separated)",
        description="Used for detect and layout operations.",
    )

    inputs.str(
        "prompt",
        default=None,
        required=False,
        label="Prompt",
        description=("Free-form phrase for grounding/point/text_grounding/gui_box."),
    )

    inputs.bool(
        "single_instance",
        default=False,
        label="Single Instance (grounding only)",
    )

    inputs.enum(
        "generation_mode",
        values=["hybrid", "fast", "slow"],
        default="hybrid",
        label="Generation Mode",
    )

    inputs.int("max_new_tokens", default=2048, label="Max New Tokens")
    inputs.bool("do_sample", default=True, label="Use Sampling")
    inputs.float("temperature", default=0.7, label="Temperature")
    inputs.float("top_p", default=0.9, label="Top-p")
    inputs.float("repetition_penalty", default=1.1, label="Repetition Penalty")

    inputs.int(
        "frames",
        default=8,
        label="Video: # Sampled Frames",
        description="Number of evenly-spaced frames per video.",
    )
    inputs.float(
        "fps",
        default=None,
        required=False,
        label="Video: Target FPS",
        description="Overrides frame count.",
    )

    return types.Property(inputs)
