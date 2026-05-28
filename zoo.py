"""LocateAnything zoo model implementation.

See .ref/2026-05-28-fo-locate-anything-design.md for full architecture.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Literal

import fiftyone as fo
import fiftyone.core.labels as fol
import fiftyone.core.models as fom
import fiftyone.utils.torch as fout
from fiftyone.core.models import SupportsGetItem, TorchModelMixin
from fiftyone.utils.torch import ImageGetItem

logger = logging.getLogger(__name__)


VALID_OPERATIONS = {
    "detect",
    "grounding",
    "point",
    "scene_text",
    "layout",
    "text_grounding",
    "gui_box",
}
LAYOUT_DEFAULT_CLASSES = ["title", "paragraph", "figure", "table"]


# ============================================================================
# Config
# ============================================================================


class LocateAnythingConfig(fout.TorchImageModelConfig):
    """Config for LocateAnything-3B (image and video)."""

    def __init__(self, d: dict[str, Any]) -> None:
        if "raw_inputs" not in d:
            d["raw_inputs"] = True
        super().__init__(d)

        self.model_path = self.parse_string(
            d, "model_path", default="nvidia/LocateAnything-3B"
        )
        self.model_name = d.get("model_name") or self.model_path
        self.media_type = self.parse_string(d, "media_type", default="image")
        if self.media_type not in ("image", "video"):
            raise ValueError(
                f"Invalid media_type '{self.media_type}'. Must be 'image' or 'video'."
            )

        self.operation = self.parse_string(d, "operation", default="detect")
        if self.operation not in VALID_OPERATIONS:
            raise ValueError(
                f"Invalid operation '{self.operation}'. "
                f"Must be one of {sorted(VALID_OPERATIONS)}"
            )

        # Detect/layout class list (raw dict access; no list parser available)
        self.classes = d.get("classes")
        self.prompt = self.parse_string(d, "prompt", default=None)
        self.single_instance = self.parse_bool(d, "single_instance", default=False)

        # Generation: defaults verbatim from locateanything_worker.py
        self.generation_mode = self.parse_string(d, "generation_mode", default="hybrid")
        self.max_new_tokens = self.parse_number(d, "max_new_tokens", default=2048)
        self.do_sample = self.parse_bool(d, "do_sample", default=True)
        self.temperature = self.parse_number(d, "temperature", default=0.7)
        self.top_p = self.parse_number(d, "top_p", default=0.9)
        self.repetition_penalty = self.parse_number(
            d, "repetition_penalty", default=1.1
        )

        # Video-only sampling controls
        self.frames = self.parse_number(d, "frames", default=8)
        self.fps = d.get("fps")
        self.every_nth = d.get("every_nth")


# ============================================================================
# Output parsing
# ============================================================================
#
# Box / point regexes are VERBATIM from Eagle/Embodied/locateanything_worker.py
# (lines 149 and 163). The two patterns are non-overlapping: the point pattern
# requires </box> after the 2nd coord, which a 4-coord box doesn't have.

_BOX_RE = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")
_POINT_RE = re.compile(r"<box><(\d+)><(\d+)></box>")
_REF_TAG_RE = re.compile(r"<ref>([^<]*)</ref>")  # FO-only: extract labels
_NONE_RE = re.compile(r"<box>\s*none\s*</box>", re.IGNORECASE)


def _nearest_preceding_ref(text: str, before_pos: int) -> str | None:
    """Find the most recent <ref>...</ref> label before `before_pos`."""
    label = None
    for m in _REF_TAG_RE.finditer(text, endpos=before_pos):
        candidate = m.group(1).strip()
        if candidate:
            label = candidate
    return label


def parse_output(raw: str, mode: Literal["box", "point"]) -> list[dict[str, Any]]:
    """Parse raw model output into structured entries.

    Each entry: {"label": str | None, "coords": tuple[int, ...]}.
    Returns [] for `<box>none</box>` or no valid matches.

    `mode="box"` matches 4-int coords (worker's parse_boxes regex).
    `mode="point"` matches 2-int coords (worker's parse_points regex).
    Coords are validated to lie in [0, 1000].
    """
    if not raw:
        return []

    pattern = _BOX_RE if mode == "box" else _POINT_RE
    num_coords = 4 if mode == "box" else 2
    results: list[dict[str, Any]] = []

    for m in pattern.finditer(raw):
        coords = tuple(int(m.group(i)) for i in range(1, 1 + num_coords))
        if not all(0 <= c <= 1000 for c in coords):
            continue
        label = _nearest_preceding_ref(raw, m.start())
        results.append({"label": label, "coords": coords})

    return results


# ============================================================================
# Label converters
# ============================================================================


def to_detections(
    parsed: list[dict[str, Any]], default_label: str = "object"
) -> fo.Detections:
    """Convert parsed box entries to fo.Detections.

    Coords are integer xyxy in [0, 1000]; converts to FiftyOne's [x, y, w, h]
    in [0, 1]. Degenerate boxes (w <= 0 or h <= 0) are skipped.
    """
    dets: list[fo.Detection] = []
    for entry in parsed:
        coords = entry["coords"]
        if len(coords) != 4:
            continue
        x1, y1, x2, y2 = coords
        x, y = x1 / 1000.0, y1 / 1000.0
        w, h = (x2 - x1) / 1000.0, (y2 - y1) / 1000.0
        if w <= 0 or h <= 0:
            continue
        dets.append(
            fo.Detection(
                label=entry.get("label") or default_label,
                bounding_box=[x, y, w, h],
            )
        )
    return fo.Detections(detections=dets)


def to_keypoints(
    parsed: list[dict[str, Any]], default_label: str = "point"
) -> fo.Keypoints:
    """Convert parsed point entries to fo.Keypoints."""
    kps: list[fo.Keypoint] = []
    for entry in parsed:
        coords = entry["coords"]
        if len(coords) != 2:
            continue
        x, y = coords
        kps.append(
            fo.Keypoint(
                label=entry.get("label") or default_label,
                points=[[x / 1000.0, y / 1000.0]],
            )
        )
    return fo.Keypoints(keypoints=kps)


# ============================================================================
# Prompt builder
# ============================================================================
#
# All templates are VERBATIM from Embodied/locateanything_worker.py.
# Subtle: "detect" uses "matches" (singular verb); "grounding" multi uses
# "match" (plural). Preserve exactly.

_T_DETECT = "Locate all the instances that matches the following description: {}."
_T_GROUNDING_MULTI = (
    "Locate all the instances that match the following description: {}."
)
_T_GROUNDING_SINGLE = (
    "Locate a single instance that matches the following description: {}."
)
_T_POINT = "Point to: {}."
_T_SCENE_TEXT = "Detect all the text in box format."
_T_TEXT_GROUNDING = "Please locate the text referred as {}."
_T_GUI_BOX = "Locate the region that matches the following description: {}."

_CATS_SEPARATOR = "</c>"


def build_prompt(
    operation: str,
    classes: list[str] | None,
    prompt: str | None,
    single_instance: bool,
) -> str:
    """Build the operation-specific prompt string (verbatim Eagle templates)."""
    if operation == "detect":
        if not classes:
            raise ValueError("operation='detect' requires non-empty classes=[...]")
        return _T_DETECT.format(_CATS_SEPARATOR.join(classes))

    if operation == "layout":
        used = classes or LAYOUT_DEFAULT_CLASSES
        return _T_DETECT.format(_CATS_SEPARATOR.join(used))

    if operation == "grounding":
        if not prompt:
            raise ValueError("operation='grounding' requires a prompt")
        tpl = _T_GROUNDING_SINGLE if single_instance else _T_GROUNDING_MULTI
        return tpl.format(prompt)

    if operation == "point":
        if not prompt:
            raise ValueError("operation='point' requires a prompt")
        return _T_POINT.format(prompt)

    if operation == "scene_text":
        return _T_SCENE_TEXT

    if operation == "text_grounding":
        if not prompt:
            raise ValueError("operation='text_grounding' requires a prompt")
        return _T_TEXT_GROUNDING.format(prompt)

    if operation == "gui_box":
        if not prompt:
            raise ValueError("operation='gui_box' requires a prompt")
        return _T_GUI_BOX.format(prompt)

    raise ValueError(f"Unknown operation: {operation}")


# ============================================================================
# Device + backend helpers
# ============================================================================


def _get_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _select_dtype(device: str):
    import torch

    if device == "cuda":
        try:
            cap = torch.cuda.get_device_capability()
            return torch.bfloat16 if cap[0] >= 8 else torch.float16
        except Exception:
            return torch.float16
    if device == "mps":
        # bf16 on MPS has partial op coverage; fp16 is the safer default.
        return torch.float16
    return torch.float32


def _select_video_backend() -> str | None:
    """Pick the first available frame-extraction backend.

    Probe order: decord (native or via eva-decord fork) → cv2 → torchvision.io.
    Returns the backend name or None if none are available.
    """
    for name, mod in (
        ("decord", "decord"),
        ("cv2", "cv2"),
        ("torchvision", "torchvision.io"),
    ):
        try:
            __import__(mod)
            return name
        except ImportError:
            continue
    return None


# ============================================================================
# Base model
# ============================================================================


class LocateAnythingBaseModel(
    fom.Model, fom.SamplesMixin, SupportsGetItem, TorchModelMixin
):
    """Shared base for LocateAnything image and video zoo models."""

    def __init__(self, config: LocateAnythingConfig) -> None:
        fom.SamplesMixin.__init__(self)
        SupportsGetItem.__init__(self)

        self._preprocess: bool = False
        self.config: LocateAnythingConfig = config
        self.device: str = _get_device()
        self._dtype = _select_dtype(self.device)
        self._fields: dict[str, str] = {}
        self._model = None
        self._processor = None
        self._tokenizer = None

    # -- FiftyOne boilerplate (mirrors gemma4) ------------------------------

    @property
    def transforms(self) -> None:
        return None

    @property
    def preprocess(self) -> bool:
        return self._preprocess

    @preprocess.setter
    def preprocess(self, value: bool) -> None:
        self._preprocess = value

    @property
    def ragged_batches(self) -> bool:
        return False

    @property
    def needs_fields(self) -> dict[str, str]:
        return self._fields

    @needs_fields.setter
    def needs_fields(self, fields: dict[str, str]) -> None:
        self._fields = fields

    @property
    def has_collate_fn(self) -> bool:
        return True

    # collate_fn is inherited from TorchModelMixin; DO NOT override.

    def build_get_item(
        self, field_mapping: dict[str, str] | None = None
    ) -> ImageGetItem:
        return ImageGetItem(field_mapping=field_mapping, raw_inputs=True)

    # -- Model loading -----------------------------------------------------

    def _load_model(self) -> None:
        """Lazy-load tokenizer, processor, and model (trust_remote_code=True)."""
        from transformers import AutoModel, AutoProcessor, AutoTokenizer

        logger.info(
            "Loading LocateAnything from %s (device=%s, dtype=%s)",
            self.config.model_path,
            self.device,
            self._dtype,
        )
        if self.device == "mps":
            logger.warning(
                "Running on MPS with fp16 fallback (bf16 has partial op coverage). "
                "Expect slower inference vs CUDA bf16."
            )

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path, trust_remote_code=True
        )
        self._processor = AutoProcessor.from_pretrained(
            self.config.model_path, trust_remote_code=True
        )
        self._model = (
            AutoModel.from_pretrained(
                self.config.model_path,
                torch_dtype=self._dtype,
                trust_remote_code=True,
            )
            .to(self.device)
            .eval()
        )

        logger.info("Model loaded")

    # -- Generation core ---------------------------------------------------

    def _generate(self, image: Any, prompt: str) -> str:
        """Run LocateAnything on (image, prompt) and return the raw text.

        Uses the verbatim invocation pattern from Eagle's
        locateanything_worker.py.
        """
        import torch

        if self._model is None:
            self._load_model()

        assert self._model is not None
        assert self._processor is not None
        assert self._tokenizer is not None

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # py_apply_chat_template is the actual method name (verified from source).
        text = self._processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = self._processor.process_vision_info(messages)
        inputs = self._processor(
            text=[text], images=images, videos=videos, return_tensors="pt"
        ).to(self.device)

        pixel_values = inputs["pixel_values"].to(self._dtype)

        gen_kwargs: dict[str, Any] = {
            "pixel_values": pixel_values,
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "image_grid_hws": inputs.get("image_grid_hws"),
            "tokenizer": self._tokenizer,  # YES, tokenizer is a generate kwarg
            "max_new_tokens": self.config.max_new_tokens,
            "use_cache": True,
            "generation_mode": self.config.generation_mode,
            "do_sample": self.config.do_sample,
        }
        if self.config.do_sample:
            gen_kwargs["temperature"] = self.config.temperature
            gen_kwargs["top_p"] = self.config.top_p
            gen_kwargs["repetition_penalty"] = self.config.repetition_penalty
        # fast/hybrid generation modes need n_future_tokens=6 (per eval scripts)
        if self.config.generation_mode in ("fast", "hybrid"):
            gen_kwargs["n_future_tokens"] = 6

        with torch.no_grad():
            response = self._model.generate(**gen_kwargs)

        # Worker source: response is either a tuple or a string.
        if isinstance(response, tuple):
            raw = response[0]
        elif isinstance(response, str):
            raw = response
        else:
            input_len = inputs["input_ids"].shape[-1]
            raw = self._tokenizer.decode(
                response[0][input_len:],
                skip_special_tokens=False,
            )

        if not isinstance(raw, str):
            raw = str(raw)
        logger.debug("Raw output: %s", raw)
        return raw

    def _run_inference(self, image: Any, prompt: str | None) -> fol.Label:
        """Build the full prompt, generate, parse, and convert to a Label."""
        full_prompt = build_prompt(
            self.config.operation,
            classes=self.config.classes,
            prompt=prompt if prompt is not None else self.config.prompt,
            single_instance=self.config.single_instance,
        )
        raw = self._generate(image, full_prompt)

        if self.config.operation == "point":
            entries = parse_output(raw, mode="point")
            return to_keypoints(entries, default_label="point")

        default = "text" if self.config.operation == "scene_text" else "object"
        entries = parse_output(raw, mode="box")
        return to_detections(entries, default_label=default)

    # -- Context manager (cleanup) -----------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *args):
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return False


# ============================================================================
# Image model
# ============================================================================


class LocateAnythingImageModel(LocateAnythingBaseModel):
    @property
    def media_type(self) -> str:
        return "image"

    def predict(self, arg: Any, sample: fo.Sample | None = None) -> fol.Label:
        return self.predict_all([arg], samples=[sample] if sample else None)[0]

    def predict_all(
        self,
        batch: list[Any],
        samples: list[fo.Sample | None] | None = None,
    ) -> list[fol.Label]:
        if not batch:
            return []
        if self._model is None:
            self._load_model()

        results: list[fol.Label] = []
        for i, item in enumerate(batch):
            sample = samples[i] if samples and i < len(samples) else None
            image, call_prompt = self._resolve_inputs(item)

            if call_prompt is None and sample and "prompt_field" in self._fields:
                fn = self._fields["prompt_field"]
                if sample.has_field(fn):
                    call_prompt = sample.get_field(fn)

            results.append(self._run_inference(image, call_prompt))
        return results

    @staticmethod
    def _resolve_inputs(item: Any) -> tuple[Any, str | None]:
        """Normalize DataLoader / single / dict inputs into (image, prompt)."""
        if isinstance(item, dict):
            return item.get("filepath", item.get("image")), item.get("prompt")
        if isinstance(item, str):
            return item, None
        return item, None  # PIL Image from ImageGetItem


# ============================================================================
# Video model
# ============================================================================


class LocateAnythingVideoModel(LocateAnythingBaseModel):
    """Frame-sampled video inference.

    LocateAnything-3B's processor supports video natively, but the model has
    no video-specific code path; it treats frames as image sequences. This
    class decodes the video into frames and runs the image inference path
    per frame. Returns `{frame_num: label}` per sample so FiftyOne merges
    results into `sample.frames[N].field`.
    """

    @property
    def media_type(self) -> str:
        return "video"

    def predict(
        self, arg: Any, sample: fo.Sample | None = None
    ) -> dict[int, fol.Label]:
        return self.predict_all([arg], samples=[sample] if sample else None)[0]

    def predict_all(
        self,
        batch: list[Any],
        samples: list[fo.Sample | None] | None = None,
    ) -> list[dict[int, fol.Label]]:
        if not batch:
            return []
        if self._model is None:
            self._load_model()

        results: list[dict[int, fol.Label]] = []
        for i, item in enumerate(batch):
            filepath = self._resolve_video_filepath(item)
            if filepath is None:
                logger.error(
                    "Could not resolve a video filepath from input of type %s; "
                    "skipping sample",
                    type(item).__name__,
                )
                results.append({})
                continue

            sample = samples[i] if samples and i < len(samples) else None
            call_prompt = None
            if sample and "prompt_field" in self._fields:
                fn = self._fields["prompt_field"]
                if sample.has_field(fn):
                    call_prompt = sample.get_field(fn)

            frame_labels: dict[int, fol.Label] = {}
            for frame_num, pil_image in self._sample_frames(filepath):
                frame_labels[frame_num] = self._run_inference(pil_image, call_prompt)
            results.append(frame_labels)
        return results

    @staticmethod
    def _resolve_video_filepath(item: Any) -> str | None:
        """Normalize an input from any apply_model path into a video filepath.

        FiftyOne's `_apply_video_model` passes an `eta.core.video.VideoReader`
        instance (which stores the source path on `.inpath`). Notebook callers
        may pass a filepath string or a dict. Returns None when no path is
        recoverable so the caller can log a clear message.
        """
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            return item.get("filepath") or item.get("path") or item.get("video")
        for attr in ("inpath", "filepath", "path"):
            value = getattr(item, attr, None)
            if isinstance(value, str):
                return value
        return None

    def _compute_indices(self, total: int, src_fps: float) -> list[int]:
        if self.config.every_nth:
            return list(range(0, total, int(self.config.every_nth)))
        if self.config.fps:
            stride = max(1, int(round(src_fps / float(self.config.fps))))
            return list(range(0, total, stride))
        n = max(1, int(self.config.frames))
        if total <= n:
            return list(range(total))
        if n == 1:
            return [0]
        return [int(round(i * (total - 1) / (n - 1))) for i in range(n)]

    def _sample_frames(self, filepath: str):
        """Yield (frame_num, PIL.Image) tuples per the configured sampling.

        Backend probe order: decord (also covers eva-decord) → cv2 → torchvision.io.
        FiftyOne frame numbers are 1-indexed.
        """
        backend = _select_video_backend()
        if backend == "decord":
            yield from self._sample_frames_decord(filepath)
        elif backend == "cv2":
            yield from self._sample_frames_cv2(filepath)
        elif backend == "torchvision":
            yield from self._sample_frames_torchvision(filepath)
        else:
            raise ImportError(
                "Video inference needs a frame-extraction backend. "
                "Install one of: decord (Linux/Win), eva-decord (arm64 MacOS), "
                "or opencv-python-headless. See README install section."
            )

    def _sample_frames_decord(self, filepath: str):
        import decord
        from PIL import Image

        vr = decord.VideoReader(filepath)
        total = len(vr)
        src_fps = vr.get_avg_fps() or 30.0
        indices = self._compute_indices(total, src_fps)
        if not indices:
            return
        batch = vr.get_batch(indices).asnumpy()  # (N, H, W, 3) RGB
        for idx, arr in zip(indices, batch):
            yield idx + 1, Image.fromarray(arr)

    def _sample_frames_cv2(self, filepath: str):
        import cv2
        from PIL import Image

        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {filepath}")
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            for idx in self._compute_indices(total, src_fps):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if not ok:
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield idx + 1, Image.fromarray(rgb)
        finally:
            cap.release()

    def _sample_frames_torchvision(self, filepath: str):
        from torchvision.io import read_video
        from PIL import Image

        vid, _, info = read_video(filepath, pts_unit="sec", output_format="THWC")
        total = vid.shape[0]
        src_fps = float(info.get("video_fps", 30.0))
        for idx in self._compute_indices(total, src_fps):
            arr = vid[idx].numpy()
            yield idx + 1, Image.fromarray(arr)
