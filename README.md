# fo-locate-anything

FiftyOne Zoo remote model wrapping NVIDIA's
[LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B) —
an open-vocabulary grounding VLM from the
[Eagle](https://github.com/NVlabs/Eagle) family.

**Image + video support, 7 operations, and an Eagle JSONL dataset importer
for evaluating against Rex-Omni-EvalData benchmarks (DocLayNet, COCO, LVIS,
ScreenSpot-Pro, etc.).**

## License notice

LocateAnything-3B weights are released under the **NVIDIA License —
non-commercial research only**. This wrapper is MIT, but the model it
loads is not free for commercial use.

## Install

```bash
# image-only inference
pip install fo-locate-anything

# + video frame sampling
# (Linux/Win: opencv + decord. MacOS: opencv only — see "MacOS users" below)
pip install "fo-locate-anything[video]"

# MacOS users who want decord performance: opt into the eva-decord fork
pip install "fo-locate-anything[video,decord-mac]"

# everything
pip install "fo-locate-anything[all]"
```

**Note for MacOS users:** the native `decord` wheel is not available on arm64
Macs. The `[video]` extra installs only opencv-python-headless on MacOS; opt
into `[decord-mac]` to additionally install `eva-decord` (a maintained fork
that registers as the `decord` module). The runtime backend selector tries
`decord` first, falls back to `cv2`, then `torchvision.io`.

Then register the zoo source:

```python
import fiftyone.zoo as foz

foz.register_zoo_model_source(
    "https://github.com/Burhan-Q/fo-locate-anything"
)
```

## Operations

| Operation       | Required arg                                                   | Output          |
|-----------------|----------------------------------------------------------------|-----------------|
| `detect`        | `classes=[...]`                                                | `fo.Detections` |
| `grounding`     | `prompt="..."` (`single_instance=True` for one)                | `fo.Detections` |
| `point`         | `prompt="..."`                                                 | `fo.Keypoints`  |
| `scene_text`    | _(none)_                                                       | `fo.Detections` |
| `layout`        | optional `classes=[...]` (default: title/paragraph/figure/table) | `fo.Detections` |
| `text_grounding`| `prompt="..."` (referring to text in image)                    | `fo.Detections` |
| `gui_box`       | `prompt="..."` (GUI element region)                            | `fo.Detections` |

## Examples

### Detect specific classes

```python
model = foz.load_zoo_model(
    "nvidia/LocateAnything-3B",
    operation="detect",
    classes=["car", "person", "traffic light"],
)
dataset.apply_model(model, label_field="detections")
```

### Phrase grounding with per-sample prompts

```python
model = foz.load_zoo_model("nvidia/LocateAnything-3B", operation="grounding")
dataset.apply_model(model, label_field="grounded", prompt_field="caption")
```

### Document layout

```python
model = foz.load_zoo_model("nvidia/LocateAnything-3B", operation="layout")
dataset.apply_model(model, label_field="layout")
# Default classes: title, paragraph, figure, table
```

### Pointing (e.g., GUI element)

```python
model = foz.load_zoo_model(
    "nvidia/LocateAnything-3B",
    operation="point",
    prompt="the submit button",
)
dataset.apply_model(model, label_field="ui_point")
```

### GUI region grounding (vs point)

```python
model = foz.load_zoo_model(
    "nvidia/LocateAnything-3B",
    operation="gui_box",
    prompt="the file menu",
)
dataset.apply_model(model, label_field="ui_box")
```

### Scene text / OCR localization

```python
model = foz.load_zoo_model("nvidia/LocateAnything-3B", operation="scene_text")
dataset.apply_model(model, label_field="text")
```

### Text grounding (find a specific phrase in the image)

```python
model = foz.load_zoo_model(
    "nvidia/LocateAnything-3B",
    operation="text_grounding",
    prompt="invoice number",
)
dataset.apply_model(model, label_field="text_location")
```

### Video — frame-sampled inference

```python
video_model = foz.load_zoo_model(
    "nvidia/LocateAnything-3B",
    media_type="video",
    operation="detect",
    classes=["person", "car"],
    frames=8,                # or fps=2.0, or every_nth=15
)
video_dataset.apply_model(video_model, label_field="dets")
# Per-frame results land in sample.frames[N].dets
```

## Loading Eagle / Rex-Omni eval bundles

Eagle ships eval data as JSONL in ShareGPT format with
`<ref>label</ref><box>...</box>` ground-truth tokens. Use the included importer:

```python
from fo_locate_anything.dataset import load_eagle_jsonl

ds = load_eagle_jsonl(
    jsonl_path="~/data/rex_omni/DocLayNet/annotations.jsonl",
    image_root="~/data/rex_omni/DocLayNet/images",
    name="doclaynet-eval",
)
# ds[i].ground_truth is fo.Detections; ds[i].prompt is the human turn
```

Compatible eval bundles:
- [`Mountchicken/Rex-Omni-EvalData`](https://huggingface.co/datasets/Mountchicken/Rex-Omni-EvalData)
  (COCO, LVIS, Dense200, VisDrone, DocLayNet, M6Doc, TotalText, HierText, RefCOCOg, HumanRef)
- [`likaixin/ScreenSpot-Pro`](https://huggingface.co/datasets/likaixin/ScreenSpot-Pro)

## Configuration reference

| Kwarg | Default | Notes |
|---|---|---|
| `media_type` | `"image"` | `"image"` or `"video"` |
| `operation` | `"detect"` | One of 7 ops above |
| `classes` | `None` | Required for `detect`; optional for `layout` |
| `prompt` | `None` | Required for prompt-based ops |
| `single_instance` | `False` | Grounding only |
| `generation_mode` | `"hybrid"` | `"hybrid"` / `"fast"` / `"slow"` |
| `max_new_tokens` | `2048` | |
| `do_sample` | `True` | |
| `temperature` | `0.7` | |
| `top_p` | `0.9` | |
| `repetition_penalty` | `1.1` | |
| `frames` | `8` | Video: # evenly-spaced frames |
| `fps` | `None` | Video: target sampling FPS (overrides `frames`) |
| `every_nth` | `None` | Video: sample every Kth frame (overrides others) |

## Limitations

- **No confidence scores** — the model emits no per-detection scores;
  `fo.Detection.confidence` is `None`. Affects mAP tie-breaking.
- **Single-image inference** at the model level — no native batching.
- **bf16 on CUDA; fp16 on MPS; fp32 on CPU.** Apple Silicon works but is slower.
- **Video is frame-by-frame** — no temporal modeling, no cross-frame tracking.
- **Layout taxonomy is 4 classes** (`title`, `paragraph`, `figure`, `table`);
  for other layouts use `detect` with your own class list.

## Known-working dependency pins

The Eagle pyproject pins specific versions. If you hit issues with the
looser ranges in our manifest, try:

```
transformers==4.57.1
tokenizers==0.22.0
```
