"""Eagle JSONL dataset importer for FiftyOne.

Loads Rex-Omni-EvalData and Eagle Embodied-format JSONL files
(ShareGPT-style conversations with <ref>label</ref><box><x1><y1><x2><y2></box>
ground truth) into FiftyOne datasets.

Format reference: https://github.com/NVlabs/Eagle/blob/main/Embodied/document/DATA_PREPARATION.md
Box/point regexes are verbatim from Embodied/locateanything_worker.py:142-169.
"""

from __future__ import annotations

import json
import logging
import os
import re

import fiftyone as fo

logger = logging.getLogger(__name__)


# Verbatim from Eagle/Embodied/locateanything_worker.py:
_BOX_RE = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")
_POINT_RE = re.compile(r"<box><(\d+)><(\d+)></box>")
_REF_TAG_RE = re.compile(r"<ref>([^<]*)</ref>")


def _nearest_preceding_ref(text: str, before_pos: int) -> str | None:
    """Return the most recent <ref>...</ref> label appearing before `before_pos`."""
    label = None
    for m in _REF_TAG_RE.finditer(text, endpos=before_pos):
        candidate = m.group(1).strip()
        if candidate:
            label = candidate
    return label


def _parse_gt_text(text: str) -> tuple[fo.Detections, fo.Keypoints]:
    """Parse the gpt-turn text into (Detections, Keypoints).

    Uses the worker's verbatim regexes. <ref> labels (extension for FO usability)
    are associated with the nearest preceding box / point.
    """
    dets: list[fo.Detection] = []
    for m in _BOX_RE.finditer(text):
        x1, y1, x2, y2 = (int(m.group(i)) for i in range(1, 5))
        if not all(0 <= c <= 1000 for c in (x1, y1, x2, y2)):
            continue
        w, h = (x2 - x1) / 1000.0, (y2 - y1) / 1000.0
        if w <= 0 or h <= 0:
            continue
        label = _nearest_preceding_ref(text, m.start()) or "object"
        dets.append(
            fo.Detection(
                label=label,
                bounding_box=[x1 / 1000.0, y1 / 1000.0, w, h],
            )
        )

    kps: list[fo.Keypoint] = []
    for m in _POINT_RE.finditer(text):
        x, y = int(m.group(1)), int(m.group(2))
        if not (0 <= x <= 1000 and 0 <= y <= 1000):
            continue
        label = _nearest_preceding_ref(text, m.start()) or "point"
        kps.append(fo.Keypoint(label=label, points=[[x / 1000.0, y / 1000.0]]))

    return fo.Detections(detections=dets), fo.Keypoints(keypoints=kps)


def _gpt_turn(conversations: list[dict[str, str]]) -> str:
    for turn in conversations:
        if turn.get("from") == "gpt":
            return turn.get("value", "")
    return ""


def _human_turn(conversations: list[dict[str, str]]) -> str:
    for turn in conversations:
        if turn.get("from") == "human":
            return turn.get("value", "")
    return ""


def load_eagle_jsonl(
    jsonl_path: str,
    image_root: str,
    name: str | None = None,
    label_field: str = "ground_truth",
    point_field: str = "ground_truth_points",
    persistent: bool = False,
    max_samples: int | None = None,
) -> fo.Dataset:
    """Load a Rex-Omni / Eagle Embodied JSONL eval bundle into FiftyOne.

    Each line is a ShareGPT-style record:
        {"conversations": [{"from": "human", "value": "..."},
                            {"from": "gpt", "value": "<ref>...</ref><box>...</box>"}],
         "image": "relative/path.png"}

    The `gpt` turn's <ref>/<box> tokens are parsed into `label_field` (Detections)
    and `point_field` (Keypoints).

    Multi-image samples (`image_list`) become one sample per image with a shared
    `group_id`. Video samples (`video`/`video_list`) are loaded with the video
    filepath; no automatic frame extraction.

    Args:
        jsonl_path: Path to the JSONL file.
        image_root: Directory the JSONL's relative image paths resolve against.
        name: FiftyOne dataset name (random if None).
        label_field: Field for ground truth Detections.
        point_field: Field for ground truth Keypoints.
        persistent: If True, persist the dataset across sessions.
        max_samples: Cap the number of JSONL lines loaded (None = all).
    """
    dataset = fo.Dataset(name=name, persistent=persistent)
    image_root = os.path.expanduser(image_root)

    samples: list[fo.Sample] = []
    with open(os.path.expanduser(jsonl_path)) as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            if max_samples is not None and len(samples) >= max_samples:
                break

            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("Skipping malformed JSONL line %d: %s", line_no, e)
                continue

            conv = rec.get("conversations", [])
            gt_text = _gpt_turn(conv)
            prompt = _human_turn(conv)
            dets, kps = _parse_gt_text(gt_text)

            media_paths: list[str] = []
            for k in ("image", "image_list", "video", "video_list"):
                if k in rec:
                    v = rec[k]
                    media_paths.extend([v] if isinstance(v, str) else list(v))
                    break

            if not media_paths:
                logger.debug("Line %d has no image/video; skipping", line_no)
                continue

            group_id = f"grp{line_no}" if len(media_paths) > 1 else None

            for mp in media_paths:
                full = os.path.join(image_root, mp) if not os.path.isabs(mp) else mp
                sample = fo.Sample(filepath=full)
                sample["prompt"] = prompt
                sample["source_line_no"] = line_no
                if group_id is not None:
                    sample["group_id"] = group_id
                if len(dets.detections) > 0:
                    sample[label_field] = dets
                if len(kps.keypoints) > 0:
                    sample[point_field] = kps
                samples.append(sample)

    dataset.add_samples(samples)
    logger.info(
        "Loaded %d samples from %s into dataset '%s'",
        len(samples),
        jsonl_path,
        dataset.name,
    )
    return dataset
