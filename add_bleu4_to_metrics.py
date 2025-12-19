"""
add_bleu4_to_metrics.py

Adds BLEU-4 to existing per-image metrics JSON files by reading captions from traces.

"""

import os
import json
import time
import logging
from typing import Dict, Any, List

from pycocotools.coco import COCO


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("ADD_BLEU4")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


def compute_bleu4_for_caption(coco: COCO, img_id: int, caption: str) -> float:
    from pycocoevalcap.bleu.bleu import Bleu

    ann_ids = coco.getAnnIds(imgIds=[int(img_id)])
    anns = coco.loadAnns(ann_ids)
    refs: List[str] = [a["caption"] for a in anns]

    gts = {int(img_id): refs}
    res = {int(img_id): [caption]}

    bleu = Bleu(4)
    bleu_scores, _ = bleu.compute_score(gts, res)
    # bleu_scores = [BLEU-1, BLEU-2, BLEU-3, BLEU-4]
    return float(bleu_scores[3])


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--traces_dir", required=True, type=str)
    p.add_argument("--metrics_dir", required=True, type=str)
    p.add_argument("--ann_file", required=True, type=str)
    p.add_argument("--num_steps", default=10, type=int)
    p.add_argument("--force", action="store_true", help="Recompute BLEU_4 even if already present")
    p.add_argument("--log_every", default=50, type=int)
    args = p.parse_args()

    logger = setup_logger()

    if not os.path.isdir(args.traces_dir):
        raise FileNotFoundError(f"Missing traces_dir: {args.traces_dir}")
    if not os.path.isdir(args.metrics_dir):
        raise FileNotFoundError(f"Missing metrics_dir: {args.metrics_dir}")
    if not os.path.exists(args.ann_file):
        raise FileNotFoundError(f"Missing ann_file: {args.ann_file}")

    coco = COCO(args.ann_file)

    trace_files = sorted([f for f in os.listdir(args.traces_dir) if f.endswith(".json")])
    total = len(trace_files)
    logger.info(f"Found {total} trace files")

    updated = 0
    skipped = 0
    missing_metrics = 0
    failed = 0
    t0 = time.time()

    for i, fn in enumerate(trace_files, start=1):
        trace_path = os.path.join(args.traces_dir, fn)

        try:
            with open(trace_path, "r") as f:
                tr = json.load(f)
            img_id = int(tr["image_id"])
        except Exception as e:
            failed += 1
            logger.error(f"[{i}/{total}] Bad trace file {fn}: {e}")
            continue

        metrics_path = os.path.join(args.metrics_dir, f"{img_id}_metrics.json")
        if not os.path.exists(metrics_path):
            missing_metrics += 1
            continue

        try:
            with open(metrics_path, "r") as f:
                per_step_metrics = json.load(f)
            if not isinstance(per_step_metrics, dict):
                raise ValueError("metrics json is not a dict")
        except Exception as e:
            failed += 1
            logger.error(f"[{i}/{total}] Bad metrics file for img_id={img_id}: {e}")
            continue

        steps = tr.get("steps", []) or []
        changed = False

        for s in range(min(args.num_steps, len(steps))):
            step_key = str(s)
            if step_key not in per_step_metrics:
                per_step_metrics[step_key] = {}

            met = per_step_metrics[step_key]
            if (not args.force) and ("BLEU_4" in met) and (met["BLEU_4"] is not None):
                continue

            caption = str(steps[s].get("caption", "") or "").strip()
            if not caption:
                met["BLEU_4"] = None
                changed = True
                continue

            try:
                bleu4 = compute_bleu4_for_caption(coco, img_id, caption)
                met["BLEU_4"] = float(bleu4)
                changed = True
            except Exception as e:
                met["BLEU_4_error"] = str(e)
                met["BLEU_4"] = None
                changed = True

        if changed:
            with open(metrics_path, "w") as f:
                json.dump(per_step_metrics, f, indent=2)
            updated += 1
        else:
            skipped += 1

        if (i % args.log_every) == 0:
            elapsed_min = (time.time() - t0) / 60.0
            logger.info(
                f"Progress {i}/{total} | updated={updated} skipped={skipped} "
                f"missing_metrics={missing_metrics} failed={failed} | elapsed_min={elapsed_min:.1f}"
            )

    elapsed_min = (time.time() - t0) / 60.0
    logger.info(
        f"DONE | updated={updated} skipped={skipped} missing_metrics={missing_metrics} failed={failed} "
        f"| elapsed_min={elapsed_min:.1f}"
    )


if __name__ == "__main__":
    main()
