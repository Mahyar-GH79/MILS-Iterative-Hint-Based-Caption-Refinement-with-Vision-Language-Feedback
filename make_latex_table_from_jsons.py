"""
make_latex_table_from_jsons.py

Creates a LaTeX table from:
- per_image_metrics/<img_id>_metrics.json (BLEU_1, BLEU_2, BLEU_4, METEOR, SPICE; it could also contain CIDEr)
- per_image_traces/<img_id>.json (captions per step, used to recompute CIDEr correctly)

Output:
- results_dir/latex_table_steps.tex
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from pycocotools.coco import COCO


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("LATEX_TABLE")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


def safe_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
        if np.isnan(v):
            return None
        return v
    except Exception:
        return None


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def format_0_100(x: Optional[float], decimals: int = 2) -> str:
    if x is None:
        return "NA"
    return f"{x * 100.0:.{decimals}f}"


def compute_cider_corpus_per_step(
    coco: COCO,
    traces_dir: str,
    trace_files: List[str],
    num_steps: int,
    logger: logging.Logger,
) -> Dict[int, Optional[float]]:
    """
    Computes corpus-level CIDEr for each step using Cider scorer over all images.

    Returns: step -> CIDEr (0..10-ish typically), but we will still scale by 100
    only for reporting consistency. CIDEr is not naturally bounded by 1, but you
    asked for 0..100 scaling for all metrics, so we apply the same scaling.
    """
    from pycocoevalcap.cider.cider import Cider

    # Preload refs for all images once
    all_img_ids = []
    for fn in trace_files:
        try:
            tr = load_json(os.path.join(traces_dir, fn))
            all_img_ids.append(int(tr["image_id"]))
        except Exception:
            continue
    all_img_ids = sorted(set(all_img_ids))

    gts_all: Dict[int, List[str]] = {}
    for img_id in all_img_ids:
        ann_ids = coco.getAnnIds(imgIds=[int(img_id)])
        anns = coco.loadAnns(ann_ids)
        gts_all[int(img_id)] = [a["caption"] for a in anns]

    cider_scores: Dict[int, Optional[float]] = {}

    for s in range(num_steps):
        res: Dict[int, List[str]] = {}
        gts: Dict[int, List[str]] = {}

        n_used = 0
        for fn in trace_files:
            try:
                tr = load_json(os.path.join(traces_dir, fn))
                img_id = int(tr["image_id"])
                steps = tr.get("steps", []) or []
                if s >= len(steps):
                    continue
                cap = str(steps[s].get("caption", "") or "").strip()
                if not cap:
                    continue
                res[img_id] = [cap]
                gts[img_id] = gts_all.get(img_id, [])
                n_used += 1
            except Exception:
                continue

        if n_used == 0:
            cider_scores[s] = None
            continue

        try:
            scorer = Cider()
            score, _ = scorer.compute_score(gts, res)
            cider_scores[s] = float(score)
        except Exception as e:
            logger.error(f"CIDEr failed at step={s}: {e}")
            cider_scores[s] = None

        logger.info(f"CIDEr step={s} computed on N={n_used}")

    return cider_scores


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", required=True, type=str)
    p.add_argument("--ann_file", required=True, type=str)
    p.add_argument("--num_steps", default=9, type=int)
    p.add_argument("--out_tex", default=None, type=str)
    p.add_argument("--decimals", default=2, type=int)
    args = p.parse_args()

    logger = setup_logger()

    traces_dir = os.path.join(args.results_dir, "per_image_traces")
    metrics_dir = os.path.join(args.results_dir, "per_image_metrics")

    if not os.path.isdir(traces_dir):
        raise FileNotFoundError(f"Missing traces_dir: {traces_dir}")
    if not os.path.isdir(metrics_dir):
        raise FileNotFoundError(f"Missing metrics_dir: {metrics_dir}")
    if not os.path.exists(args.ann_file):
        raise FileNotFoundError(f"Missing ann_file: {args.ann_file}")

    out_tex = args.out_tex or os.path.join(args.results_dir, "latex_table_steps.tex")

    # Load COCO
    coco = COCO(args.ann_file)

    # Collect files
    trace_files = sorted([f for f in os.listdir(traces_dir) if f.endswith(".json")])
    metric_files = sorted([f for f in os.listdir(metrics_dir) if f.endswith("_metrics.json")])

    if len(trace_files) == 0:
        raise RuntimeError("No trace files found.")
    if len(metric_files) == 0:
        raise RuntimeError("No metric files found.")

    logger.info(f"Found traces={len(trace_files)} metrics={len(metric_files)}")

    # Map img_id -> metric json path
    metrics_by_id: Dict[int, str] = {}
    for fn in metric_files:
        try:
            img_id = int(fn.replace("_metrics.json", ""))
            metrics_by_id[img_id] = os.path.join(metrics_dir, fn)
        except Exception:
            continue

    # Step-level means from per-image metrics JSONs
    keys = ["BLEU_1", "BLEU_2", "BLEU_4", "METEOR", "SPICE"]
    step_means: Dict[int, Dict[str, Optional[float]]] = {
        s: {k: None for k in keys} for s in range(args.num_steps)
    }
    step_counts: Dict[int, Dict[str, int]] = {
        s: {k: 0 for k in keys} for s in range(args.num_steps)
    }

    # Load each per-image metrics json and accumulate
    for img_id, mp in metrics_by_id.items():
        try:
            per_step = load_json(mp)  # dict: "0"->metrics
            for s in range(args.num_steps):
                rec = per_step.get(str(s), {}) or {}
                for k in keys:
                    v = safe_float(rec.get(k, None))
                    if v is None:
                        continue
                    if step_means[s][k] is None:
                        step_means[s][k] = 0.0
                    step_means[s][k] += float(v)
                    step_counts[s][k] += 1
        except Exception:
            continue

    # Finalize means
    for s in range(args.num_steps):
        for k in keys:
            c = step_counts[s][k]
            if c > 0:
                step_means[s][k] = float(step_means[s][k]) / float(c)
            else:
                step_means[s][k] = None

    # Correct CIDEr: recompute corpus-level per step from traces + COCO refs
    cider_per_step = compute_cider_corpus_per_step(
        coco=coco,
        traces_dir=traces_dir,
        trace_files=trace_files,
        num_steps=args.num_steps,
        logger=logger,
    )

    # Build LaTeX table rows (scaled 0..100 for everything)
    rows = []
    for s in range(args.num_steps):
        row = {
            "step": s,
            "BLEU_1": step_means[s]["BLEU_1"],
            "BLEU_2": step_means[s]["BLEU_2"],
            "BLEU_4": step_means[s]["BLEU_4"],
            "METEOR": step_means[s]["METEOR"],
            "SPICE": step_means[s]["SPICE"],
            "CIDEr": cider_per_step.get(s, None),
        }
        rows.append(row)

    # Average across steps (mean of step-level values, ignoring NA)
    def avg_across_steps(metric_name: str) -> Optional[float]:
        vals = []
        for r in rows:
            v = r.get(metric_name, None)
            if v is None:
                continue
            vals.append(float(v))
        return float(np.mean(vals)) if len(vals) else None

    avg_row = {
        "step": "Avg",
        "BLEU_1": avg_across_steps("BLEU_1"),
        "BLEU_2": avg_across_steps("BLEU_2"),
        "BLEU_4": avg_across_steps("BLEU_4"),
        "METEOR": avg_across_steps("METEOR"),
        "SPICE": avg_across_steps("SPICE"),
        "CIDEr": avg_across_steps("CIDEr"),
    }

    # LaTeX
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\setlength{\tabcolsep}{6pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(r"\begin{tabular}{ccccccc}")
    lines.append(r"\hline")
    lines.append(r"Step & BLEU1 & BLEU2 & BLEU4 & METEOR & SPICE & CIDEr \\")
    lines.append(r"\hline")

    for r in rows:
        lines.append(
            f"{r['step']} & "
            f"{format_0_100(r['BLEU_1'], args.decimals)} & "
            f"{format_0_100(r['BLEU_2'], args.decimals)} & "
            f"{format_0_100(r['BLEU_4'], args.decimals)} & "
            f"{format_0_100(r['METEOR'], args.decimals)} & "
            f"{format_0_100(r['SPICE'], args.decimals)} & "
            f"{format_0_100(r['CIDEr'], args.decimals)} \\\\"
        )

    lines.append(r"\hline")
    lines.append(
        f"{avg_row['step']} & "
        f"{format_0_100(avg_row['BLEU_1'], args.decimals)} & "
        f"{format_0_100(avg_row['BLEU_2'], args.decimals)} & "
        f"{format_0_100(avg_row['BLEU_4'], args.decimals)} & "
        f"{format_0_100(avg_row['METEOR'], args.decimals)} & "
        f"{format_0_100(avg_row['SPICE'], args.decimals)} & "
        f"{format_0_100(avg_row['CIDEr'], args.decimals)} \\\\"
    )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Dataset metrics per step (scaled to 0 to 100) and average across steps.}")
    lines.append(r"\label{tab:step_metrics}")
    lines.append(r"\end{table}")

    latex = "\n".join(lines)

    with open(out_tex, "w") as f:
        f.write(latex + "\n")

    print(latex)
    logger.info(f"Wrote LaTeX table to: {out_tex}")

    # Also print a short sanity line about CIDEr not being all zeros
    nonzero = [r["CIDEr"] for r in rows if (r["CIDEr"] is not None and abs(r["CIDEr"]) > 1e-12)]
    if len(nonzero) == 0:
        logger.warning("CIDEr appears to be all zeros or missing. Check trace captions and ann_file.")
    else:
        logger.info("CIDEr looks non-zero for at least some steps.")


if __name__ == "__main__":
    main()
