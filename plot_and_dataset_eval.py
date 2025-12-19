
"""
plot_and_dataset_eval.py

Offline evaluation + plotting + dataset-level table with RESUME + timeouts.

Reads:
- results_dir/per_image_traces/*.json

Writes:
- results_dir/per_image_metrics/<img_id>_metrics.json
- results_dir/dataset_metrics_per_step.json
- results_dir/plots/*.png
- results_dir/run_eval.log

Metrics:
- BLEU_1, BLEU_2, ROUGE_L, CIDEr (always)
- METEOR, SPICE (timeouts; may be None with error message)

"""

import os
import json
import time
import logging
import tempfile
import subprocess
import traceback
import multiprocessing as mp
from typing import List, Dict, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO


# =========================
# Logging
# =========================

def setup_logger(results_dir: str) -> logging.Logger:
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "run_eval.log")

    logger = logging.getLogger("EVAL_PLOT")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info(f"Logging to {log_path}")
    return logger


# =========================
# Timeout runner 
# =========================

def run_with_timeout(fn, args, timeout_sec: int) -> Tuple[bool, Any]:
    """
    Runs fn(*args) in a separate process; kills if timeout.
    Returns (ok, result_or_error_string).
    """
    q = mp.Queue()

    def _worker(q_, fn_, args_):
        try:
            q_.put(("ok", fn_(*args_)))
        except Exception:
            q_.put(("err", traceback.format_exc()))

    p = mp.Process(target=_worker, args=(q, fn, args))
    p.start()
    p.join(timeout=timeout_sec)

    if p.is_alive():
        p.terminate()
        p.join(5)
        return False, f"timeout after {timeout_sec}s"
    if q.empty():
        return False, "no result returned"
    status, payload = q.get()
    if status == "ok":
        return True, payload
    return False, payload


# =========================
# SPICE patch with timeout
# =========================

def patch_spice_to_use_java(java_bin: str, spice_cache_dir: str, logger: logging.Logger, spice_timeout_sec: int):
    import pycocoevalcap.spice.spice as spice_mod
    os.makedirs(spice_cache_dir, exist_ok=True)

    def patched_compute_score(self, gts, res):
        spice_dir = os.path.dirname(spice_mod.__file__)
        spice_jar = os.path.join(spice_dir, "spice-1.0.jar")

        input_data = []
        for image_id in res.keys():
            hypo_list = res[image_id]
            hypo = hypo_list[0] if isinstance(hypo_list, list) and len(hypo_list) else ""
            refs = gts.get(image_id, [])
            if not isinstance(refs, list):
                refs = []
            input_data.append({"image_id": int(image_id), "test": str(hypo), "refs": [str(r) for r in refs]})

        in_file = tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".json")
        out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        json.dump(input_data, in_file)
        in_file.close()

        cmd = [
            java_bin, "-Xmx8G", "-jar", spice_jar,
            in_file.name,
            "-cache", spice_cache_dir,
            "-out", out_file.name,
            "-subset",
            "-silent",
        ]

        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=spice_timeout_sec,
            )
        except subprocess.TimeoutExpired:
            logger.warning(f"SPICE timeout after {spice_timeout_sec}s (returning zeros)")
            # return zeros instead of raising, so the pipeline continues without breaking
            return 0.0, [0.0] * len(res.keys())
        except subprocess.CalledProcessError as e:
            logger.error("SPICE failed.\nSTDOUT:\n%s\nSTDERR:\n%s", e.stdout, e.stderr)
            raise
        finally:
            if os.path.exists(in_file.name):
                os.remove(in_file.name)

        with open(out_file.name, "r") as f:
            data = json.load(f)

        if os.path.exists(out_file.name):
            os.remove(out_file.name)

        per_image_scores: List[float] = []
        for item in data:
            scores_obj = item.get("scores", {})
            all_obj = scores_obj.get("All", {})
            f_score = 0.0
            if isinstance(all_obj, dict) and "f" in all_obj:
                try:
                    f_score = float(all_obj["f"])
                except Exception:
                    f_score = 0.0
            per_image_scores.append(float(f_score))

        avg = float(np.mean(per_image_scores)) if per_image_scores else 0.0
        return avg, per_image_scores

    spice_mod.Spice.compute_score = patched_compute_score
    logger.info(f"SPICE uses Java at: {java_bin}")
    logger.info(f"SPICE cache dir: {spice_cache_dir}")
    logger.info(f"SPICE timeout sec: {spice_timeout_sec}")


# =========================
# Metric primitives
# =========================

def _meteor_compute_one(gts, res) -> float:
    from pycocoevalcap.meteor.meteor import Meteor
    m = Meteor()
    try:
        score, _ = m.compute_score(gts, res)
        return float(score)
    finally:
        try:
            m.close()
        except Exception:
            pass


def _spice_compute_one(gts, res) -> float:
    from pycocoevalcap.spice.spice import Spice
    s = Spice()
    score, _ = s.compute_score(gts, res)
    return float(score)


def compute_metrics_one_caption(
    coco: COCO,
    img_id: int,
    caption: str,
    meteor_timeout_sec: int,
    spice_timeout_sec: int,
) -> Dict[str, Any]:
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider

    ann_ids = coco.getAnnIds(imgIds=[int(img_id)])
    anns = coco.loadAnns(ann_ids)
    refs = [a["caption"] for a in anns]

    gts = {int(img_id): refs}
    res = {int(img_id): [caption]}

    out: Dict[str, Any] = {}

    # BLEU
    bleu = Bleu(4)
    bleu_scores, _ = bleu.compute_score(gts, res)
    out["BLEU_1"] = float(bleu_scores[0])
    out["BLEU_2"] = float(bleu_scores[1])

    # ROUGE_L
    rouge = Rouge()
    rouge_score, _ = rouge.compute_score(gts, res)
    out["ROUGE_L"] = float(rouge_score)

    # CIDEr
    cider = Cider()
    cider_score, _ = cider.compute_score(gts, res)
    out["CIDEr"] = float(cider_score)

    # METEOR with hard timeout
    ok_m, meteor_val = run_with_timeout(_meteor_compute_one, (gts, res), meteor_timeout_sec)
    if ok_m:
        out["METEOR"] = float(meteor_val)
    else:
        out["METEOR"] = None
        out["meteor_error"] = str(meteor_val)

    # SPICE with hard timeout
    ok_s, spice_val = run_with_timeout(_spice_compute_one, (gts, res), spice_timeout_sec)
    if ok_s:
        out["SPICE"] = float(spice_val)
    else:
        out["SPICE"] = None
        out["spice_error"] = str(spice_val)

    return out


# =========================
# Plotting
# =========================

def set_plot_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 1.0,
        "lines.linewidth": 2.0,
        "lines.markersize": 5.0,
        "figure.dpi": 120,
        "savefig.dpi": 300,
    })


def bootstrap_ci_mean(values: np.ndarray, n_boot: int = 1500, alpha: float = 0.05, seed: int = 0) -> Tuple[float, float]:
    v = values[~np.isnan(values)]
    if len(v) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        samp = rng.choice(v, size=len(v), replace=True)
        boots.append(float(np.mean(samp)))
    boots = np.array(boots, dtype=float)
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return lo, hi


def plot_part_A_B_from_arrays(
    out_dir: str,
    num_steps: int,
    mats: Dict[str, np.ndarray],
    vlm_scores: np.ndarray,
    logger: logging.Logger,
):
    set_plot_style()
    steps = np.arange(num_steps, dtype=int)

    colors = {
        "BLEU_1": "#1f77b4",
        "BLEU_2": "#ff7f0e",
        "ROUGE_L": "#2ca02c",
        "CIDEr": "#7f7f7f",
        "METEOR": "#d62728",
        "SPICE": "#9467bd",
        "VLM": "#8c564b",
    }

    plot_metric_keys = ["BLEU_1", "BLEU_2", "ROUGE_L", "METEOR", "SPICE"]  # as you wanted

    # Part A
    figA, axesA = plt.subplots(2, 3, figsize=(13.5, 7.2), sharex=True)
    axesA = axesA.flatten()

    for idx, k in enumerate(plot_metric_keys):
        ax = axesA[idx]
        y = mats[k]
        mean = np.nanmean(y, axis=0)
        lo = np.full_like(mean, np.nan)
        hi = np.full_like(mean, np.nan)
        for s in range(num_steps):
            lo[s], hi[s] = bootstrap_ci_mean(y[:, s], seed=1000 + s)
        ax.plot(steps, mean, marker="o", color=colors[k])
        ax.fill_between(steps, lo, hi, alpha=0.18, color=colors[k])
        ax.set_title(k)
        ax.set_xlabel("Step")
        ax.set_ylabel("Score")

    ax_v = axesA[5]
    mean = np.nanmean(vlm_scores, axis=0)
    lo = np.full_like(mean, np.nan)
    hi = np.full_like(mean, np.nan)
    for s in range(num_steps):
        lo[s], hi[s] = bootstrap_ci_mean(vlm_scores[:, s], seed=2000 + s)

    ax_v.plot(steps, mean, marker="o", color=colors["VLM"])
    ax_v.fill_between(steps, lo, hi, alpha=0.18, color=colors["VLM"])
    ax_v.set_title("VLM faithfulness score")
    ax_v.set_xlabel("Step")
    ax_v.set_ylabel("Score (0 to 10)")

    figA.suptitle("Part A: Mean metrics vs step (bootstrap 95 percent CI)", y=1.02, fontsize=14)
    figA.tight_layout()
    figA.savefig(os.path.join(out_dir, "PartA_mean_metrics_bleu12.png"), bbox_inches="tight")
    plt.close(figA)

    # Part B1: delta from step 0
    figB1, axesB1 = plt.subplots(2, 3, figsize=(13.5, 7.2), sharex=True)
    axesB1 = axesB1.flatten()

    for idx, k in enumerate(plot_metric_keys):
        ax = axesB1[idx]
        y = mats[k]
        base = y[:, [0]]
        delta = y - base
        mean = np.nanmean(delta, axis=0)

        lo = np.full_like(mean, np.nan)
        hi = np.full_like(mean, np.nan)
        for s in range(num_steps):
            lo[s], hi[s] = bootstrap_ci_mean(delta[:, s], seed=3000 + s)

        ax.plot(steps, mean, marker="o", color=colors[k])
        ax.fill_between(steps, lo, hi, alpha=0.18, color=colors[k])
        ax.axhline(0.0, linewidth=1.0, alpha=0.6, color="black")
        ax.set_title(f"Delta {k}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Delta")

    axesB1[5].axis("off")
    figB1.suptitle("Part B1: Delta from step 0 (bootstrap 95 percent CI)", y=1.02, fontsize=14)
    figB1.tight_layout()
    figB1.savefig(os.path.join(out_dir, "PartB1_delta_from_step0_bleu12.png"), bbox_inches="tight")
    plt.close(figB1)

    # Part B2: win rate vs step0
    figB2, axesB2 = plt.subplots(2, 3, figsize=(13.5, 7.2), sharex=True)
    axesB2 = axesB2.flatten()

    for idx, k in enumerate(plot_metric_keys):
        ax = axesB2[idx]
        y = mats[k]
        base = y[:, 0]
        win = []
        for s in range(num_steps):
            cur = y[:, s]
            ok = ~np.isnan(cur) & ~np.isnan(base)
            win.append(float(np.mean(cur[ok] > base[ok])) if np.sum(ok) else np.nan)
        win = np.array(win, dtype=float)

        ax.plot(steps, win, marker="o", color=colors[k])
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"Win rate {k}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Fraction improved")

    axesB2[5].axis("off")
    figB2.suptitle("Part B2: Win rate vs step", y=1.02, fontsize=14)
    figB2.tight_layout()
    figB2.savefig(os.path.join(out_dir, "PartB2_win_rate_bleu12.png"), bbox_inches="tight")
    plt.close(figB2)

    # Part B3: best so far
    figB3, axesB3 = plt.subplots(2, 3, figsize=(13.5, 7.2), sharex=True)
    axesB3 = axesB3.flatten()

    for idx, k in enumerate(plot_metric_keys):
        ax = axesB3[idx]
        y = mats[k]
        best = np.full_like(y, np.nan)
        for i in range(y.shape[0]):
            running = -np.inf
            for s in range(y.shape[1]):
                v = y[i, s]
                if not np.isnan(v):
                    running = max(running, v)
                best[i, s] = running if running > -np.inf else np.nan

        mean = np.nanmean(best, axis=0)
        lo = np.full_like(mean, np.nan)
        hi = np.full_like(mean, np.nan)
        for s in range(num_steps):
            lo[s], hi[s] = bootstrap_ci_mean(best[:, s], seed=4000 + s)

        ax.plot(steps, mean, marker="o", color=colors[k])
        ax.fill_between(steps, lo, hi, alpha=0.18, color=colors[k])
        ax.set_title(f"Best so far {k}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Score")

    axesB3[5].axis("off")
    figB3.suptitle("Part B3: Best so far vs step (bootstrap 95 percent CI)", y=1.02, fontsize=14)
    figB3.tight_layout()
    figB3.savefig(os.path.join(out_dir, "PartB3_best_so_far_bleu12.png"), bbox_inches="tight")
    plt.close(figB3)

    logger.info(f"Saved plots to: {out_dir}")


# =========================
# Dataset table
# =========================

def compute_dataset_metrics_per_step_mean(per_image_metrics: Dict[int, Dict[int, Dict[str, Any]]], num_steps: int) -> Dict[int, Dict[str, Any]]:
    keys = ["BLEU_1", "BLEU_2", "ROUGE_L", "CIDEr", "METEOR", "SPICE"]
    out = {}
    for step in range(num_steps):
        step_dict = per_image_metrics.get(step, {})
        img_ids = sorted(step_dict.keys())
        row = {"num_images": len(img_ids)}
        for k in keys:
            vals = []
            for img_id in img_ids:
                v = step_dict[img_id].get(k, None)
                if v is None:
                    continue
                try:
                    vals.append(float(v))
                except Exception:
                    pass
            row[k] = float(np.mean(vals)) if len(vals) else None
        # BLEU avg12
        if row["BLEU_1"] is not None and row["BLEU_2"] is not None:
            row["BLEU_AVG12"] = 0.5 * (row["BLEU_1"] + row["BLEU_2"])
        else:
            row["BLEU_AVG12"] = None
        out[step] = row
    return out


def print_dataset_metrics_table(metrics: Dict[int, Dict[str, Any]]):
    headers = ["Step", "N", "BLEU1", "BLEU2", "BLEUavg12", "ROUGE_L", "CIDEr", "METEOR", "SPICE"]
    rows = []
    for step in sorted(metrics.keys()):
        m = metrics[step]

        def fmt(x):
            if x is None:
                return "NA"
            try:
                return f"{float(x):.4f}"
            except Exception:
                return "NA"

        rows.append([
            str(step),
            str(m.get("num_images", "")),
            fmt(m.get("BLEU_1")),
            fmt(m.get("BLEU_2")),
            fmt(m.get("BLEU_AVG12")),
            fmt(m.get("ROUGE_L")),
            fmt(m.get("CIDEr")),
            fmt(m.get("METEOR")),
            fmt(m.get("SPICE")),
        ])

    widths = [len(h) for h in headers]
    for r in rows:
        for j, cell in enumerate(r):
            widths[j] = max(widths[j], len(cell))

    def line(ch: str):
        return "┼" + "┼".join([ch * (w + 2) for w in widths]) + "┼"

    print(line("─"))
    print("│ " + " │ ".join([headers[j].ljust(widths[j]) for j in range(len(headers))]) + " │")
    print(line("═"))
    for r in rows:
        print("│ " + " │ ".join([r[j].ljust(widths[j]) for j in range(len(r))]) + " │")
    print(line("─"))


# =========================
# Resume helpers
# =========================

def metrics_file_complete(metrics_path: str, num_steps: int) -> bool:
    if not os.path.exists(metrics_path):
        return False
    try:
        with open(metrics_path, "r") as f:
            d = json.load(f)
        # must contain steps 0..num_steps-1
        for s in range(num_steps):
            if str(s) not in d:
                return False
            # must contain at least BLEU_1 as a sanity check
            if "BLEU_1" not in d[str(s)]:
                return False
        return True
    except Exception:
        return False


# =========================
# Per-image evaluation worker 
# =========================

def _eval_one_trace(
    coco: COCO,
    img_id: int,
    steps: List[Dict[str, Any]],
    num_steps: int,
    meteor_timeout_sec: int,
    spice_timeout_sec: int,
):
    metric_keys = ["BLEU_1", "BLEU_2", "ROUGE_L", "CIDEr", "METEOR", "SPICE"]
    per_step_out = {}

    rows = {k: [np.nan] * num_steps for k in metric_keys}
    vlm_row = [np.nan] * num_steps

    for s in range(min(num_steps, len(steps))):
        cap = str(steps[s].get("caption", "") or "")
        hint = steps[s].get("vlm_hint", {}) or {}
        try:
            vlm_row[s] = float(hint.get("score", np.nan))
        except Exception:
            vlm_row[s] = np.nan

        met = compute_metrics_one_caption(
            coco=coco,
            img_id=img_id,
            caption=cap,
            meteor_timeout_sec=meteor_timeout_sec,
            spice_timeout_sec=spice_timeout_sec,
        )
        per_step_out[str(s)] = met

        for k in metric_keys:
            v = met.get(k, None)
            if v is None:
                continue
            try:
                rows[k][s] = float(v)
            except Exception:
                pass

    return per_step_out, rows, vlm_row


# =========================
# Main
# =========================

def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("-r", "--results_dir", type=str, required=True)
    p.add_argument("-a", "--ann_file", type=str, required=True)
    p.add_argument("-n", "--num_steps", type=int, default=10)
    p.add_argument("-j", "--java11_bin", type=str, default="/usr/lib/jvm/java-11-openjdk-amd64/bin/java")

    p.add_argument("--spice_timeout_sec", type=int, default=180)
    p.add_argument("--meteor_timeout_sec", type=int, default=60)
    p.add_argument("--image_timeout_sec", type=int, default=240)

    p.add_argument("--refresh", action="store_true", help="Recompute metrics even if metrics file exists")

    args = p.parse_args()

    logger = setup_logger(args.results_dir)

    traces_dir = os.path.join(args.results_dir, "per_image_traces")
    metrics_dir = os.path.join(args.results_dir, "per_image_metrics")
    plots_dir = os.path.join(args.results_dir, "plots")
    spice_cache_dir = os.path.join(args.results_dir, "spice_cache")

    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(spice_cache_dir, exist_ok=True)

    if not os.path.isdir(traces_dir):
        raise FileNotFoundError(f"Missing traces dir: {traces_dir}")
    if not os.path.exists(args.java11_bin):
        raise FileNotFoundError(f"Java 11 not found at: {args.java11_bin}")

    # Patch SPICE 
    patch_spice_to_use_java(args.java11_bin, spice_cache_dir, logger, args.spice_timeout_sec)

    # Load COCO
    coco = COCO(args.ann_file)

    # Load trace file list
    trace_files = sorted([f for f in os.listdir(traces_dir) if f.endswith(".json")])
    logger.info(f"Found {len(trace_files)} trace files")

    # per_image_metrics[step][img_id] = metrics dict
    per_image_metrics: Dict[int, Dict[int, Dict[str, Any]]] = {s: {} for s in range(args.num_steps)}

    # Matrices for plots
    metric_keys_plot = ["BLEU_1", "BLEU_2", "ROUGE_L", "METEOR", "SPICE"]
    mats = {k: [] for k in metric_keys_plot}
    vlm_mat = []

    done = 0
    skipped = 0
    failed = 0
    t0 = time.time()

    for idx, fn in enumerate(trace_files, start=1):
        fp = os.path.join(traces_dir, fn)
        with open(fp, "r") as f:
            tr = json.load(f)

        img_id = int(tr["image_id"])
        steps = tr.get("steps", []) or []
        out_metrics_path = os.path.join(metrics_dir, f"{img_id}_metrics.json")

        if (not args.refresh) and metrics_file_complete(out_metrics_path, args.num_steps):
            # still load it into arrays/table so plotting uses all images
            skipped += 1
            try:
                with open(out_metrics_path, "r") as f:
                    per_step_out = json.load(f)
            except Exception:
                # if load fails, recompute
                per_step_out = None

            if isinstance(per_step_out, dict):
                # build arrays from saved metrics (fast)
                rows = {k: [np.nan] * args.num_steps for k in metric_keys_plot}
                vlm_row = [np.nan] * args.num_steps
                for s in range(min(args.num_steps, len(steps))):
                    hint = steps[s].get("vlm_hint", {}) or {}
                    try:
                        vlm_row[s] = float(hint.get("score", np.nan))
                    except Exception:
                        vlm_row[s] = np.nan

                    met = per_step_out.get(str(s), {}) or {}
                    for k in metric_keys_plot:
                        v = met.get(k, None)
                        if v is None:
                            continue
                        try:
                            rows[k][s] = float(v)
                        except Exception:
                            pass

                    
                    per_image_metrics[s][img_id] = met

                for k in metric_keys_plot:
                    mats[k].append(rows[k])
                vlm_mat.append(vlm_row)
                continue

        logger.info(f"[{idx}/{len(trace_files)}] Evaluating img_id={img_id}")

        ok, payload = run_with_timeout(
            _eval_one_trace,
            (coco, img_id, steps, args.num_steps, args.meteor_timeout_sec, args.spice_timeout_sec),
            args.image_timeout_sec,
        )

        if not ok:
            failed += 1
            logger.warning(f"SKIP img_id={img_id} reason={payload}")
            continue

        per_step_out, rows, vlm_row = payload

        # Save per-image metrics JSON
        with open(out_metrics_path, "w") as f:
            json.dump(per_step_out, f, indent=2)

        # Store for dataset mean and plotting
        for s in range(args.num_steps):
            met = per_step_out.get(str(s), None)
            if met is not None:
                per_image_metrics[s][img_id] = met

        for k in metric_keys_plot:
            mats[k].append(rows[k])
        vlm_mat.append(vlm_row)

        done += 1

        if (idx % 100) == 0:
            elapsed = (time.time() - t0) / 60.0
            logger.info(f"Progress {idx}/{len(trace_files)} | done={done} skipped={skipped} failed={failed} | elapsed_min={elapsed:.1f}")

    # Convert plot mats to numpy
    for k in metric_keys_plot:
        mats[k] = np.array(mats[k], dtype=float)
    vlm_mat = np.array(vlm_mat, dtype=float)

    # Dataset-level metrics mean per step
    dataset_metrics = compute_dataset_metrics_per_step_mean(per_image_metrics, args.num_steps)
    out_path = os.path.join(args.results_dir, "dataset_metrics_per_step.json")
    with open(out_path, "w") as f:
        json.dump({str(k): v for k, v in dataset_metrics.items()}, f, indent=2)

    print("\nDATASET-LEVEL METRICS PER STEP (mean over images)")
    print_dataset_metrics_table(dataset_metrics)
    logger.info(f"Saved dataset-level metrics to: {out_path}")

    # Plots
    plot_part_A_B_from_arrays(
        out_dir=plots_dir,
        num_steps=args.num_steps,
        mats=mats,
        vlm_scores=vlm_mat,
        logger=logger,
    )

    elapsed = (time.time() - t0) / 60.0
    logger.info(f"Done. computed={done} skipped={skipped} failed={failed} elapsed_min={elapsed:.1f}")
    logger.info(f"Metrics dir: {metrics_dir}")
    logger.info(f"Plots dir: {plots_dir}")


if __name__ == "__main__":
    main()


# python plot_and_dataset_eval.py \
#   -r hint_only_results_qwenllm \
#   -a /home/mahyar/UMD_FinalProject/data/captions_val2017.json \
#   -n 10 \
#   -j /usr/lib/jvm/java-11-openjdk-amd64/bin/java \
#   --spice_timeout_sec 180 \
#   --meteor_timeout_sec 60 \
#   --image_timeout_sec 240
