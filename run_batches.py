"""
run_generation_only.py

Generates captions + VLM objects/hints for a selected number of COCO val images.
NO METRICS. NO PLOTS. NO JAVA.

Outputs (in results_dir):
- per_image_traces/<img_id>.json    (full trace: objects + captions + hints per step)

RESUME BEHAVIOR:
- If per_image_traces/<img_id>.json already exists, the image is skipped.

Example:
python run_generation_only.py \
  --coco_images_dir /path/to/val2017 \
  --ann_file /path/to/captions_val2017.json \
  --results_dir hint_only_results_qwenllm \
  --max_images 1000 \
  --num_steps 10
"""

import os
import re
import json
import time
import logging
import tempfile
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from pycocotools.coco import COCO

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import AutoProcessor, AutoModelForVision2Seq


# =========================
# Config
# =========================

@dataclass
class Config:
    coco_images_dir: str
    ann_file: str

    # VLM (image+text)
    qwen_vl_name: str = "Qwen/Qwen2-VL-7B-Instruct"

    # LLM (text-only)
    qwen_llm_name: str = "Qwen/Qwen2.5-7B-Instruct"

    dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    max_images: Optional[int] = 10
    num_steps: int = 10
    max_objects: int = 20

    results_dir: str = "hint_only_results_qwenllm"
    traces_subdir: str = "per_image_traces"

    num_workers: int = 2


# =========================
# Logging
# =========================

def setup_logger(results_dir: str) -> logging.Logger:
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "run_generation.log")

    logger = logging.getLogger("GEN_ONLY")
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
# Dataset
# =========================

class CocoValDataset(Dataset):
    def __init__(self, cfg: Config):
        self.coco = COCO(cfg.ann_file)
        self.img_ids = self.coco.getImgIds()
        if cfg.max_images is not None:
            self.img_ids = self.img_ids[: cfg.max_images]
        self.img_dir = cfg.coco_images_dir

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int):
        img_id = int(self.img_ids[idx])
        info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, info["file_name"])
        image = Image.open(img_path).convert("RGB")
        return img_id, img_path, image


# =========================
# Model loading
# =========================

def load_qwen_text_llm(cfg: Config):
    tok = AutoTokenizer.from_pretrained(cfg.qwen_llm_name, trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id

    mdl = AutoModelForCausalLM.from_pretrained(
        cfg.qwen_llm_name,
        torch_dtype=cfg.dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    mdl.eval()
    return tok, mdl


def load_qwen_vl(cfg: Config):
    proc = AutoProcessor.from_pretrained(cfg.qwen_vl_name, trust_remote_code=True)
    mdl = AutoModelForVision2Seq.from_pretrained(
        cfg.qwen_vl_name,
        torch_dtype=cfg.dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    mdl.eval()
    return proc, mdl


# =========================
# Robust list sanitation 
# =========================

def ensure_list(x) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    if isinstance(x, str):
        s = x.strip()
        return [s] if s else []
    if isinstance(x, (int, float, bool)):
        return [x]
    return [x]


def list_of_strings(x) -> List[str]:
    items = ensure_list(x)
    out: List[str] = []
    for it in items:
        s = str(it).strip()
        if s:
            out.append(s)
    return out


# =========================
# Text helpers
# =========================

def extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def safe_strip(s: str, max_len: int = 350) -> str:
    s = (s or "").strip()
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def decode_new_tokens_only(tokenizer, input_ids: torch.Tensor, output_ids: torch.Tensor) -> str:
    prompt_len = input_ids.shape[-1]
    gen_ids = output_ids[0, prompt_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def clean_caption_text(raw: str) -> str:
    if raw is None:
        return ""
    t = raw.strip()
    t = re.sub(r"^\s*Caption\s*:\s*", "", t, flags=re.IGNORECASE)

    cut_markers = [
        "#####", "Step ", "STEP ", "Reasoning:", "Analysis:",
        "Current caption score", "Current caption coherence"
    ]
    for mk in cut_markers:
        idx = t.find(mk)
        if idx != -1:
            t = t[:idx].strip()

    t = t.splitlines()[0].strip()
    t = re.sub(r"\s+", " ", t).strip()
    t = t.strip(' "\'')
    return t


def extract_single_caption_from_generation(gen_text: str) -> str:
    if not gen_text:
        return ""
    t = gen_text.strip()

    if t.startswith("{") and "caption" in t.lower():
        try:
            m = re.search(r"\{.*\}", t, re.DOTALL)
            if m:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict) and "caption" in obj:
                    return clean_caption_text(str(obj["caption"]))
        except Exception:
            pass

    lines = [l.strip() for l in t.splitlines() if l.strip()]
    if not lines:
        return ""
    return clean_caption_text(lines[0])


# =========================
# Qwen2-VL calls
# =========================

def qwen_vl_generate_json(
    processor,
    model,
    image: Image.Image,
    system_text: str,
    user_text: str,
    max_new_tokens: int = 256,
) -> Tuple[Optional[Dict[str, Any]], str]:
    from qwen_vl_utils import process_vision_info

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user",
         "content": [{"type": "image", "image": image}, {"type": "text", "text": user_text}]},
    ]

    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[prompt_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    out = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
    return extract_json_block(out), out


def qwen_vl_extract_objects_only(processor, model, image: Image.Image, max_objects: int) -> Dict[str, Any]:
    system_text = (
        "You are a visual object inventory tool. Only list objects that are clearly visible and important for a short and high-level image captioning. "
        "Return JSON with key 'objects' as a list of strings. Do not add any other keys. "
        "Do not describe the scene. Do not describe actions."
    )
    user_text = f"List up to {max_objects} visible objects. Return only JSON."
    data, raw = qwen_vl_generate_json(processor, model, image, system_text, user_text, max_new_tokens=256)

    if isinstance(data, dict) and isinstance(data.get("objects", None), list):
        objs = [str(x).strip() for x in data["objects"] if str(x).strip()]
        return {"objects": objs[:max_objects], "raw": raw}
    return {"objects": [], "raw": raw}


def qwen_vl_hint_feedback_no_spoilers(processor, model, image: Image.Image, objects: List[str], caption: str) -> Dict[str, Any]:
    obj_text = ", ".join(objects) if objects else "unknown"

    system_text = (
        "You are a strict caption critic who must follow two rules:\n"
        "Rule 1: Never quote, repeat, or reveal the candidate caption text.\n"
        "Rule 2: Never provide a correct caption or a full description of the image.\n"
        "Give minimal hints only.\n"
        "Return JSON with keys:\n"
        "missing_objects, hallucinated_objects, genericity_hint, edit_instructions, score (0-10)\n"
        "No other keys."
    )

    user_text = (
        f"Provided object list: [{obj_text}]\n"
        "Candidate caption is provided but you must not quote it.\n"
        f"Candidate caption: {caption}\n"
        "Return only JSON."
    )

    data, raw = qwen_vl_generate_json(processor, model, image, system_text, user_text, max_new_tokens=256)

    out = {
        "missing_objects": [],
        "hallucinated_objects": [],
        "genericity_hint": "",
        "edit_instructions": [],
        "score": 0,
        "raw": raw,
    }

    if isinstance(data, dict):
        out["missing_objects"] = list_of_strings(data.get("missing_objects", []))
        out["hallucinated_objects"] = list_of_strings(data.get("hallucinated_objects", []))
        out["genericity_hint"] = str(data.get("genericity_hint", "") or "").strip()
        out["edit_instructions"] = list_of_strings(data.get("edit_instructions", []))
        try:
            out["score"] = int(data.get("score", 0))
        except Exception:
            out["score"] = 0

    out["score"] = int(max(0, min(10, out["score"])))
    return out


# =========================
# Qwen TEXT-ONLY LLM
# =========================

def qwen_llm_generate_one_caption(qwen_tok, qwen_llm, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    chat_prompt = qwen_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = qwen_tok(chat_prompt, return_tensors="pt").to(qwen_llm.device)

    with torch.no_grad():
        out_ids = qwen_llm.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.3,
            top_p=0.95,
            pad_token_id=qwen_tok.pad_token_id,
            eos_token_id=qwen_tok.eos_token_id,
        )

    gen_text = decode_new_tokens_only(qwen_tok, inputs["input_ids"], out_ids)
    return extract_single_caption_from_generation(gen_text)


def qwen_llm_generate_single_caption_from_objects(qwen_tok, qwen_llm, objects: List[str]) -> str:
    obj_text = ", ".join(objects) if objects else "unknown"

    prompt = (
        "You do not see the image. You only see a list of objects visible in the image.\n"
        "Write exactly one COCO-style caption with a focus on the general idea of the image.\n"
        "Style rules:\n"
        "1) One sentence, 6 to 10 words.\n"
        "2) Natural and human-like, not a list.\n"
        "3) Mention only 2 to 4 most salient objects.\n"
        "4) Use only objects from the list, no new objects.\n"
        "5) Avoid commas and avoid enumerations.\n"
        "6) Prefer high-level semantics (what the photo is about).\n"
        f"Objects: [{obj_text}]\n"
        "Caption:"
    )

    cap = clean_caption_text(qwen_llm_generate_one_caption(qwen_tok, qwen_llm, prompt))
    if not cap:
        cap = "A photo of " + (objects[0] if objects else "something") + "."
    return cap


def qwen_llm_refine_single_caption_from_hints(qwen_tok, qwen_llm, objects: List[str], current_caption: str, hint_json: Dict[str, Any]) -> str:
    obj_text = ", ".join(objects) if objects else "unknown"

    missing = list_of_strings(hint_json.get("missing_objects", []))
    halluc = list_of_strings(hint_json.get("hallucinated_objects", []))
    genericity = str(hint_json.get("genericity_hint", "") or "")
    edits = list_of_strings(hint_json.get("edit_instructions", []))

    hint_block = (
        f"Missing objects to consider: {', '.join(missing) if missing else 'none'}\n"
        f"Hallucinated objects to remove: {', '.join(halluc) if halluc else 'none'}\n"
        f"Genericity hint: {genericity if genericity else 'none'}\n"
        f"Edit instructions: {', '.join(edits) if edits else 'none'}\n"
    )

    prompt = (
        "You do not see the image. You only have an object list and minimal hints.\n"
        "Rewrite the caption to better match the image.\n"
        "Style rules:\n"
        "1) One sentence, 6 to 10 words.\n"
        "2) Natural and COCO-like, not a list.\n"
        "3) Mention only 2 to 4 most salient objects.\n"
        "4) Use only objects from the object list.\n"
        "5) Remove hallucinated objects.\n"
        "6) Optionally add missing objects if important.\n"
        "7) Avoid commas and avoid enumerations.\n"
        f"Objects: [{obj_text}]\n"
        f"Current caption: {current_caption}\n"
        f"Hints:\n{hint_block}\n"
        "Rewritten caption:"
    )

    cap = clean_caption_text(qwen_llm_generate_one_caption(qwen_tok, qwen_llm, prompt))
    return cap if cap else current_caption


# =========================
# Atomic JSON write 
# =========================

def atomic_write_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=os.path.dirname(path), suffix=".tmp") as tf:
        json.dump(obj, tf, indent=2)
        tmp_name = tf.name
    os.replace(tmp_name, path)


# =========================
# Per image loop
# =========================

def run_for_one_image(
    img_id: int,
    img_path: str,
    image: Image.Image,
    qwen_vl_proc,
    qwen_vl_model,
    qwen_llm_tok,
    qwen_llm_model,
    cfg: Config,
) -> Dict[str, Any]:
    trace: Dict[str, Any] = {
        "image_id": img_id,
        "image_path": img_path,
        "objects": [],
        "template": "",
        "steps": [],
        "summary": {},
    }

    t0 = time.time()

    obj_pack = qwen_vl_extract_objects_only(qwen_vl_proc, qwen_vl_model, image, cfg.max_objects)
    objects = obj_pack["objects"]
    trace["objects"] = objects
    trace["template"] = f"We have an image with these objects: {objects}. Generate a caption based only on these objects."
    trace["qwen_objects_raw"] = obj_pack.get("raw", "")

    caption = safe_strip(qwen_llm_generate_single_caption_from_objects(qwen_llm_tok, qwen_llm_model, objects))

    for step in range(cfg.num_steps):
        hint = qwen_vl_hint_feedback_no_spoilers(qwen_vl_proc, qwen_vl_model, image, objects, caption)

        trace["steps"].append({
            "step": step,
            "caption": caption,
            "vlm_hint": {
                "missing_objects": hint.get("missing_objects", []),
                "hallucinated_objects": hint.get("hallucinated_objects", []),
                "genericity_hint": hint.get("genericity_hint", ""),
                "edit_instructions": hint.get("edit_instructions", []),
                "score": hint.get("score", 0),
                "raw": hint.get("raw", ""),
            },
        })

        if step < cfg.num_steps - 1:
            caption = safe_strip(qwen_llm_refine_single_caption_from_hints(
                qwen_llm_tok, qwen_llm_model, objects, caption, hint
            ))

    trace["summary"] = {
        "image_id": img_id,
        "image_path": img_path,
        "final_caption": trace["steps"][-1]["caption"] if trace["steps"] else "",
        "total_time_sec": round(time.time() - t0, 2),
    }
    return trace


# =========================
# Main
# =========================

def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--coco_images_dir", type=str, required=True)
    p.add_argument("--ann_file", type=str, required=True)
    p.add_argument("--results_dir", type=str, default="hint_only_results_qwenllm")
    p.add_argument("--max_images", type=int, default=10)
    p.add_argument("--num_steps", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=2)
    args = p.parse_args()

    cfg = Config(
        coco_images_dir=args.coco_images_dir,
        ann_file=args.ann_file,
        results_dir=args.results_dir,
        max_images=args.max_images,
        num_steps=args.num_steps,
        num_workers=args.num_workers,
    )

    logger = setup_logger(cfg.results_dir)

    traces_dir = os.path.join(cfg.results_dir, cfg.traces_subdir)
    os.makedirs(traces_dir, exist_ok=True)

    logger.info("Loading dataset")
    dataset = CocoValDataset(cfg)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=lambda b: b,
    )
    logger.info(f"Dataset size (requested): {len(dataset)}")

    # -------------------------
    # RESUME SANITY CHECK
    # -------------------------
    existing = set()
    for fn in os.listdir(traces_dir):
        if fn.endswith(".json"):
            stem = fn[:-5]
            if stem.isdigit():
                existing.add(int(stem))

    requested_ids = list(dataset.img_ids)  # CocoValDataset stores in-order
    requested_set = set(int(x) for x in requested_ids)

    already_done = len(existing.intersection(requested_set))
    remaining = len(requested_set) - already_done

    logger.info(f"Resume check: {already_done}/{len(requested_set)} traces already exist in {traces_dir}")
    logger.info(f"Resume check: {remaining} images remaining to generate")

  
    preview_missing = []
    for img_id in requested_ids:
        if int(img_id) not in existing:
            preview_missing.append(int(img_id))
        if len(preview_missing) >= 10:
            break
    if preview_missing:
        logger.info(f"First missing image_ids to process (preview): {preview_missing}")
    else:
        logger.info("All requested images already have traces. Nothing to do.")
        return

    logger.info("Loading Qwen TEXT-ONLY LLM")
    qwen_llm_tok, qwen_llm_model = load_qwen_text_llm(cfg)

    logger.info("Loading Qwen VLM")
    qwen_vl_proc, qwen_vl_model = load_qwen_vl(cfg)

    logger.info("Starting generation loop with RESUME (skip if trace exists)")
    total = len(dataset)

    n_skipped = 0
    n_done = 0
    t_start = time.time()

    for i, batch in enumerate(loader, start=1):
        img_id, img_path, image = batch[0]
        img_id = int(img_id)

        out_path = os.path.join(traces_dir, f"{img_id}.json")

        # Resume behavior: skip if already exists
        if os.path.exists(out_path):
            n_skipped += 1
            if (n_skipped % 50) == 0:
                elapsed = time.time() - t_start
                logger.info(f"Skipped={n_skipped} | Done={n_done} | elapsed_min={elapsed/60:.1f}")
            continue

        logger.info(f"{i}/{total} generating image_id={img_id}")

        trace = run_for_one_image(
            img_id=img_id,
            img_path=img_path,
            image=image,
            qwen_vl_proc=qwen_vl_proc,
            qwen_vl_model=qwen_vl_model,
            qwen_llm_tok=qwen_llm_tok,
            qwen_llm_model=qwen_llm_model,
            cfg=cfg,
        )

        atomic_write_json(out_path, trace)
        n_done += 1

        print(json.dumps(trace["summary"], indent=2))

        # lightweight progress heartbeat
        if (n_done % 25) == 0:
            elapsed = time.time() - t_start
            logger.info(f"Progress: done={n_done}, skipped={n_skipped}, elapsed_min={elapsed/60:.1f}")

    elapsed = time.time() - t_start
    logger.info(f"Done. New traces written: {n_done}, skipped: {n_skipped}, elapsed_min={elapsed/60:.1f}")
    logger.info(f"Traces folder: {traces_dir}")


if __name__ == "__main__":
    main()


# python run_batches.py \
#   --coco_images_dir /home/mahyar/UMD_FinalProject/data/val2017 \
#   --ann_file /home/mahyar/UMD_FinalProject/data/captions_val2017.json \
#   --results_dir hint_only_results_qwenllm \
#   --max_images 1000 \
#   --num_steps 10
