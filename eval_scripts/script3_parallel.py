"""
Parallel, faster evaluator for feature localization experiments.

What’s new vs script3.py
- Parallel over dataset items with --max-workers
- Optional parallel over runs per item with --runs-workers
- Same metrics/CSV columns as the original
- Safe fallback to sequential when --stop-on-fp (interactive adjudication)

Example:
  python script3_parallel.py \
      --dataset oskardropship-dataset-30Sep.json \
      --code-context oskardropship-gptree.txt \
      --model gpt-4o-mini \
      --outdir runs \
      --line-tolerance 10 \
      --num-runs 5 \
      --max-workers 8 \
      --runs-workers 1 \
      --use-precise vague|precise

Requires:
  pip install python-dotenv openai (>=1.30), numpy
  .env with OPENAI_API_KEY=...
"""

import argparse
import csv
import json
import os
import re
import sys
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from dotenv import load_dotenv

# ----------- OpenAI client (API v1)
try:
    from openai import OpenAI
except Exception:
    print("Please install openai>=1.30: pip install --upgrade openai", file=sys.stderr)
    raise


# ---------- Helpers

def load_text(path: str, max_chars: Optional[int] = None) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        txt = f.read()
    if max_chars and len(txt) > max_chars:
        head = txt[: max_chars // 2]
        tail = txt[-max_chars // 2 :]
        txt = head + "\\n\\n...[TRUNCATED]...\\n\\n" + tail
    return txt


def parse_line_range(s: Optional[str]) -> Optional[Tuple[int, int]]:
    """
    Accepts "147-153", "147", "any", "unknown", "", None.
    Returns (start, end) inclusive, or None if unknown/any.
    """
    if s is None:
        return None
    if isinstance(s, str):
        z = s.strip().lower()
        if z in {"", "any", "unknown", "n/a", "null", "none"}:
            return None
        m = re.match(r"^\\s*(\\d+)\\s*-\\s*(\\d+)\\s*$", z)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a > b:
                a, b = b, a
            return (a, b)
        if z.isdigit():
            v = int(z)
            return (v, v)
    if isinstance(s, (int, float)):
        v = int(s)
        return (v, v)
    return None


def within_tolerance(pred: Optional[int], gold_range: Optional[Tuple[int,int]], tol: int) -> bool:
    """
    - If gold_range is None -> always True (anywhere in file).
    - If pred is None -> False (no point to compare).
    - Else True if pred inside [start, end] or within ±tol of nearest bound.
    """
    if gold_range is None:
        return True
    if pred is None:
        return False
    start, end = gold_range
    if start <= pred <= end:
        return True
    delta = min(abs(pred - start), abs(pred - end))
    return delta <= tol


def ranges_overlap_with_tol(r1: Optional[Tuple[int,int]], r2: Optional[Tuple[int,int]], tol: int) -> bool:
    if r1 is None or r2 is None:
        return False
    a1, b1 = r1
    a2, b2 = r2
    # Expand both by tol on each side
    a1e, b1e = a1 - tol, b1 + tol
    a2e, b2e = a2 - tol, b2 + tol
    return not (b1e < a2e or b2e < a1e)


def extract_json_list(text: str) -> List[Dict[str, Any]]:
    """Pull the first top-level JSON list from the model response."""
    # Fast path
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass
    # Fallback: slice between first '[' and last ']'
    first = text.find('[')
    last = text.rfind(']')
    if first != -1 and last != -1 and last > first:
        snippet = text[first:last+1]
        try:
            obj = json.loads(snippet)
            if isinstance(obj, list):
                return obj
        except Exception:
            return []
    return []


def to_csv_safe_list(xs: List[Any]) -> str:
    return ";".join(map(str, xs))


def fmt_line_range(r: Optional[Tuple[int, int]]) -> str:
    if r is None:
        return "any"
    a, b = r
    return f"{a}" if a == b else f"{a}-{b}"


# === Prompt / API =============================================================

def build_task_text(tag: str,
                    superfeature: str,
                    feature_desc: str,
                    user_story: str = "",
                    use_case: str = "",
                    feature_specification: str = "") -> str:
    parts = [
        f"You are a senior software engineer assisting in feature localization.",
        f"Given the repository below, locate where **{tag}** of a feature **{superfeature}**: **{feature_desc}** needs to take place.",
        'Answer **only** with a JSON array, each element an object with keys ["file_path","line_number"].',
        'If the edit can be done anywhere in a given file, set "line_number" to null or "any".',
    ]
    if user_story or use_case or feature_specification:
        parts.append("\\nContext:")
        if user_story:
            parts.append(f"- User story: {user_story}")
        if use_case:
            parts.append(f"- Use case: {use_case}")
        if feature_specification:
            parts.append(f"- Spec: {feature_specification}")
    return "\\n".join(parts)


def prompt_template(tag: str,
                    superfeature: str,
                    feature_desc: str,
                    code_context: str,
                    user_story: str = "",
                    use_case: str = "",
                    feature_specification: str = "") -> List[Dict[str, str]]:
    task = build_task_text(tag, superfeature, feature_desc, user_story, use_case, feature_specification)
    user_content = task + "\\n\\n" + "Codebase context (gptree_output.txt excerpt):\\n" + code_context
    return [
        {"role": "system", "content": "You are a precise code navigation assistant. Reply only in JSON as instructed."},
        {"role": "user", "content": user_content},
    ]


def call_openai(client: OpenAI, model: str, messages: List[Dict[str, str]]) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"} if model.startswith("gpt-4.1") else None,
    )
    return resp.choices[0].message.content or ""


# === Atomic helpers ===========================================================

def gold_buckets_for_item(required_paths: List[str],
                          optional_paths: List[str],
                          gold_lines_map: Dict[str, Optional[Tuple[int, int]]],
                          optional_map: Dict[str, bool]) -> Dict[str, Tuple[Optional[Tuple[int,int]], bool]]:
    buckets: Dict[str, Tuple[Optional[Tuple[int,int]], bool]] = {}
    for p in set(required_paths + optional_paths):
        buckets[p] = (gold_lines_map.get(p, None), bool(optional_map.get(p, False)))
    return buckets


def atomic_hit_status(path: str,
                      pred_point: Optional[int],
                      pred_range: Optional[Tuple[int,int]],
                      buckets: Dict[str, Tuple[Optional[Tuple[int,int]], bool]],
                      tol: int) -> str:
    """
    Return 'required', 'optional', or 'fp' for a single predicted path.
    """
    if path not in buckets:
        return 'fp'
    gold_range, is_opt = buckets[path]
    if gold_range is None:  # any line is ok
        return 'optional' if is_opt else 'required'

    # prefer any overlapping predicted range
    if pred_range is not None and ranges_overlap_with_tol(pred_range, gold_range, tol):
        return 'optional' if is_opt else 'required'
    # else fall back to point-within-tol
    if within_tolerance(pred_point, gold_range, tol):
        return 'optional' if is_opt else 'required'
    return 'fp'


def parse_pred_line_or_range(value):
    """
    Returns a tuple: (point_or_None, range_or_None_as_tuple)
      - int/float -> (int, None)
      - "null"/"none"/"any"/"" -> (None, None)
      - "a-b" -> (midpoint_int, (a,b))
      - "123" -> (123, None)
      - else -> (None, None)
    """
    if value is None:
        return None, None
    if isinstance(value, (int, float)):
        return int(value), None
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"null", "none", "any", ""}:
            return None, None
        m = re.match(r"^\\s*(\\d+)\\s*-\\s*(\\d+)\\s*$", s)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a > b:
                a, b = b, a
            mid = (a + b) // 2
            return mid, (a, b)
        if s.isdigit():
            return int(s), None
    return None, None


def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """
    HumanEval-style estimator.
    """
    n, c = int(num_samples), int(num_correct)
    if n - c < k:
        return 1.0
    arr = np.arange(n - c + 1, n + 1, dtype=float)
    return float(1.0 - np.prod(1.0 - k / arr))


# === Dataset flattening =======================================================

def flatten_dataset(dataset: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Emit items with:
      g_idx, f_idx, superfeature, tag, feature_desc,
      user_story, use_case, feature_specification, spec_type,
      required_paths, optional_paths,
      gold_lines_map (path -> Optional[(start,end)]),
      optional_map (path -> bool)
    """
    items: List[Dict[str, Any]] = []
    for g_idx, group in enumerate(dataset):
        for f_idx, feat in enumerate(group.get("features", [])):
            fps = feat.get("file_paths", []) or []
            required_paths: List[str] = []
            optional_paths: List[str] = []
            lines_map: Dict[str, Optional[Tuple[int,int]]] = {}
            optional_map: Dict[str, bool] = {}
            for entry in fps:
                if isinstance(entry, str):
                    p = entry.strip()
                    if not p: 
                        continue
                    required_paths.append(p)
                    lines_map[p] = None
                    optional_map[p] = False
                elif isinstance(entry, dict):
                    p = entry.get("path", "").strip()
                    if not p:
                        continue
                    is_opt = bool(entry.get("optional", False))
                    ln = entry.get("lines", None)
                    required_paths.append(p) if not is_opt else optional_paths.append(p)
                    lines_map[p] = parse_line_range(ln)
                    optional_map[p] = is_opt
            items.append({
                "g_idx": g_idx,
                "f_idx": f_idx,
                "superfeature": group.get("superfeature", ""),
                "tag": feat.get("tag", ""),
                "feature_desc": feat.get("feature_desc", ""),
                "user_story": feat.get("user_story", ""),
                "use_case": feat.get("use_case", ""),
                "feature_specification": feat.get("feature_specification", ""),
                "spec_type": "precise" if (feat.get("user_story") or feat.get("use_case") or feat.get("feature_specification")) else "vague",
                "required_paths": required_paths,
                "optional_paths": optional_paths,
                "gold_lines_map": lines_map,
                "optional_map": optional_map,
            })
    return items


# === Per-item evaluation (optionally parallelizing internal runs) ============

def eval_item(
    item: Dict[str, Any],
    code_context: str,
    client: OpenAI,
    model: str,
    num_runs: int,
    line_tolerance: int,
    runs_workers: int,
    stop_on_fp: bool,
) -> Dict[str, Any]:
    """
    Return a dict with:
      - 'rows': List[csv_row_for_this_example]
      - 'agg': dict of macro contributions for this example
    """
    tag = item["tag"]
    superfeature = item["superfeature"]
    feature_desc = item["feature_desc"]
    user_story = item.get("user_story", "")
    use_case = item.get("use_case", "")
    feature_specification = item.get("feature_specification", "")
    required_paths: List[str] = item.get("required_paths", [])
    optional_paths: List[str] = item.get("optional_paths", [])
    gold_lines_map: Dict[str, Optional[Tuple[int,int]]] = item.get("gold_lines_map", {})
    optional_map: Dict[str, bool] = item.get("optional_map", {})

    # Build messages (vague vs precise)
    if item.get("spec_type") == "precise":
        short_prompt = build_task_text(tag, superfeature, feature_desc, user_story, use_case, feature_specification)
        messages = prompt_template(tag, superfeature, feature_desc, code_context, user_story, use_case, feature_specification)
        prompt_spec_mode = "precise"
    else:
        short_prompt = build_task_text(tag, superfeature, feature_desc)
        messages = prompt_template(tag, superfeature, feature_desc, code_context)
        prompt_spec_mode = "vague"

    # Launch runs (maybe parallel)
    def one_run_call(_: int) -> Dict[str, Any]:
        try:
            raw = call_openai(client, model, messages)
        except Exception as e:
            raw = ""
        preds = extract_json_list(raw)
        pred_paths: List[str] = []
        pred_points_map: Dict[str, Optional[int]] = {}
        pred_ranges_map: Dict[str, Optional[Tuple[int, int]]] = {}
        for obj in preds:
            fp = obj.get("file_path")
            point, rng = parse_pred_line_or_range(obj.get("line_number", None))
            if isinstance(fp, str) and fp.strip():
                path_clean = fp.strip()
                pred_paths.append(path_clean)
                pred_points_map[path_clean] = point
                pred_ranges_map[path_clean] = rng
        return {
            "raw": raw,
            "pred_set": set(pred_paths),
            "pred_points": pred_points_map,
            "pred_ranges": pred_ranges_map,
        }

    run_results: List[Dict[str, Any]] = []
    n = max(1, int(num_runs))

    if runs_workers > 1 and not stop_on_fp:
        # per-item parallel over runs
        with ThreadPoolExecutor(max_workers=runs_workers) as tp:
            futs = [tp.submit(one_run_call, r) for r in range(n)]
            for fut in as_completed(futs):
                run_results.append(fut.result())
    else:
        # sequential runs
        for r in range(n):
            run_results.append(one_run_call(r))

    # Gather per-run artifacts
    run_raws = [rr["raw"] for rr in run_results]
    run_pred_sets = [rr["pred_set"] for rr in run_results]
    run_pred_points = [rr["pred_points"] for rr in run_results]
    run_pred_ranges = [rr["pred_ranges"] for rr in run_results]

    # Per-run metrics (macro for this example)
    set_required: Set[str] = set(required_paths)
    set_optional: Set[str] = set(optional_paths)
    perrun_precisions: List[float] = []
    perrun_recalls: List[float] = []
    perrun_f1s: List[float] = []
    perrun_ems: List[int] = []

    buckets = gold_buckets_for_item(required_paths, optional_paths, gold_lines_map, optional_map)

    for set_pred, pred_points_map, pred_ranges_map in zip(run_pred_sets, run_pred_points, run_pred_ranges):
        tp_req = tp_opt = fp_ = 0
        matched_req: Set[str] = set()

        for p in set_pred:
            status = atomic_hit_status(
                p,
                pred_points_map.get(p, None),
                pred_ranges_map.get(p, None),
                buckets,
                line_tolerance
            )
            if status == 'required':
                tp_req += 1
                matched_req.add(p)
            elif status == 'optional':
                tp_opt += 1
            else:
                fp_ += 1

        fn_req = len(set_required - matched_req)
        total_pred = len(set_pred)
        precision = (tp_req + tp_opt) / total_pred if total_pred > 0 else 0.0
        recall = tp_req / len(set_required) if len(set_required) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        em = int(fp_ == 0 and fn_req == 0)

        perrun_precisions.append(precision)
        perrun_recalls.append(recall)
        perrun_f1s.append(f1)
        perrun_ems.append(em)

    ex_perrun_precision = float(np.mean(perrun_precisions)) if perrun_precisions else 0.0
    ex_perrun_recall = float(np.mean(perrun_recalls)) if perrun_recalls else 0.0
    ex_perrun_f1 = float(np.mean(perrun_f1s)) if perrun_f1s else 0.0
    ex_perrun_em = float(np.mean(perrun_ems)) if perrun_ems else 0.0

    # UNION metrics (atomic)
    union_pred_paths: Set[str] = set().union(*run_pred_sets) if run_pred_sets else set()

    adjudicated = False  # interactive adjudication is disabled in parallel path

    # Build path -> list of predicted points/ranges across runs
    path_to_points: Dict[str, List[Optional[int]]] = {p: [] for p in union_pred_paths}
    path_to_ranges: Dict[str, List[Optional[Tuple[int, int]]]] = {p: [] for p in union_pred_paths}
    for pts_map, rng_map in zip(run_pred_points, run_pred_ranges):
        for p in union_pred_paths:
            path_to_points[p].append(pts_map.get(p, None))
            path_to_ranges[p].append(rng_map.get(p, None))

    buckets_union = gold_buckets_for_item(required_paths, optional_paths, gold_lines_map, optional_map)

    union_hit_required: Set[str] = set()
    union_hit_optional: Set[str] = set()
    union_fp = 0

    for p in union_pred_paths:
        status = None
        if p in buckets_union:
            gold_range, is_opt = buckets_union[p]
            if gold_range is None:
                status = 'optional' if is_opt else 'required'
            else:
                # Any overlapping predicted range wins
                for pr in path_to_ranges[p]:
                    if pr is not None and ranges_overlap_with_tol(pr, gold_range, line_tolerance):
                        status = 'optional' if is_opt else 'required'
                        break
                # If no range hit, try any predicted point within tolerance
                if status is None:
                    for pt in path_to_points[p]:
                        if pt is not None and within_tolerance(pt, gold_range, line_tolerance):
                            status = 'optional' if is_opt else 'required'
                            break
        if status == 'required':
            union_hit_required.add(p)
        elif status == 'optional':
            union_hit_optional.add(p)
        else:
            union_fp += 1

    union_tp_required = len(union_hit_required)
    union_tp_optional = len(union_hit_optional)
    union_tp_total = union_tp_required + union_tp_optional
    union_fn_required = len(set_required - union_hit_required)

    denom_pred = len(union_pred_paths)
    union_precision = union_tp_total / denom_pred if denom_pred > 0 else 0.0
    denom_req = len(set_required)
    union_recall = union_tp_required / denom_req if denom_req > 0 else 0.0
    union_f1 = 2 * union_precision * union_recall / (union_precision + union_recall) if (union_precision + union_recall) > 0 else 0.0
    union_em = int(union_fp == 0 and union_fn_required == 0)

    # Pass@{1,3,5,k} (strict EM per run)
    n = max(1, num_runs)
    c = int(sum(perrun_ems))
    ks = [1, 3, 5, n]
    pass_scores = {}
    for kk in ks:
        kk_eff = min(n, kk)
        pass_scores[f"pass_at_{kk}"] = estimate_pass_at_k(num_samples=n, num_correct=c, k=kk_eff)

    # CSV row for this example
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    row = {
        "timestamp": timestamp,
        "dataset": "",  # filled in caller
        "model": "",    # filled in caller
        "superfeature": superfeature,
        "tag": tag,
        "feature_desc": feature_desc,
        "spec_type": item.get("spec_type"),
        "user_story": user_story,
        "use_case": use_case,
        "feature_specification": feature_specification,
        "prompt_spec_mode": prompt_spec_mode,
        "num_runs": n,
        "gold_required_paths": to_csv_safe_list(sorted(set_required)),
        "gold_optional_paths": to_csv_safe_list(sorted(set(optional_paths))),
        "gold_with_lines": ";".join(
            f"{p}@{fmt_line_range(gold_lines_map.get(p))}{('[opt]' if optional_map.get(p) else '')}"
            for p in sorted((set(required_paths) | set(optional_paths)))
        ),
        # Per-run macro (this example)
        "perrun_macro_precision": round(ex_perrun_precision, 4),
        "perrun_macro_recall_required": round(ex_perrun_recall, 4),
        "perrun_macro_f1": round(ex_perrun_f1, 4),
        "perrun_macro_em": round(ex_perrun_em, 4),
        # Union metrics (this example)
        "union_pred_paths": to_csv_safe_list(sorted(union_pred_paths)),
        "union_TP_total": union_tp_total,
        "union_TP_required": union_tp_required,
        "union_TP_optional": union_tp_optional,
        "union_FP": union_fp,
        "union_FN_required": union_fn_required,
        "union_precision": round(union_precision, 4),
        "union_recall_required": round(union_recall, 4),
        "union_f1": round(union_f1, 4),
        "union_em": round(union_em, 4),
        # Pass@k (this example)
        "pass_at_1": round(pass_scores["pass_at_1"], 4),
        "pass_at_3": round(pass_scores.get("pass_at_3", 0.0), 4),
        "pass_at_5": round(pass_scores.get("pass_at_5", 0.0), 4),
        "pass_at_k": round(pass_scores[f"pass_at_{n}"], 4),
        "adjudicated": adjudicated,
        "prompt_no_context": short_prompt,
        "raw_llm_response_all_runs": "\\n---\\n".join(run_raws[:10]),
        "perrun_em_vector": to_csv_safe_list([int(x) for x in perrun_ems]),
    }

    return {
        "row": row,
        "perrun_macro": (ex_perrun_precision, ex_perrun_recall, ex_perrun_f1, ex_perrun_em),
        "union_macro": (union_precision, union_recall, union_f1, float(union_em)),
        "pass_macro": (pass_scores["pass_at_1"], pass_scores.get("pass_at_3", 0.0), pass_scores.get("pass_at_5", 0.0), pass_scores[f"pass_at_{n}"]),
    }


# === Main =====================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Path to dataset JSON.")
    ap.add_argument("--code-context", required=True, help="Path to gptree_output.txt (or similar).")
    ap.add_argument("--model", required=True, help="OpenAI model name, e.g., gpt-4o, gpt-4o-mini, gpt-4.1-mini.")
    ap.add_argument("--outdir", default="runs", help="Directory to write CSV logs.")
    ap.add_argument("--line-tolerance", type=int, default=10, help="± line tolerance outside gold range.")
    ap.add_argument("--max-context-chars", type=int, default=120000, help="Truncate repo context to this many chars.")
    ap.add_argument("--num-runs", type=int, default=1, help="Number of independent runs per example (k).")
    ap.add_argument("--use-precise", choices=["vague", "precise", "all"], default="all", help="Eval subset.")
    ap.add_argument("--stop-on-fp", action="store_true", help="Interactive adjudication (disables parallel).")

    # Parallelism knobs
    ap.add_argument("--max-workers", type=int, default=8, help="Max parallel examples (threads).")
    ap.add_argument("--runs-workers", type=int, default=1, help="Max parallel runs per example (threads).")

    args = ap.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Missing OPENAI_API_KEY in .env", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # Load
    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset_obj = json.load(f)
    code_context = load_text(args.code_context, max_chars=args.max_context_chars)
    flat_items = flatten_dataset(dataset_obj)

    # Optional filtering for --use-precise
    if args.use_precise == "precise":
        flat_items = [it for it in flat_items if it.get("spec_type") == "precise"]
    elif args.use_precise == "vague":
        flat_items = [it for it in flat_items if it.get("spec_type") != "precise"]

    if not flat_items:
        print("No items found to evaluate after filtering.")
        # Create an empty CSV with headers so tooling doesn't break
        os.makedirs(args.outdir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]
        csv_path = os.path.join(args.outdir, f"eval_{dataset_name}_{args.model}_{timestamp}.csv")
        fieldnames = [
            "timestamp","dataset","model","superfeature","tag","feature_desc",
            "spec_type","user_story","use_case","feature_specification","prompt_spec_mode",
            "num_runs",
            "gold_required_paths","gold_optional_paths","gold_with_lines",
            "perrun_macro_precision","perrun_macro_recall_required","perrun_macro_f1","perrun_macro_em",
            "union_pred_paths","union_TP_total","union_TP_required","union_TP_optional",
            "union_FP","union_FN_required","union_precision","union_recall_required","union_f1","union_em",
            "pass_at_1","pass_at_3","pass_at_5","pass_at_k","adjudicated","prompt_no_context",
            "raw_llm_response_all_runs","perrun_em_vector"
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
        print(f"CSV written: {csv_path}")
        return

    os.makedirs(args.outdir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]
    csv_path = os.path.join(args.outdir, f"eval_{dataset_name}_{args.model}_{timestamp}.csv")

    # Aggregates (macro across examples)
    agg_perrun_precision = 0.0
    agg_perrun_recall = 0.0
    agg_perrun_f1 = 0.0
    agg_perrun_em = 0.0

    agg_union_precision = 0.0
    agg_union_recall = 0.0
    agg_union_f1 = 0.0
    agg_union_em = 0.0

    agg_pass_at_1 = 0.0
    agg_pass_at_3 = 0.0
    agg_pass_at_5 = 0.0
    agg_pass_at_k = 0.0

    rows: List[Dict[str, Any]] = []

    # Choose parallel strategy
    max_workers = max(1, int(args.max_workers))
    runs_workers = max(1, int(args.runs_workers))
    if args.stop_on_fp:
        print("--stop-on-fp enabled: running sequentially to support interactive adjudication.")
        max_workers = 1
        runs_workers = 1

    # Launch per-item tasks
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            fut_to_idx = {}

            for idx, item in enumerate(flat_items, 1):
                fut = pool.submit(
                    eval_item,
                    item,
                    code_context,
                    client,
                    args.model,
                    args.num_runs,
                    args.line_tolerance,
                    runs_workers,
                    args.stop_on_fp,
                )
                fut_to_idx[fut] = idx

            for fut in as_completed(fut_to_idx):
                idx = fut_to_idx[fut]
                try:
                    out = fut.result()
                except Exception as e:
                    print(f"[item #{idx}] failed: {e}", file=sys.stderr)
                    continue

                row = out["row"]
                row["dataset"] = dataset_name
                row["model"] = args.model
                rows.append(row)

                pr_p, pr_r, pr_f1, pr_em = out["perrun_macro"]
                un_p, un_r, un_f1, un_em = out["union_macro"]
                p1, p3, p5, pk = out["pass_macro"]

                agg_perrun_precision += pr_p
                agg_perrun_recall += pr_r
                agg_perrun_f1 += pr_f1
                agg_perrun_em += pr_em

                agg_union_precision += un_p
                agg_union_recall += un_r
                agg_union_f1 += un_f1
                agg_union_em += un_em

                agg_pass_at_1 += p1
                agg_pass_at_3 += p3
                agg_pass_at_5 += p5
                agg_pass_at_k += pk
    else:
        # Sequential over items
        for idx, item in enumerate(flat_items, 1):
            out = eval_item(
                item, code_context, client, args.model, args.num_runs,
                args.line_tolerance, runs_workers, args.stop_on_fp
            )
            row = out["row"]
            row["dataset"] = dataset_name
            row["model"] = args.model
            rows.append(row)

            pr_p, pr_r, pr_f1, pr_em = out["perrun_macro"]
            un_p, un_r, un_f1, un_em = out["union_macro"]
            p1, p3, p5, pk = out["pass_macro"]

            agg_perrun_precision += pr_p
            agg_perrun_recall += pr_r
            agg_perrun_f1 += pr_f1
            agg_perrun_em += pr_em

            agg_union_precision += un_p
            agg_union_recall += un_r
            agg_union_f1 += un_f1
            agg_union_em += un_em

            agg_pass_at_1 += p1
            agg_pass_at_3 += p3
            agg_pass_at_5 += p5
            agg_pass_at_k += pk

    # Write CSV
    fieldnames = [
        "timestamp","dataset","model","superfeature","tag","feature_desc",
        "spec_type","user_story","use_case","feature_specification","prompt_spec_mode",
        "num_runs",
        "gold_required_paths","gold_optional_paths","gold_with_lines",
        "perrun_macro_precision","perrun_macro_recall_required","perrun_macro_f1","perrun_macro_em",
        "union_pred_paths","union_TP_total","union_TP_required","union_TP_optional",
        "union_FP","union_FN_required","union_precision","union_recall_required","union_f1","union_em",
        "pass_at_1","pass_at_3","pass_at_5","pass_at_k","adjudicated","prompt_no_context",
        "raw_llm_response_all_runs","perrun_em_vector"
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

        # Aggregate row (macro averages across examples)
        m = max(1, len(rows))
        macro_perrun_precision = agg_perrun_precision / m
        macro_perrun_recall = agg_perrun_recall / m
        macro_perrun_f1 = agg_perrun_f1 / m
        macro_perrun_em = agg_perrun_em / m

        macro_union_precision = agg_union_precision / m
        macro_union_recall = agg_union_recall / m
        macro_union_f1 = agg_union_f1 / m
        macro_union_em = agg_union_em / m

        macro_pass1 = agg_pass_at_1 / m
        macro_pass3 = agg_pass_at_3 / m
        macro_pass5 = agg_pass_at_5 / m
        macro_passk = agg_pass_at_k / m

        w.writerow({
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "dataset": dataset_name,
            "model": args.model,
            "superfeature": "",
            "tag": "",
            "feature_desc": "",
            "spec_type": "",
            "user_story": "",
            "use_case": "",
            "feature_specification": "",
            "prompt_spec_mode": "",
            "num_runs": "",
            "gold_required_paths": "",
            "gold_optional_paths": "",
            "gold_with_lines": "",
            # Store macro averages in the same metric columns
            "perrun_macro_precision": round(macro_perrun_precision, 4),
            "perrun_macro_recall_required": round(macro_perrun_recall, 4),
            "perrun_macro_f1": round(macro_perrun_f1, 4),
            "perrun_macro_em": round(macro_perrun_em, 4),
            # Union columns carry macro union metrics
            "union_pred_paths": "",
            "union_TP_total": "",
            "union_TP_required": "",
            "union_TP_optional": "",
            "union_FP": "",
            "union_FN_required": "",
            "union_precision": round(macro_union_precision, 4),
            "union_recall_required": round(macro_union_recall, 4),
            "union_f1": round(macro_union_f1, 4),
            "union_em": round(macro_union_em, 4),
            # Pass@{1,3,5,k} macro
            "pass_at_1": round(macro_pass1, 4),
            "pass_at_3": round(macro_pass3, 4),
            "pass_at_5": round(macro_pass5, 4),
            "pass_at_k": round(macro_passk, 4),
            "adjudicated": "",
            "prompt_no_context": "AGGREGATE ROW",
            "raw_llm_response_all_runs": "",
            "perrun_em_vector": "",
        })

    print(f"CSV written: {csv_path}")


if __name__ == "__main__":
    main()
