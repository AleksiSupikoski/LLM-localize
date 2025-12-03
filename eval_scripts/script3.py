"""
Evaluate LLMs on locating implementation points for feature requests.

Core design:
- Atomic correctness: (filepath, line-range) must match. A prediction is correct iff:
    * file paths match AND
    * predicted line is inside the gold range, OR within ±tolerance of its bounds, OR gold lines are None/"any".
- Required vs Optional:
    * Recall counts ONLY REQUIRED gold locations (OPTIONAL never creates FNs).
    * Predictions hitting OPTIONAL gold count as TPs (affect precision), not recall.

Metrics (rank-free, multi-run):
- --num-runs K: query the model K times per example.
- Per-run macro (per example): average Precision/Recall/F1/EM over K runs, then macro-average over examples.
- Union metrics (per example): Precision/Recall/F1/EM on the union of unique predictions across K runs (atomic).
- Pass@{1,3,5,k} (per example): HumanEval-style estimator using c = #runs with EM=1 (atomic, strict: FN=0 & FP=0),
  macro-averaged over examples.

Also supports interactive adjudication (--stop-on-fp) applied to the UNION of FPs (adds as OPTIONAL).

Usage:
  python script.py \
      --dataset data.json \
      --code-context gptree_output.txt \
      --model gpt-4o-mini \
      --outdir runs \
      --line-tolerance 10 \
      --max-context-chars 120000 \
      --num-runs 5 \
      --use-precise vague|precise \
      [--stop-on-fp]

Requires:
  pip install python-dotenv openai (>=1.30), pandas (optional), numpy
  .env with OPENAI_API_KEY=...
"""

import argparse
import csv
import json
import os
import re
import sys
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Set

try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False

from dotenv import load_dotenv

# ----------- OpenAI client (API v1)
try:
    from openai import OpenAI
except Exception as e:
    print("Please install openai>=1.30: pip install --upgrade openai", file=sys.stderr)
    raise


# ---------- Helpers

def load_text(path: str, max_chars: Optional[int] = None) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        txt = f.read()
    if max_chars and len(txt) > max_chars:
        head = txt[: max_chars // 2]
        tail = txt[-max_chars // 2 :]
        txt = head + "\n\n...[TRUNCATED]...\n\n" + tail
    return txt


def parse_line_range(s: Optional[str]) -> Optional[Tuple[int, int]]:
    """
    Accepts "147-153", "147", "any", "unknown", "", None.
    Returns (start, end) inclusive, or None if unknown/any.
    """
    if s is None:
        return None
    if isinstance(s, (int, float)):
        n = int(s)
        return (n, n)
    s = str(s).strip().lower()
    if s in {"any", "unknown", "null", "n/a", ""}:
        return None
    m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if a > b:
            a, b = b, a
        return (a, b)
    m = re.match(r"^\s*(\d+)\s*$", s)
    if m:
        n = int(m.group(1))
        return (n, n)
    return None


def within_tolerance(pred: Optional[int], gold_range: Optional[Tuple[int, int]], tol: int) -> bool:
    """
    Atomic line-match predicate:
      - If gold_range is None -> accept any pred (including None).
      - If pred is None and gold_range exists -> mismatch.
      - Else match if pred within range or within ±tol of nearest bound.
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


def extract_json_list(text: str) -> List[Dict[str, Any]]:
    """Pull the first top-level JSON list from the model response."""
    # Fast path
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass
    # Fallback: bracket slice
    first = text.find('[')
    last = text.rfind(']')
    if first != -1 and last != -1 and last > first:
        snippet = text[first:last + 1]
        try:
            obj = json.loads(snippet)
            if isinstance(obj, list):
                return obj
        except Exception:
            pass
    return []


def _is_nonempty_str(x: Optional[str]) -> bool:
    return isinstance(x, str) and x.strip() != ""


def flatten_dataset(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Emit items with:
      g_idx, f_idx, superfeature, tag, feature_desc,
      user_story, use_case, feature_specification, spec_type,
      required_paths, optional_paths,
      gold_lines_map (path -> Optional[(start,end)]),
      optional_map (path -> bool)
    """
    items = []
    for g_idx, group in enumerate(dataset):
        for f_idx, feat in enumerate(group.get("features", [])):
            fps = feat.get("file_paths", []) or []
            required_paths: List[str] = []
            optional_paths: List[str] = []
            lines_map: Dict[str, Optional[Tuple[int, int]]] = {}
            optional_map: Dict[str, bool] = {}
            for fp in fps:
                p = fp.get("path")
                if not p:
                    continue
                opt = bool(fp.get("optional", False))
                rng = parse_line_range(fp.get("lines"))
                lines_map[p] = rng
                optional_map[p] = opt
                (optional_paths if opt else required_paths).append(p)

            user_story = feat.get("user_story")
            use_case = feat.get("use_case")
            feature_spec = feat.get("feature_specification")
            has_precise = all([
                _is_nonempty_str(user_story),
                _is_nonempty_str(use_case),
                _is_nonempty_str(feature_spec),
            ])

            items.append({
                "g_idx": g_idx,
                "f_idx": f_idx,
                "superfeature": group.get("superfeature"),
                "tag": feat.get("tag"),
                "feature_desc": feat.get("feature_desc"),
                "user_story": user_story if _is_nonempty_str(user_story) else "",
                "use_case": use_case if _is_nonempty_str(use_case) else "",
                "feature_specification": feature_spec if _is_nonempty_str(feature_spec) else "",
                "spec_type": "precise" if has_precise else "vague",
                "required_paths": required_paths,
                "optional_paths": optional_paths,
                "gold_lines_map": lines_map,
                "optional_map": optional_map,
            })
    return items


def build_task_text(
    tag: str,
    superfeature: str,
    feature_desc: str,
    user_story: str = "",
    use_case: str = "",
    feature_specification: str = "",
) -> str:
    """
    VAGUE PROMPT (base) + (optionally) the precise fields appended:
    """
    base = (
        "You are a senior software engineer assisting in feature localization. "
        f"Given the repository below, locate where **{tag}** of a feature **{superfeature}**: **{feature_desc}** "
        "needs to take place. Provide the required file_path and the code line number where this change should occur. "
        "There can be multiple locations. Only if the edit can be done anywhere in a given file, set line_number to null "
        "(for example to accomplish the solution the new code has to be appended anywhere in the file). "
        'Answer **only** with a JSON array, each element an object with keys ["file_path","line_number"]. No commentary.\n'
        "Example:\n"
        '[{"file_path":"full_path1/file1.extension","line_number":120},'
        '{"file_path":"full_path2/file2.extension","line_number":null}]\n'
        "Example:\n"
        '[{"file_path":"full_path3/file3.extension","line_number":"15-24"}]\n'
        "First, think step by step: identify likely files or modules, then narrow down to specific functions or lines.\n"
    )

    precise_chunks = []
    if _is_nonempty_str(user_story):
        precise_chunks.append(f"User story: {user_story}")
    if _is_nonempty_str(use_case):
        precise_chunks.append(f"Use case: {use_case}")
    if _is_nonempty_str(feature_specification):
        precise_chunks.append(f"Feature specification: {feature_specification}")
    if precise_chunks:
        base += "\n" + "\n".join(precise_chunks) + "\n"

    base += "Context: ... * Codebase tree structure and files contents *"
    return base


def prompt_template(
    tag: str,
    superfeature: str,
    feature_desc: str,
    code_context: str,
    user_story: str = "",
    use_case: str = "",
    feature_specification: str = "",
) -> List[Dict[str, str]]:
    """Build chat messages. Model must answer with JSON list of {file_path, line_number}."""
    task = build_task_text(tag, superfeature, feature_desc, user_story, use_case, feature_specification)
    user_content = task + "\n\n" + "Codebase context (gptree_output.txt excerpt):\n" + code_context
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
    return resp.choices[0].message.content


def to_csv_safe_list(xs: List[Any]) -> str:
    return ";".join(map(str, xs))


def add_fp_to_dataset(dataset_path: str, dataset_obj: List[Dict[str, Any]],
                      item_idx: Tuple[int, int], new_path: str) -> None:
    """
    Add new OPTIONAL file path with lines='any' to dataset for the specified (group_idx, feature_idx).
    Writes back to dataset_path (overwrites).
    """
    g_idx, f_idx = item_idx
    try:
        group = dataset_obj[g_idx]
        feat = group["features"][f_idx]
        if "file_paths" not in feat or not isinstance(feat["file_paths"], list):
            feat["file_paths"] = []
        feat["file_paths"].append({
            "path": new_path,
            "lines": "any",
            "optional": True,
            "code_preview": []
        })
        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump(dataset_obj, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Failed to update dataset with new path '{new_path}': {e}", file=sys.stderr)


def fmt_line_range(r: Optional[Tuple[int, int]]) -> str:
    if r is None:
        return "any"
    a, b = r
    return f"{a}" if a == b else f"{a}-{b}"


# === Atomic helpers ===========================================================

def gold_buckets_for_item(required_paths: List[str],
                          optional_paths: List[str],
                          gold_lines_map: Dict[str, Optional[Tuple[int, int]]],
                          optional_map: Dict[str, bool]) -> Dict[str, Tuple[Optional[Tuple[int,int]], bool]]:
    """Return { path -> (gold_range_or_None, is_optional_bool) }"""
    buckets = {}
    for p in set(required_paths + optional_paths):
        buckets[p] = (gold_lines_map.get(p, None), bool(optional_map.get(p, False)))
    return buckets


# === New: predicted ranges + range-overlap tolerance wiring ==================

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
        m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", s)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a > b:
                a, b = b, a
            return (a + b) // 2, (a, b)
        m = re.match(r"^\s*(\d+)\s*$", s)
        if m:
            return int(m.group(1)), None
    return None, None


def ranges_overlap_with_tol(pred_range: Tuple[int, int],
                            gold_range: Tuple[int, int],
                            tol: int) -> bool:
    """
    pred_range and gold_range are (start,end) inclusive.
    Overlap if they intersect when gold is expanded by ±tol.
    """
    ps, pe = pred_range
    gs, ge = gold_range
    gs_exp, ge_exp = gs - tol, ge + tol
    return not (pe < gs_exp or ps > ge_exp)


def atomic_hit_status(path: str,
                      pred_point: Optional[int],
                      pred_range: Optional[Tuple[int, int]],
                      buckets: Dict[str, Tuple[Optional[Tuple[int,int]], bool]],
                      tol: int) -> Optional[str]:
    """
    Returns:
      'required' | 'optional' if (path, line/overlap) hits a gold bucket atomically
      None otherwise
    """
    if path not in buckets:
        return None
    gold_range, is_opt = buckets[path]
    if gold_range is None:  # 'any' in gold
        return 'optional' if is_opt else 'required'
    # Prefer true range overlap if we have a predicted range
    if pred_range is not None:
        if ranges_overlap_with_tol(pred_range, gold_range, tol):
            return 'optional' if is_opt else 'required'
        return None
    # Fallback to point check
    if pred_point is None:
        return None
    if within_tolerance(pred_point, gold_range, tol):
        return 'optional' if is_opt else 'required'
    return None


# === HumanEval-style pass@k estimator =======================================

def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """
    Calculates 1 - comb(n - c, k) / comb(n, k) using numerically stable product form.
    """
    n, c = int(num_samples), int(num_correct)
    if n - c < k:
        return 1.0
    arr = np.arange(n - c + 1, n + 1, dtype=float)
    return float(1.0 - np.prod(1.0 - k / arr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Path to dataset JSON.")
    ap.add_argument("--code-context", required=True, help="Path to gptree_output.txt (or similar).")
    ap.add_argument("--model", required=True, help="OpenAI model name, e.g., gpt-4o, gpt-4o-mini, gpt-4.1-mini.")
    ap.add_argument("--outdir", default="runs", help="Directory to write CSV logs.")
    ap.add_argument("--line-tolerance", type=int, default=10, help="± line tolerance outside gold range.")
    ap.add_argument("--max-context-chars", type=int, default=120000, help="Truncate repo context to this many chars.")
    ap.add_argument("--stop-on-fp", action="store_true",
                    help="Interactively adjudicate UNION FPs (modifies dataset on 'yes').")
    ap.add_argument("--num-runs", type=int, default=1, help="Number of independent runs per example (k).")
    ap.add_argument("--use-precise", choices=["vague", "precise"], default="vague",
                    help="Prompt mode: 'vague' ignores precise fields; 'precise' runs only items that have user_story+use_case+feature_specification.")
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
        if not flat_items:
            print("No items with precise fields found. Nothing to evaluate.")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]
            os.makedirs(args.outdir, exist_ok=True)
            csv_path = os.path.join(args.outdir, f"eval_{dataset_name}_{args.model}_{timestamp}.csv")
            # create an empty CSV with headers
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
            print(f"CSV written: {csv_path}")
            return

    os.makedirs(args.outdir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]
    csv_path = os.path.join(args.outdir, f"eval_{dataset_name}_{args.model}_{timestamp}.csv")

    rows: List[Dict[str, Any]] = []

    # Aggregates (macro across examples)
    agg_perrun_precision = 0.0
    agg_perrun_recall = 0.0
    agg_perrun_f1 = 0.0
    agg_perrun_em = 0.0

    agg_union_precision = 0.0
    agg_union_recall = 0.0
    agg_union_f1 = 0.0
    agg_union_em = 0.0

    # macro Pass@{1,3,5,k}
    agg_pass_at_1 = 0.0
    agg_pass_at_3 = 0.0
    agg_pass_at_5 = 0.0
    agg_pass_at_k = 0.0

    # For union scoring, we need to carry both points and ranges across runs
    run_pred_points_list: List[Dict[str, Optional[int]]] = []
    run_pred_ranges_list: List[Dict[str, Optional[Tuple[int, int]]]] = []

    for idx, item in enumerate(flat_items, 1): # DEBUG: limit to first 3 items [:3]
        tag = item["tag"]
        superfeature = item["superfeature"]
        feature_desc = item["feature_desc"]
        user_story = item.get("user_story", "")
        use_case = item.get("use_case", "")
        feature_specification = item.get("feature_specification", "")
        required_paths: List[str] = item["required_paths"]
        optional_paths: List[str] = item["optional_paths"]
        gold_lines_map: Dict[str, Optional[Tuple[int, int]]] = item["gold_lines_map"]
        optional_map: Dict[str, bool] = item["optional_map"]

        # Prompt mode
        include_precise = (args.use_precise == "precise")

        if include_precise:
            short_prompt = build_task_text(tag, superfeature, feature_desc, user_story, use_case, feature_specification)
            messages = prompt_template(tag, superfeature, feature_desc, code_context,
                                       user_story, use_case, feature_specification)
            prompt_spec_mode = "precise"
        else:
            short_prompt = build_task_text(tag, superfeature, feature_desc)
            messages = prompt_template(tag, superfeature, feature_desc, code_context)
            prompt_spec_mode = "vague"

        # Runs
        run_raws: List[str] = []
        run_pred_sets: List[Set[str]] = []
        run_pred_points: List[Dict[str, Optional[int]]] = []
        run_pred_ranges: List[Dict[str, Optional[Tuple[int, int]]]] = []
        perrun_precisions: List[float] = []
        perrun_recalls: List[float] = []
        perrun_f1s: List[float] = []
        perrun_ems: List[int] = []  # strict EM per run (FN=0 and FP=0)

        set_required: Set[str] = set(required_paths)
        set_optional: Set[str] = set(optional_paths)

        for r in range(max(1, args.num_runs)):
            try:
                raw = call_openai(client, args.model, messages)
            except Exception as e:
                print(f"[{idx}/{len(flat_items)}] OpenAI call failed (run {r+1}): {e}", file=sys.stderr)
                raw = ""
            run_raws.append(raw)

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

            set_pred: Set[str] = set(pred_paths)
            run_pred_sets.append(set_pred)
            run_pred_points.append(pred_points_map)
            run_pred_ranges.append(pred_ranges_map)

            # Per-run atomic metrics
            buckets = gold_buckets_for_item(required_paths, optional_paths, gold_lines_map, optional_map)
            tp_req = tp_opt = fp_ = 0
            matched_req: Set[str] = set()

            for p in set_pred:
                status = atomic_hit_status(
                    p,
                    pred_points_map.get(p, None),
                    pred_ranges_map.get(p, None),
                    buckets,
                    args.line_tolerance
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

            perrun_precisions.append(precision)
            perrun_recalls.append(recall)
            perrun_f1s.append(f1)
            # STRICT EM: must have all required and NO FPs
            perrun_ems.append(1 if (fn_req == 0 and fp_ == 0) else 0)

        # Print block (verbose for traceability)
        print(f"\n=== [{idx}/{len(flat_items)}] Evaluating: {tag} — {superfeature} ===")
        print("-" * 80)
        print("Prompt (no context)")
        print("-" * 80)
        print(short_prompt)
        print("-" * 80)
        print("Gold answers from dataset")
        print("-" * 80)
        all_gold = required_paths + optional_paths
        if all_gold:
            for p in sorted(set(all_gold)):
                label_opt = " [opt]" if optional_map.get(p) else ""
                print(f"{p} @ {fmt_line_range(gold_lines_map.get(p))}{label_opt}")
        else:
            print("(none)")
        print("-" * 80)
        print(f"Spec type: {item.get('spec_type')} | Prompt mode used: {prompt_spec_mode}")
        print("-" * 80)
        print("Raw LLM response(s)")
        print("-" * 80)
        if run_raws:
            def _pp(s, max_chars=4000):
                s = (s or "").strip()
                return s if len(s) <= max_chars else s[:max_chars] + "\n...[TRUNCATED IN CONSOLE]..."
            for ri, raw in enumerate(run_raws, 1):
                print(f"[Run {ri}] {_pp(raw)}\n")
        else:
            print("(empty)")
        print("=" * 80 + "\n")

        # Per-run macro (this example)
        ex_perrun_precision = float(np.mean(perrun_precisions)) if perrun_precisions else 0.0
        ex_perrun_recall = float(np.mean(perrun_recalls)) if perrun_recalls else 0.0
        ex_perrun_f1 = float(np.mean(perrun_f1s)) if perrun_f1s else 0.0
        ex_perrun_em = float(np.mean(perrun_ems)) if perrun_ems else 0.0

        agg_perrun_precision += ex_perrun_precision
        agg_perrun_recall += ex_perrun_recall
        agg_perrun_f1 += ex_perrun_f1
        agg_perrun_em += ex_perrun_em

        # UNION metrics (atomic)
        union_pred_paths: Set[str] = set().union(*run_pred_sets) if run_pred_sets else set()

        # Adjudication before scoring
        adjudicated = False
        if args.stop_on_fp and union_pred_paths:
            set_required = set(required_paths); set_optional = set(optional_paths)
            set_gold_all = set_required | set_optional
            union_fps_paths = sorted(list(union_pred_paths - set_gold_all))
            if union_fps_paths:
                print("\n--- Human adjudication required (UNION) ---")
                print(f"Feature: [{tag}] {superfeature} -> {feature_desc}")
                for fp_candidate in list(union_fps_paths):
                    print(f"\nPredicted file not in gold (union): {fp_candidate}")
                    action = input(
                        "Press Enter to add as OPTIONAL(any). Type 'req' to add as REQUIRED. "
                        "Type 'skip' to leave as FP: "
                    ).strip().lower()

                    if action not in {"", "req"}:
                        print("Keeping as FP.")
                        continue

                    ln = input(
                        "Provide gold line range (e.g., '147-153', '147', or press Enter for 'any'): "
                    ).strip().lower()
                    lines_value = "any" if ln in {"", "any", "unknown", "null", "n/a"} else ln

                    try:
                        feat = dataset_obj[item["g_idx"]]["features"][item["f_idx"]]
                        if "file_paths" not in feat or not isinstance(feat["file_paths"], list):
                            feat["file_paths"] = []
                        feat["file_paths"].append({
                            "path": fp_candidate,
                            "lines": lines_value,
                            "optional": (action != "req"),
                            "code_preview": []
                        })
                        with open(args.dataset, "w", encoding="utf-8") as f:
                            json.dump(dataset_obj, f, ensure_ascii=False, indent=2)

                        is_opt = (action != "req")
                        optional_map[fp_candidate] = is_opt
                        gold_lines_map[fp_candidate] = parse_line_range(lines_value)
                        if is_opt:
                            set_optional.add(fp_candidate)
                        else:
                            set_required.add(fp_candidate)
                        set_gold_all.add(fp_candidate)

                        adjudicated = True
                        union_pred_paths.add(fp_candidate)

                        print(f"Added {fp_candidate} as {'OPTIONAL' if is_opt else 'REQUIRED'} "
                              f"with lines='{lines_value}'.")
                    except Exception as e:
                        print(f"Failed to update dataset with '{fp_candidate}': {e}")

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
                        if pr is not None and ranges_overlap_with_tol(pr, gold_range, args.line_tolerance):
                            status = 'optional' if is_opt else 'required'
                            break
                    # If no range hit, try any predicted point within tolerance
                    if status is None:
                        for pt in path_to_points[p]:
                            if pt is not None and within_tolerance(pt, gold_range, args.line_tolerance):
                                status = 'optional' if is_opt else 'required'
                                break
            if status == 'required':
                union_hit_required.add(p)
            elif status == 'optional':
                union_hit_optional.add(p)
            else:
                union_fp += 1

        set_required = set(required_paths)
        union_tp_required = len(union_hit_required)
        union_tp_optional = len(union_hit_optional)
        union_tp_total = union_tp_required + union_tp_optional
        union_fn_required = len(set_required - union_hit_required)

        union_pred_count = len(union_pred_paths)
        union_precision = union_tp_total / union_pred_count if union_pred_count > 0 else 0.0
        union_recall = union_tp_required / len(set_required) if len(set_required) > 0 else 0.0
        union_f1 = (2 * union_precision * union_recall / (union_precision + union_recall)
                    if union_precision + union_recall > 0 else 0.0)
        union_em = 1.0 if union_fn_required == 0 else 0.0

        agg_union_precision += union_precision
        agg_union_recall += union_recall
        agg_union_f1 += union_f1
        agg_union_em += union_em

        # Pass@{1,3,5,k} (strict EM per run)
        n = max(1, args.num_runs)
        c = int(sum(perrun_ems))
        ks = [1, 3, 5, n]
        pass_scores = {}
        for kk in ks:
            kk_eff = min(n, kk)
            pass_scores[f"pass_at_{kk}"] = estimate_pass_at_k(num_samples=n, num_correct=c, k=kk_eff)

        # Aggregate these
        agg_pass_at_1 += pass_scores["pass_at_1"]
        agg_pass_at_3 += pass_scores["pass_at_3"]
        agg_pass_at_5 += pass_scores["pass_at_5"]
        agg_pass_at_k += pass_scores[f"pass_at_{n}"]

        # CSV row
        gold_with_lines = ";".join(
            f"{p}@{fmt_line_range(gold_lines_map.get(p))}{('[opt]' if optional_map.get(p) else '')}"
            for p in sorted((set(required_paths) | set(optional_paths)))
        )
        row = {
            "timestamp": timestamp,
            "dataset": dataset_name,
            "model": args.model,
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
            "gold_with_lines": gold_with_lines,
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
            # Pass@{1,3,5,k}
            "pass_at_1": round(pass_scores["pass_at_1"], 4),
            "pass_at_3": round(pass_scores["pass_at_3"], 4),
            "pass_at_5": round(pass_scores["pass_at_5"], 4),
            "pass_at_k": round(pass_scores[f"pass_at_{n}"], 4),
            # Logs
            "adjudicated": adjudicated,
            "prompt_no_context": short_prompt,
            "raw_llm_response_all_runs": "\n\n---\n\n".join(run_raws),
            # Optional: store the per-run EM vector for transparency/debugging
            "perrun_em_vector": "".join(map(str, perrun_ems)),
        }
        rows.append(row)

        # Short console summary
        print(f"[{idx}/{len(flat_items)}] {tag} | {superfeature} | "
              f"PerRunMacro P:{row['perrun_macro_precision']} R:{row['perrun_macro_recall_required']} "
              f"F1:{row['perrun_macro_f1']} EM:{row['perrun_macro_em']} "
              f"| Union P:{row['union_precision']} R:{row['union_recall_required']} F1:{row['union_f1']} EM:{row['union_em']} "
              f"| Pass@1:{row['pass_at_1']} Pass@3:{row['pass_at_3']} Pass@5:{row['pass_at_5']} Pass@k:{row['pass_at_k']}")

    # Macro across examples
    N = len(rows) if rows else 1
    macro_perrun_precision = agg_perrun_precision / N
    macro_perrun_recall = agg_perrun_recall / N
    macro_perrun_f1 = agg_perrun_f1 / N
    macro_perrun_em = agg_perrun_em / N

    macro_union_precision = agg_union_precision / N
    macro_union_recall = agg_union_recall / N
    macro_union_f1 = agg_union_f1 / N
    macro_union_em = agg_union_em / N

    macro_pass1 = agg_pass_at_1 / N
    macro_pass3 = agg_pass_at_3 / N
    macro_pass5 = agg_pass_at_5 / N
    macro_passk = agg_pass_at_k / N

    print("\n=== Aggregate (macro-averaged across examples) ===")
    print(f"Model: {args.model}")
    print(f"Dataset: {dataset_name}")
    print(f"Features evaluated: {len(flat_items)}")
    print(f"Per-run macro: Precision={macro_perrun_precision:.4f} "
          f"Recall(required)={macro_perrun_recall:.4f} "
          f"F1={macro_perrun_f1:.4f} "
          f"EM={macro_perrun_em:.4f}")
    print(f"Union: Precision={macro_union_precision:.4f} "
          f"Recall(required)={macro_union_recall:.4f} "
          f"F1={macro_union_f1:.4f} "
          f"EM={macro_union_em:.4f}")
    print(f"Pass@1={macro_pass1:.4f}  Pass@3={macro_pass3:.4f}  "
          f"Pass@5={macro_pass5:.4f}  Pass@k={macro_passk:.4f}")

    # Append an AGGREGATE row into the *same* CSV so the global metrics are logged too
    if rows:
        agg_row = {
            "timestamp": timestamp,
            "dataset": dataset_name,
            "model": args.model,
            "superfeature": "__AGGREGATE__",
            "tag": "__SUMMARY__",
            "feature_desc": f"features_evaluated={len(flat_items)}; mode={args.use_precise}",
            "spec_type": "aggregate",
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
        }
        rows.append(agg_row)

    # Write CSV
    if rows:
        if HAS_PANDAS:
            pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8")
        else:
            fieldnames = list(rows[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for r in rows:
                    w.writerow(r)
    else:
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

    print(f"CSV written: {csv_path}")


if __name__ == "__main__":
    main()
