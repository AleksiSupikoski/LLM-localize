"""
Recalculate per-run metrics (Precision, Recall, F1, EM)
for multiple line tolerances including file-only (∞).
Aggregates results per model across all given CSVs.

Usage:
  python recalculate_perrun_tolerances.py
"""

import re, json, pandas as pd, numpy as np
from typing import Optional, Tuple, Dict, Any, List

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODELS = {
    "gpt-5": [
        "PRECISE_eval_oskardropship-dataset-30Sep_gpt-5_20251029_170159.csv",
        "VAGUE_eval_particleclicker-dataset-30Sep_gpt-5_20251028_130917.csv",
        "PRECISE_eval_particleclicker-dataset-30Sep_gpt-5_20251027_174443.csv",
        "eval_oskardropship-dataset-30Sep_gpt-5_20251031_162023.csv",
    ],
    "gpt-5-mini": [
        "PRECISE_eval_oskardropship-dataset-30Sep_gpt-5-mini_20251002_143840.csv",
        "VAGUE_eval_oskardropship-dataset-30Sep_gpt-5-mini_20251002_160300.csv",
        "PRECISE_eval_particleclicker_dataset_30Sep_gpt_5_mini_20251001_230339.csv",
        "VAGUE_eval_particleclicker_dataset_30Sep_gpt_5_mini_20251001_175727.csv",
    ]
}

TOLERANCES = [3, 10, 15, float("inf")]  # ∞ = file-only


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_line_range(s: Optional[str]) -> Optional[Tuple[int, int]]:
    if not s or not isinstance(s, str):
        return None
    s = s.strip().lower().replace("any", "")
    if s in {"", "unknown", "null", "none", "n/a"}:
        return None
    m = re.match(r"^(\d+)\s*-\s*(\d+)$", s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return (min(a, b), max(a, b))
    if s.isdigit():
        v = int(s)
        return (v, v)
    return None


def within_tolerance(pred: Optional[int], gold_range: Optional[Tuple[int, int]], tol: float) -> bool:
    if tol == float("inf"):  # file-only
        return True
    if gold_range is None:
        return True
    if pred is None:
        return False
    a, b = gold_range
    return a - tol <= pred <= b + tol


def ranges_overlap_with_tol(r1, r2, tol: float) -> bool:
    if tol == float("inf"):
        return True
    if r1 is None or r2 is None:
        return False
    a1, b1 = r1
    a2, b2 = r2
    return not (b1 + tol < a2 - tol or b2 + tol < a1 - tol)


def extract_json_list(text: str) -> List[Dict[str, Any]]:
    if not isinstance(text, str):
        return []
    text = text.strip()
    if not text:
        return []
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass
    first, last = text.find("["), text.rfind("]")
    if first != -1 and last > first:
        try:
            obj = json.loads(text[first:last + 1])
            if isinstance(obj, list):
                return obj
        except Exception:
            pass
    return []


def parse_pred_line_or_range(value):
    if value is None:
        return None, None
    if isinstance(value, (int, float)):
        return int(value), None
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"", "null", "none", "any"}:
            return None, None
        m = re.match(r"^(\d+)\s*-\s*(\d+)$", s)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            return (a + b) // 2, (min(a, b), max(a, b))
        if s.isdigit():
            return int(s), None
    return None, None


# ---------------------------------------------------------------------------
# Re-evaluate per-run metrics
# ---------------------------------------------------------------------------

def evaluate_perrun_row(row: Dict[str, Any], tol: float) -> Dict[str, float]:
    gold_req = [p for p in str(row.get("gold_required_paths", "")).split(";") if p]
    gold_opt = [p for p in str(row.get("gold_optional_paths", "")).split(";") if p]
    gold_with_lines = str(row.get("gold_with_lines", ""))

    # Map each gold path -> (line_range, is_optional)
    lines_map, opt_map = {}, {}
    for entry in gold_with_lines.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        m = re.match(r"^(.*?)@(.+)$", entry)
        if not m:
            continue
        path = m.group(1)
        rest = m.group(2)
        is_opt = "[opt]" in rest
        rest = rest.replace("[opt]", "").strip()
        lines_map[path] = parse_line_range(rest)
        opt_map[path] = is_opt

    required = set(gold_req)
    optional = set(gold_opt)

    runs_text = str(row.get("raw_llm_response_all_runs", "") or "")
    runs = runs_text.split("\n---\n")

    perrun_metrics = []
    for run_text in runs:
        preds = extract_json_list(run_text)
        pred_set, pred_points, pred_ranges = set(), {}, {}
        for obj in preds:
            fp = obj.get("file_path")
            if not isinstance(fp, str) or not fp.strip():
                continue
            path = fp.strip()
            pred_set.add(path)
            p, r = parse_pred_line_or_range(obj.get("line_number"))
            pred_points[path] = p
            pred_ranges[path] = r

        tp_req = tp_opt = fp = 0
        matched_req = set()
        for p in pred_set:
            if p in required or p in optional:
                gold_range = lines_map.get(p)
                # always true for file-only
                if tol == float("inf") or within_tolerance(pred_points.get(p), gold_range, tol):
                    if p in optional:
                        tp_opt += 1
                    else:
                        tp_req += 1
                        matched_req.add(p)
                else:
                    fp += 1
            else:
                fp += 1
        fn = len(required - matched_req)
        total_pred = len(pred_set)
        precision = (tp_req + tp_opt) / total_pred if total_pred > 0 else 0.0
        recall = tp_req / len(required) if len(required) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        em = 1.0 if (fp == 0 and fn == 0) else 0.0
        perrun_metrics.append((precision, recall, f1, em))

    if not perrun_metrics:
        return {"precision": 0, "recall": 0, "f1": 0, "em": 0}

    arr = np.array(perrun_metrics)
    return dict(zip(["precision", "recall", "f1", "em"], np.mean(arr, axis=0)))


def evaluate_model_group(model_name: str, csv_paths: List[str]) -> pd.DataFrame:
    rows = []
    for tol in TOLERANCES:
        tol_name = "∞" if tol == float("inf") else f"±{int(tol)}"
        all_metrics = []
        for path in csv_paths:
            df = pd.read_csv(path)
            if "prompt_no_context" in df.columns:
                df = df[df["prompt_no_context"] != "AGGREGATE ROW"]
            for _, r in df.iterrows():
                all_metrics.append(evaluate_perrun_row(r, tol))
        arr = pd.DataFrame(all_metrics).mean()
        rows.append({
            "Model": model_name,
            "Tolerance": tol_name,
            "Precision": round(arr["precision"], 3),
            "Recall": round(arr["recall"], 3),
            "F1": round(arr["f1"], 3),
            "EM": round(arr["em"], 3),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    all_results = []
    for model, paths in MODELS.items():
        print(f"Evaluating {model} ...")
        all_results.append(evaluate_model_group(model, paths))

    summary = pd.concat(all_results, ignore_index=True)
    print("\n=== Per-run Metrics by Model and Tolerance ===\n")
    print(summary.to_string(index=False))

    # Optionally save
    summary.to_csv("summary_perrun_tolerances.csv", index=False)
    print("\nSaved: summary_perrun_tolerances.csv")
