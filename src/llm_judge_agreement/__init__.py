"""Measure an LLM judge against human labels.

Give it one row per test case: a case ID, the human label (pass, fail or blank) and the judge's label
(pass or fail). It reports how many real failures the judge caught (TPR), how many good outputs it passed
(TNR), each with a 95% Wilson interval, raw agreement and Cohen's kappa. Rows with a blank human label were
graded by the judge alone, and their pass rate is corrected for the judge's measured error rates
(Rogan-Gladen, with the adjusted interval of Lee et al., arXiv:2511.21140).

Labelled rows must come from a test split that was never used to write the judge's prompt.
Method: https://www.volansaquilae.com/guides/llm-as-a-judge-calibration/
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
from statistics import NormalDist

__version__ = "1.0.0"
__all__ = ["analyse", "format_report", "main", "wilson", "corrected_pass_rate", "demo_rows"]

MIN_TPR, MIN_TNR = 0.80, 0.90   # the default bars; each interval's lower end must reach its bar
Z = NormalDist().inv_cdf(0.975)  # 95% intervals


def wilson(k: int, n: int) -> tuple[float, float]:
    """95% Wilson score interval for k successes out of n."""
    centre = (k + Z * Z / 2) / (n + Z * Z)
    half = Z / (n + Z * Z) * math.sqrt(k * (n - k) / n + Z * Z / 4)
    return max(0.0, centre - half), min(1.0, centre + half)


def _clip(x: float) -> float:
    return min(1.0, max(0.0, x))


def corrected_pass_rate(passed: int, n: int, tp: int, n_fail: int, tn: int, n_pass: int) -> tuple[float, float, float]:
    """Rogan-Gladen pass rate, with the adjusted 95% interval of Lee et al. (arXiv:2511.21140).

    Returns (estimate, low, high). The estimate is unclipped, so a value outside 0 to 1 shows the error
    rates don't fit the judge-only rows."""
    tpr, tnr = tp / n_fail, tn / n_pass
    estimate = (passed / n + tpr - 1) / (tpr + tnr - 1)
    p = (passed + Z * Z / 2) / (n + Z * Z)  # their eq. 7: pseudo-counts on all three rates
    q0, q1 = (tp + 1) / (n_fail + 2), (tn + 1) / (n_pass + 2)
    if q0 + q1 <= 1:  # too few labels to bound it
        return estimate, 0.0, 1.0
    n, m0, m1 = n + Z * Z, n_fail + 2, n_pass + 2
    theta = (p + q0 - 1) / (q0 + q1 - 1)
    shift = 2 * Z * Z * (-(1 - theta) * q0 * (1 - q0) / m0 + theta * q1 * (1 - q1) / m1)  # eq. 8
    se = math.sqrt(p * (1 - p) / n + (1 - theta) ** 2 * q0 * (1 - q0) / m0
                   + theta ** 2 * q1 * (1 - q1) / m1) / (q0 + q1 - 1)  # eq. 6
    return estimate, _clip(theta + shift - Z * se), _clip(theta + shift + Z * se)


def analyse(rows, min_tpr: float = MIN_TPR, min_tnr: float = MIN_TNR) -> dict:
    """Score the judge. rows: dicts with case_id, human_label and judge_label. Raises ValueError on bad input."""
    pairs, judge_only, seen = [], [], set()
    for number, row in enumerate(rows, start=1):
        case, human, judge = ((row.get(key) or "").strip() for key in ("case_id", "human_label", "judge_label"))
        human, judge = human.lower(), judge.lower()
        if not case or case in seen or judge not in ("pass", "fail") or human not in ("pass", "fail", ""):
            raise ValueError(f"row {number}: need a new case_id, a judge_label of pass or fail "
                             "and a human_label of pass, fail or blank")
        seen.add(case)
        (pairs.append((human, judge)) if human else judge_only.append(judge))
    tp, fn = pairs.count(("fail", "fail")), pairs.count(("fail", "pass"))
    fp, tn = pairs.count(("pass", "fail")), pairs.count(("pass", "pass"))
    n_fail, n_pass, n = tp + fn, fp + tn, len(pairs)
    if not (n_fail and n_pass):
        raise ValueError("the labelled rows need at least one human fail and one human pass")
    p_o = (tp + tn) / n
    p_e = (n_fail * (tp + fp) + n_pass * (fn + tn)) / n ** 2
    result = {
        "labelled": n, "human_fail": n_fail, "human_pass": n_pass,
        "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
        "tpr": {"k": tp, "n": n_fail, "rate": tp / n_fail, "interval": wilson(tp, n_fail), "bar": min_tpr},
        "tnr": {"k": tn, "n": n_pass, "rate": tn / n_pass, "interval": wilson(tn, n_pass), "bar": min_tnr},
        "agreement": p_o,
        "kappa": (p_o - p_e) / (1 - p_e) if p_e < 1 else None,
        "judge_only": None,
    }
    for key in ("tpr", "tnr"):
        result[key]["met"] = result[key]["interval"][0] >= result[key]["bar"]
    if judge_only:
        if tp / n_fail + tn / n_pass <= 1:
            raise ValueError("TPR + TNR is 1 or less: the judge is no better than chance")
        passed, m = judge_only.count("pass"), len(judge_only)
        estimate, low, high = corrected_pass_rate(passed, m, tp, n_fail, tn, n_pass)
        result["judge_only"] = {"rows": m, "passed": passed, "raw_rate": passed / m, "raw_interval": wilson(passed, m),
                                "corrected": _clip(estimate), "corrected_interval": (low, high),
                                "clipped": not 0 <= estimate <= 1}
    result["ok"] = result["tpr"]["met"] and result["tnr"]["met"]
    return result


def _rate(part: dict) -> str:
    low, high = part["interval"]
    return f"{part['k']}/{part['n']} = {part['rate']:.1%} ({100 * low:.1f}-{100 * high:.1f})"


def format_report(r: dict) -> str:
    """The plain-text report the command prints."""
    c = r["confusion"]
    out = io.StringIO()
    line = lambda label, text: out.write(label.ljust(26) + text + "\n")  # noqa: E731
    out.write(f"labelled rows: {r['labelled']} (human labels: {r['human_fail']} fail, {r['human_pass']} pass)\n")
    out.write(f"{'':12}{'judge fail':>12}{'judge pass':>12}\n")
    out.write(f"{'human fail':12}{c['tp']:>12}{c['fn']:>12}\n")
    out.write(f"{'human pass':12}{c['fp']:>12}{c['tn']:>12}\n")
    for label, key in (("TPR, failures caught:", "tpr"), ("TNR, good outputs passed:", "tnr")):
        part = r[key]
        line(label, f"{_rate(part)}  bar {part['bar']:.0%} at the lower end: {'met' if part['met'] else 'missed'}")
    line("raw agreement:", f"{c['tp'] + c['tn']}/{r['labelled']} = {r['agreement']:.1%}")
    line("Cohen's kappa:", "undefined" if r["kappa"] is None else f"{r['kappa']:.3f}")
    j = r["judge_only"]
    if j:
        out.write(f"judge-only rows: {j['rows']}\n")
        low, high = j["raw_interval"]
        line("judge's raw pass rate:", f"{j['passed']}/{j['rows']} = {j['raw_rate']:.1%} ({100 * low:.1f}-{100 * high:.1f})")
        low, high = j["corrected_interval"]
        line("corrected pass rate:", f"{j['corrected']:.1%} ({100 * low:.1f}-{100 * high:.1f})")
        if j["clipped"]:
            out.write("warning: estimate clipped; these error rates don't fit the judge-only rows\n")
    out.write("decision: " + ("the judge meets both bars" if r["ok"]
                              else "the judge missed a bar; don't let it gate a release yet") + "\n")
    return out.getvalue()


def demo_rows() -> list[dict]:
    """Made-up rows for one failure class: 120 test-split labels, 400 judge-only verdicts."""
    counts = {("fail", "fail"): 30, ("fail", "pass"): 10, ("pass", "fail"): 2,
              ("pass", "pass"): 78, ("", "fail"): 80, ("", "pass"): 320}
    labels = [pair for pair, count in counts.items() for _ in range(count)]
    return [{"case_id": f"tone-{i:03d}", "human_label": human, "judge_label": judge}
            for i, (human, judge) in enumerate(labels, start=1)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="llm-judge-agreement",
        description="Measure an LLM judge against human labels. The CSV needs the columns case_id, "
                    "human_label and judge_label; each label is pass or fail, and a blank human label "
                    "marks a row the judge graded alone.",
        epilog="Exit status: 0 when the judge meets both bars, 1 when it misses one, 2 when the input "
               "can't be used. Method: https://www.volansaquilae.com/guides/llm-as-a-judge-calibration/")
    ap.add_argument("labels", nargs="?", help="CSV file of labels, or - to read standard input")
    ap.add_argument("--demo", action="store_true", help="run on made-up rows to see the report")
    ap.add_argument("--min-tpr", type=float, default=MIN_TPR, help="bar for the TPR interval's lower end (default 0.80)")
    ap.add_argument("--min-tnr", type=float, default=MIN_TNR, help="bar for the TNR interval's lower end (default 0.90)")
    ap.add_argument("--json", action="store_true", help="print the result as JSON, for CI")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = ap.parse_args(argv)
    if args.demo:
        rows = demo_rows()
    elif args.labels == "-":
        rows = list(csv.DictReader(sys.stdin))
    elif args.labels:
        try:
            with open(args.labels, newline="", encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
        except OSError as e:
            print(f"llm-judge-agreement: {e}", file=sys.stderr)
            return 2
    else:
        ap.print_usage(sys.stderr)
        print("llm-judge-agreement: give a CSV file, - for standard input, or --demo", file=sys.stderr)
        return 2
    try:
        result = analyse(rows, args.min_tpr, args.min_tnr)
    except ValueError as e:
        print(f"llm-judge-agreement: {e}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2) if args.json else format_report(result), end="" if not args.json else "\n")
    return 0 if result["ok"] else 1
