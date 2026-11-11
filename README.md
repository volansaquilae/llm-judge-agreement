# llm-judge-agreement

Check an LLM judge against human labels before you let it gate a release.

You give it one row per test case with your label and the judge's label. It tells you how many real failures the judge caught (TPR) and how many good outputs it passed (TNR), each with a 95% interval, plus raw agreement and Cohen's kappa. For rows the judge graded on its own, it corrects the judge's pass rate for the error rates it just measured.

No dependencies. Python 3.9 or later.

```
pip install llm-judge-agreement
llm-judge-agreement --demo
```

## Use it

Make a CSV with three columns:

```
case_id,human_label,judge_label
refund-001,fail,fail
refund-002,pass,pass
refund-003,,pass
```

Each label is `pass` or `fail`. Leave `human_label` blank for rows the judge graded alone. Labelled rows must come from a test split you never used to write the judge's prompt, or the numbers flatter the judge.

```
llm-judge-agreement labels.csv
```

```
labelled rows: 120 (human labels: 40 fail, 80 pass)
              judge fail  judge pass
human fail            30          10
human pass             2          78
TPR, failures caught:     30/40 = 75.0% (59.8-85.8)  bar 80% at the lower end: missed
TNR, good outputs passed: 78/80 = 97.5% (91.3-99.3)  bar 90% at the lower end: met
raw agreement:            108/120 = 90.0%
Cohen's kappa:            0.763
judge-only rows: 400
judge's raw pass rate:    320/400 = 80.0% (75.8-83.6)
corrected pass rate:      75.9% (67.3-84.1)
decision: the judge missed a bar; don't let it gate a release yet
```

A bar is met when the lower end of the interval reaches it, so a judge can't pass on a lucky handful of labels. The defaults are 80% for TPR and 90% for TNR. Agree your own before the run and set them with `--min-tpr` and `--min-tnr`.

Options:

- `--json` prints the result as JSON, for CI.
- `-` reads the CSV from standard input.
- Exit status is 0 when the judge meets both bars, 1 when it misses one and 2 when the input can't be used, so a CI step fails on its own.

From Python:

```python
from llm_judge_agreement import analyse, format_report

result = analyse(rows, min_tpr=0.8, min_tnr=0.9)  # rows: dicts with case_id, human_label, judge_label
print(format_report(result))
```

## The method

The intervals are Wilson score intervals. The corrected pass rate is the Rogan-Gladen estimator, with the adjusted interval from Lee et al., [arXiv:2511.21140](https://arxiv.org/abs/2511.21140). Raw agreement and kappa are reported, but the decision rests on TPR and TNR, because a judge can agree 90% of the time and still miss a quarter of real failures, as the example shows.

The full method, including how many cases to label, how to split them and what to do when a judge misses its bar, is in [How to calibrate an LLM as a judge for agent evals](https://www.volansaquilae.com/guides/llm-as-a-judge-calibration/), a guide by [Volans Aquilae](https://www.volansaquilae.com/).

## Licence

MIT.
