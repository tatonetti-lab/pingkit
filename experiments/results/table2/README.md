# Table 2 — correctness probe artifacts

Trained probes and evaluation logs for the SimpleQA correctness probe and its
BoolQ cross-task transfer (manuscript Table 2). These let the reported numbers be
verified without rerunning any model inference. Model: Gemma-2-9b-it (frozen),
residual stream, last-token pooling, layer 39.

Probes follow the `pingkit` prefix convention (`<name>.pt` weights + `<name>.json`
meta) and load with `pingkit.model.load_artifacts("<name>")`.

## Files by Table 2 row

| Row (Trained → Tested) | Reported | Artifact / log |
|---|---|---|
| SimpleQA → SimpleQA (in-domain) | AUROC 0.753, acc 81.6%, ECE 5.4%, Brier 0.132 | `simpleqa_probe.{pt,json}`, `simpleqa_metrics.json`, `simpleqa_holdout_predictions.csv` |
| SimpleQA → BoolQ (cross-task transfer) | AUROC 0.627, acc 53.3% | probe = `simpleqa_probe`; eval log `simpleqa_to_boolq_eval.txt` (via `../../simpleqa_experiments/bool_eval.py`) |
| BoolQ → BoolQ (reference) | AUROC 0.678, acc 61.9% | `boolq_probe.{pt,json}`; eval log `boolq_to_boolq_eval.txt` |

`simpleqa_metrics.json` is the authoritative in-domain result. Its held-out set is
**not** class-balanced: 3,746 examples (59 correct / 3,687 incorrect; 1.6% correct;
majority-class baseline 98.4%), evaluated at a 0.5 decision threshold. AUROC and
balanced (macro) accuracy are the imbalance-robust metrics; raw accuracy is reported
only alongside its baseline. `simpleqa_holdout_predictions.csv` has the per-example
probabilities behind those numbers.

## Regenerating from scratch

See `../../simpleqa_experiments/run_simpleqa_eval.sh` (SimpleQA in-domain) and
`../../simpleqa_experiments/bool_eval.py` (BoolQ transfer / reference).
