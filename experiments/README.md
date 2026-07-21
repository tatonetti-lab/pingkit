# Reproducing the manuscript

This directory contains the analysis scripts behind the figures and tables in
*Probing Hidden States for Calibrated, Alignment-Resistant Predictions in LLMs*.

Scripts are grouped by experiment:

- `mmlu/` — MMLU accuracy/calibration, layer sweeps, component ablation, subject probes (Tables 1, 3, 4; Figures 2, 4).
- `fine_tune_experiments/` — DPO+LoRA refusal analysis on MedMCQA (Figure 3).
- `simpleqa_experiments/` — SimpleQA correctness probe and BoolQ transfer (Table 2).
- `general_use/` — reusable drivers (activation extraction, layer sweep, multi-layer training).
- `results/` — a few small summary files so Figure 2 and Table 1 can be redrawn directly (see below).

## What you need to reproduce a result

Every result comes from the same three-stage pipeline:

1. **Generate** any model outputs the experiment needs (multiple-choice predictions,
   free-form answers, DPO fine-tuning). These use the inference/fine-tuning scripts
   and the open-weight models listed below.
2. **Extract activations** from the frozen model with `general_use/embed.py`, which
   writes a pingkit embedding directory (pooled residual/attention/MLP features per
   layer). This is the "embeddings" step referenced throughout.
3. **Train and evaluate** a probe on those embeddings with the per-experiment script
   (layer sweep, data-size sweep, multi-layer, subject, ablation), then plot.

The raw activations and generations are multi-GB and are **not** shipped. Given the
embeddings from step 2, the scripts below regenerate each figure and table without
any further model inference. Figure 2 and Table 1 also ship their small summary
files under `results/`, so those two can be redrawn with no GPU at all.

Open-weight models used: `google/gemma-2-2b-it`, `google/gemma-2-9b-it`,
`meta-llama/Llama-3.1-8B`, `meta-llama/Llama-3.1-8B-Instruct`,
`meta-llama/Llama-3.3-70B-Instruct`.

## Figure and table index

| Manuscript item | What it shows | Generating script(s) | How to regenerate |
|---|---|---|---|
| **Table 1** | MMLU accuracy / ECE / Brier, generative vs. PING, with temperature & isotonic baselines | Generative: `mmlu/inf_mmlu.py`, `mmlu/llm_inference_gemma.py`, `mmlu/llm_inference_llama.py`. PING: `mmlu/train.py`, `mmlu/train_eval.py`. Baselines: `mmlu/calibration.py`, `mmlu/calib_iso_ts.py`, `mmlu/calib_iso_ts_drop.py` | Shipped: `results/table1_mmlu/*_bootstrap_stats*.csv` hold the per-model numbers. To rebuild: run generative inference, embed, then `train_eval.py` + the calibration scripts |
| **Table 1 (CoT row)** | Gemma-2-9B-it with chain-of-thought | `mmlu/add_reasoning.py`, `mmlu/llm_inference_CoT_train.py` | Shipped: `results/table1_mmlu/mmlu_gemma_9b_test_CoT_bootstrap_stats*.csv` |
| **Fig. 1** | Framework schematic | — (illustration, no data) | — |
| **Fig. 2A** | Layer-wise probe accuracy per model | Sweep: `general_use/layer_eval.py`. Plot: `results/fig2a_layers/plot.py` | Shipped: redraw from `results/fig2a_layers/<model>.json` (see command below). From scratch: embed, then `layer_eval.py` per model |
| **Fig. 2B** | Probe accuracy vs. training-set size | Sweep: `mmlu/train_eval.py`. Plot: `results/fig2b_dataeff/plot.py` | Shipped: redraw from `results/fig2b_dataeff/train_size_<model>.json` (see command below). From scratch: embed, then `train_eval.py` sweeping the train size |
| **Fig. 3A** | Fine-tuning methodology flowchart | — (illustration, no data) | — |
| **Fig. 3B** | DPO outcomes (refused/correct/incorrect), stratified by base correctness | `fine_tune_experiments/train.py` (DPO+LoRA), `fine_tune_experiments/eval.py`, `fine_tune_experiments/correct_refusal.py`, `fine_tune_experiments/incorrect_refusal.py` | DPO fine-tune, run `eval.py` over MedMCQA, then the refusal-tally scripts |
| **Fig. 3C** | Layer-wise probe accuracy on the refused subset | `fine_tune_experiments/embed.py`, `fine_tune_experiments/split.py`, `fine_tune_experiments/subset_correct.py`, `fine_tune_experiments/subset_incorrect.py`, `fine_tune_experiments/train_ping.py`, `fine_tune_experiments/knowledge_lost.py`, `fine_tune_experiments/knowledge_neverhad.py`. Sweep: `general_use/layer_eval.py` | Embed the refused subset, then run `general_use/layer_eval.py` on those embeddings |
| **Fig. 4** | Component ablation (residual / attention / MLP), layer-wise | Sweep: `mmlu/layer_eval_ab.py`. Plot: `mmlu/plot_ab.py` | Embed with all three component types retained, run `layer_eval_ab.py`, then `plot_ab.py --json_file <sweep>.json` |
| **Table 2** | SimpleQA correctness probe + BoolQ transfer | SimpleQA: `simpleqa_experiments/llm_gen_answers.py`, `simpleqa_experiments/check_acc.py`, `simpleqa_experiments/run.py`. BoolQ: `simpleqa_experiments/llm_gen_answers_boolq.py`, `simpleqa_experiments/check_acc_bool.py`, `simpleqa_experiments/split_boolq.py`, `simpleqa_experiments/bool_eval.py` | Generate answers, score with the GPT judge, embed the last generated token, then `bool_eval.py` for the probe |
| **Table 3** | MMLU subject classification (57 subjects) | `mmlu/train_subject.py` | Embed MMLU, then `train_subject.py` with the subject label column |
| **Table 4** | Single-layer vs. multi-layer probes | `mmlu/train_multilayer.py`, `general_use/train_mult.py` | Embed with multiple layers retained, then `train_multilayer.py` / `train_mult.py` |

## Model naming in result filenames

| Filename token | Model |
|---|---|
| `gemma2` | Gemma-2-2B-it |
| `gemma` | Gemma-2-9B-it |
| `gemma_9b_..._CoT` | Gemma-2-9B-it with chain-of-thought |
| `llama8` | Llama-3.1-8B (base) |
| `llama8I` | Llama-3.1-8B-Instruct |
| `llama70` | Llama-3.3-70B-Instruct |

## Redrawing the two shipped figures

```bash
# Fig. 2A — layer-wise accuracy
python experiments/results/fig2a_layers/plot.py \
    --dir experiments/results/fig2a_layers --glob '*.json'

# Fig. 2B — data efficiency
python experiments/results/fig2b_dataeff/plot.py \
    --dir experiments/results/fig2b_dataeff --glob 'train_size_*.json'
```

The `*_bootstrap_stats*.csv` files under `results/table1_mmlu/` contain the
per-model calibration-comparison numbers (generative vs. temperature vs.
isotonic) reported in Table 1.
