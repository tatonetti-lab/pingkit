#!/usr/bin/env bash
# Reproduce the SimpleQA in-domain correctness probe (Table 2, first row).
#
# Pipeline:
#   1. llm_gen_answers.py  -> Gemma-2-9b-it answers each SimpleQA question
#   2. check_acc.py        -> GPT judge scores answers -> simpleqa_scored.csv
#   3. split_simpleqa.py   -> label correct/incorrect -> simpleqa_labeled.csv
#   4. ../general_use/embed.py -> residual-stream activation at the last generated
#                                 token, layer 39, written to $EMB
#   5. this script         -> train the probe (300 incorrect / 200 correct) and
#                             evaluate on the imbalanced held-out remainder.
#
# The training draw is 300 for class 0 (incorrect) and 200 for class 1 (correct);
# everything else is held out, so the evaluation set keeps SimpleQA's native
# imbalance (majority-class baseline ~98.4%). AUROC is the metric of record.
#
# The trained probe and its metrics ship under experiments/results/table2/, so the
# reported numbers can be verified without rerunning any model inference.
set -euo pipefail

EMB=${1:-./simpleqa_embeddings}   # embedding dir from general_use/embed.py

python split_simpleqa.py

python train_correctness.py \
    --data_dirs         "$EMB" \
    --labels_csvs       simpleqa_labeled.csv \
    --label_col         correct \
    --id_col            id \
    --parts             rs \
    --layers            39 \
    --model_type        mlp \
    --restack           true \
    --train_class_sizes 300 200 \
    --output            sqa_model
