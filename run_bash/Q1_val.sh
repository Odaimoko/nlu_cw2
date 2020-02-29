#!/usr/bin/env bash
ROOT=$(git rev-parse --show-toplevel)
## Local variables for current experiment
# EXP_ROOT="checkpoints/Q1-10.77"
EXP_ROOT="result/Q1-10.77"
DATA_DIR="${ROOT}/europarl_prepared"
TEST_EN_GOLD="${ROOT}/europarl_raw/test.en"
TEST_EN_PRED="${EXP_ROOT}/model_translations.txt"
# perplexity and loss
python validate.py --cuda True --restore-file checkpoint_best.pt --save-dir $EXP_ROOT
# translate
python translate.py   --checkpoint-path "${EXP_ROOT}/checkpoint_best.pt"     --output "${TEST_EN_PRED}" --cuda True
perl multi-bleu.perl -lc ${TEST_EN_GOLD} < ${TEST_EN_PRED} | tee "${EXP_ROOT}/bleu.txt"
