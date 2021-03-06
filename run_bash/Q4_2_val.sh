#!/usr/bin/env bash
ROOT=$(git rev-parse --show-toplevel)
RESULTS_ROOT="${ROOT}/results"
## Local variables for current experiment
#EXP_ROOT="checkpoints/Q4-9.96"
EXP_ROOT="result/Q4-9.90"
DATA_DIR="${ROOT}/europarl_prepared"
TEST_EN_GOLD="${ROOT}/europarl_raw/test.en"
TEST_EN_PRED="${EXP_ROOT}/model_translations.txt"
ckpt_name="checkpoint_best.pt"
mkdir -p ${EXP_ROOT}
mkdir -p ${RESULTS_ROOT}
# perplexity and loss
python validate.py --cuda True --restore-file $ckpt_name --save-dir $EXP_ROOT --encoder-num-layers 2 --decoder-num-layers 3 --log-file "${EXP_ROOT}/val.txt"
# translate
python translate.py --checkpoint-path "${EXP_ROOT}/$ckpt_name" --output "${TEST_EN_PRED}" --cuda True
perl multi-bleu.perl -lc ${TEST_EN_GOLD} <${TEST_EN_PRED} | tee "${EXP_ROOT}/bleu.txt"
