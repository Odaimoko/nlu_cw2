#!/usr/bin/env bash
ROOT="."
## Local variables for current experiment
EXP_ROOT="result/Q5"
DATA_DIR="${ROOT}/europarl_prepared"
TEST_EN_GOLD="${ROOT}/europarl_raw/test.en"
TEST_EN_PRED="${EXP_ROOT}/model_translations.txt"
ckpt_name="checkpoint_best.pt"
mkdir -p ${EXP_ROOT}
# perplexity and loss
python validate.py --cuda True --restore-file $ckpt_name --save-dir $EXP_ROOT --decoder-use-lexical-model True
# translate
python translate.py --checkpoint-path "${EXP_ROOT}/$ckpt_name" --output "${TEST_EN_PRED}" --cuda True
perl multi-bleu.perl -lc ${TEST_EN_GOLD} <${TEST_EN_PRED} | tee "${EXP_ROOT}/bleu.txt"
