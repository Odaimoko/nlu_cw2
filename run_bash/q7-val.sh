#!/usr/bin/env bash
ROOT="."
EXP_ROOT="oda_exp/q7-nodropout"
DATA_DIR="${ROOT}/europarl_prepared"
TEST_EN_GOLD="${ROOT}/europarl_raw/test.en"
TEST_EN_PRED="${EXP_ROOT}/model_translation.txt"
ckpt_name="checkpoint_best.pt"
mkdir -p ${EXP_ROOT}
# perplexity and loss
python validate.py --cuda True --restore-file $ckpt_name --save-dir $EXP_ROOT --arch \
                transformer --log-file "${EXP_ROOT}/val_log.txt"
# translate
python translate.py --checkpoint-path "${EXP_ROOT}/$ckpt_name" --output "${TEST_EN_PRED}" --cuda True
perl multi-bleu.perl -lc ${TEST_EN_GOLD} <${TEST_EN_PRED} | tee "${EXP_ROOT}/bleu.txt"
