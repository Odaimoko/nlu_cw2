#!/usr/bin/env bash
# Define a location for all your experiments to save
. /opt/anaconda3/etc/profile.d/conda.sh 
conda activate nlu
ROOT="."
RESULTS_ROOT="${ROOT}/oda_exp"

### NAME YOUR EXPERIMENT HERE ##
EXP_NAME="q7"
################################

## Local variables for current experiment
EXP_ROOT="${RESULTS_ROOT}/${EXP_NAME}"
DATA_DIR="${ROOT}/europarl_prepared"
TEST_EN_GOLD="${ROOT}/europarl_raw/test.en"
TEST_EN_PRED="${EXP_ROOT}/model_translation.txt"
mkdir -p ${EXP_ROOT}
mkdir -p ${RESULTS_ROOT}

# Train model. Defaults are used for any argument not specified here. Use "\" to add arguments over multiple lines.
python train.py --save-dir "${EXP_ROOT}" \
                --log-file "${EXP_ROOT}/train_log.txt"  \
                --data "${DATA_DIR}" \
                --max-epoch 150 \
                 --cuda True \
                --arch \
                transformer  --restore-file ".pt"

                ### ADDITIONAL ARGUMENTS HERE ###

## Prediction step
python translate.py     --checkpoint-path "${EXP_ROOT}/checkpoint_best.pt"     --output "${TEST_EN_PRED}" --cuda True

## Calculate BLEU score for model outputs
perl multi-bleu.perl -lc ${TEST_EN_GOLD} < ${TEST_EN_PRED} | tee "${EXP_ROOT}/bleu.txt"
