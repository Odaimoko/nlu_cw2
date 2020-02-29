#!/usr/bin/env bash
###########
# USAGE NOTE:
# The following script is designed to help you get started running models with the codebase.
# At a minimum, all you need to do is give a name do your experiment using the ${EXP_NAME} variable.
# Additional arguments can be added to the train.py command for different functionality.
# We recommend copying this script and modifying it for each experiment you try (multi-layer, lexical etc)
###########

# Activate Conda Environment [assuming your Miniconda installation is in your root directory]
source ~/miniconda3/bin/activate nlu

# Define a location for all your experiments to save
ROOT=$(git rev-parse --show-toplevel)
RESULTS_ROOT="${ROOT}/result"
mkdir -p ${RESULTS_ROOT}

### NAME YOUR EXPERIMENT HERE ##
EXP_NAME="Q4"
################################

## Local variables for current experiment
EXP_ROOT="${RESULTS_ROOT}/${EXP_NAME}"
DATA_DIR="${ROOT}/europarl_prepared"
TEST_EN_GOLD="${ROOT}/europarl_raw/test.en"
TEST_EN_PRED="${EXP_ROOT}/model_translations.txt"
mkdir -p ${EXP_ROOT}

# Train model. Defaults are used for any argument not specified here. Use "\" to add arguments over multiple lines.
python train.py --save-dir "${EXP_ROOT}" \
                --log-file "${EXP_ROOT}/log.txt"  \
                --data "${DATA_DIR}" \
		--encoder-num-layers 2 \
	       	--decoder-num-layers 3 \
		--cuda True \
#		--decoder-use-lexical-model True \
	#	--arch transformer \
                ### ADDITION AL ARGUMENTS HERE ###
cp "${EXP_ROOT}/log.txt" "${EXP_ROOT}/train_log.out"
## Prediction step
python translate.py \
    --checkpoint-path "${EXP_ROOT}/checkpoint_best.pt" \
    --output "${TEST_EN_PRED}" \
    --cuda True
## Calculate BLEU score for model outputs
perl multi-bleu.perl -lc ${TEST_EN_GOLD} < ${TEST_EN_PRED} | tee "${EXP_ROOT}/bleu.txt"
