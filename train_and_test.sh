#!/bin/bash

# DO NOT FORGET TO EXPORT THESE VARIABLES OR EXPORT BELOW:
#TOOLS="${HOME}/jellies/tools"
#export YOLO_MODEL_ZOO="${TOOLS}/ultralytics/models"
#export DT2_MODEL_ZOO="${TOOLS}/detectron2/configs"
#export DT2_FASTER_RCNN_WEIGHTS="${TOOLS}/detectron2_models/faster_rcnn_R_50_FPN_3x/model_final_280758.pkl"

DATA_DIR="../data"
RESULTS_DIR="../results"
echo "Using ${DATA_DIR}"

##
# ARGUMENTS #
##

# MANDATORY
SEEDS=${1} # 0 or 1 or empty (then just a test run with stdout)
DEVICE="${2}" # cuda:0
EPOCHS="${3}" # 100
BATCH="${4}" # ?
MODELS="${5}" # "yolo8" or "dt2"
GRPS="${6}" # "Groups" or "noG"
DATASETS="${7}" # "CAP ARCR"
PPS="${8}" # "MSRCR MSRCR_with_gamma_up"

FOLDS="0 1 2"
QUICKY="" # for debugging: "--quick_debug"
RESUME="" # resume training - experimental - do not use: "--resume"

if [[ -z ${DEVICE} ]]; then
  DEVICE="--device cuda:0"
else
  DEVICE="--device ${DEVICE}"
fi
if [[ -z ${EPOCHS} ]]; then
  EPOCHS=100
fi

if [[ -z ${BATCH} ]]; then
  BATCH=-1
fi

if [[ -z ${MODELS} ]]; then
  MODELS="dt2 yolo8"
fi

if [[ -z ${GRPS} || ${GRPS} == "NONE" ]]; then
  GRPS="Groups NoGroups"
fi

if [[ -z ${DATASETS} ]]; then
  echo "Specify DATASETS"
  exit 1
fi

if [[ -z ${PPS} ]]; then
  echo "Specify PPS"
  exit 1
fi

##
# ITERATE THROUGH ALL JOBS #
##
for PP in ${PPS}
do
  for MODEL in ${MODELS}
  do
    if [[ ${MODEL} == 'yolo8' || ${MODEL} == 'dt2' ]]; then
      ENV="${MODEL}"
    else
      echo "Unknown model ${MODEL}"
      exit 1
    fi
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate ${ENV}

    ITERS_OR_EPOCHS="--epochs ${EPOCHS}"
    if [[ -n ${QUICKY} ]]; then
      if [[ ${MODEL} == 'dt2' ]]; then ITERS_OR_EPOCHS="--max_iter 1"; else ITERS_OR_EPOCHS="--epochs 1"; fi
    fi
    ##
    for GRP in ${GRPS}
    do
      if [[ -z ${GRP} || ${GRP} == "noG" || ${GRP} == "noGroups" || ${GRP} == "NoG" || ${GRP} == "NoGroups" ]]; then
        GRPS_ARG="--multi_label"
      else
        GRPS_ARG="--filename_groups"
      fi

      if [[ ${PP} == "NONE" ]]; then
        IMG_DIR="images"
        SUFFIX="${EXTRA_NAME}"
      else
        IMG_DIR="${PP}"
        SUFFIX="${EXTRA_NAME}.${PP}"
      fi

      ##
      for DD in ${DATASETS}
      do

        ###########
        # WARNING #
        # This was done to allow preprocessing with MSRCR on img size 1333 x 800 pixels:
        # if [[ "${PP}" == *"resized"* ]]; then
        #   ANNOT_FN="${DD}.resized.json"
        # else
        #   ANNOT_FN="${DD}.original.json"
        # fi
        #
        # For simplicity, you can skip this detail, but in practice, input resolution must be considered
        ANNOT_FN="${DD}.json"
        ###########

        if [[ ${SEEDS} == "DEFAULT" ]]; then
          if [[ ${DD} == "GLOW_low_visibility_estuaries" ]]; then SS="9 16 17"; else SS="2 7 9"; fi
        else
          SS=${SEEDS}
        fi
        ##
        for SEED in ${SS}
        do
          # DATA
          NAME="${DD}.CV.${GRP}${SUFFIX}"
          OUT="${MODEL}.${NAME}.seed${SEED}${QUICKY}"
          #
          echo "Model: ${MODEL}"
          echo "Seed: ${SEED}"
          echo "Images: ${IMG_DIR}"
          echo "Annot: ${ANNOT_FN}"
          echo "Device: ${DEVICE}"
          echo "Epochs: ${EPOCHS}"
          echo "Resume: ${RESUME}"
          echo "Iters/epochs: ${ITERS_OR_EPOCHS}"
          echo "Conda: ${ENV}"
          echo "Dataset: ${DD}"
          echo "Groups: ${GRP}"
          echo "Quicky: ${QUICKY}"
          echo "Output: ${RESULTS_DIR}/${OUT}"

          if [[ -n ${RESUME} || -n ${FOLD} || ! -d ${RESULTS_DIR}/${OUT} ]]; then
            mkdir -p "${RESULTS_DIR}/${OUT}"

            # RUN
            if [[ ${BATCH} == "DEFAULT" ]]; then
              if [[ ${MODEL} == "dt2" ]]; then B=4; elif [[ ${MODEL} == "yolo8" ]]; then B=5; else B=-1; fi
            else
              B=${BATCH}
            fi

            for FFF in ${FOLDS}
            do
              TRAIN_DATA="--annot_fn ${DATA_DIR}/resources/${DD}/${ANNOT_FN} --img_dir ${DATA_DIR}/resources/${DD}/${IMG_DIR}"
              TEST_DATA="--n_folds 3 --val_size 0.2"
              DEFAULTS="--do_not_save_pred_frames --eval_period epoch --vis_threshold 0.1 --threshold 0 --detections_per_image 100 --run_predictions val test --prediction_models best last"
              echo "Fold: ${FFF}"
              CMD="python train.py ${EXTRA_ARGS} --fold ${FFF} ${GRPS_ARG} ${RESUME} ${QUICKY} --model ${MODEL} --seed ${SEED} --imgs_per_batch ${B} ${ITERS_OR_EPOCHS} ${TEST_DATA} ${TRAIN_DATA} --output_dir "${RESULTS_DIR}/${OUT}" ${DEFAULTS} ${DEVICE}"
              echo ${CMD}
              echo "START TRAINING:" `date`
              if [[ -n ${QUICKY} ]]; then ${CMD}; else ${CMD} &> "${RESULTS_DIR}/${OUT}/train.fold${FFF}.log"; fi
              echo "END TRAINING:" `date`
            done
            #
          else
            echo "Directory already exists! Skipping."
          fi
        done
      done
    done
  done
done
conda deactivate
