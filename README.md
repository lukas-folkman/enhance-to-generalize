# enhance-to-generalize

<img src="https://lukas-folkman.github.io/img/jell-AI-fish.png?raw=true" alt="jelly-fish-AI-logo" width="150"/>

## This repository provides the code for the following paper

Folkman L, Pitt KA, Stantic B (2025). <B>A data-centric framework for combating domain shift in underwater object detection with image enhancement</B>. <I>Applied Intelligence</I>, 55, 272. <A HREF="https://doi.org/10.1007/s10489-024-06224-0" TARGET="_blank">https://doi.org/10.1007/s10489-024-06224-0</A>

## Installation of required packages

### `dt2` environment to run Faster R-CNN (detectron2) models

```
conda env create --file envs/dt2.with_cuda.yaml
conda activate dt2

# pip install of local detectron repository
cd x/y/z
mkdir -p tools
cd tools/
git clone https://github.com/lukas-folkman/detectron2.git
cd detectron2/
python -m pip install -e .
```

### `yolo8` environment to run YOLOv8 (ultralytics) models

```
conda env create --file envs/yolo8.with_cuda.yaml
conda activate yolo8
```

### `enhance_analysis` environment to preprocess data and analyse results

```
conda env create --file envs/enhance_analysis.no_cuda.yaml
conda activate enhance_analysis
```

### Download model weights pretrained on the COCO dataset

- YOLOv8 large model: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt
- Faster R-CNN model: https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl

### Set the following environmental variables
- `YOLO_MODELS` to point to the directory containing the `yolov8l.pt` model file
- `DT2_MODEL_ZOO` to point to the `x/y/z/detectron2/configs` directory
- `DT2_FASTER_RCNN_WEIGHTS` to point to the `model_final_280758.pkl` model file

```
export YOLO_MODELS="x/y/z/ultralytics_models"
export DT2_MODEL_ZOO="x/y/z/detectron2/configs"
export DT2_FASTER_RCNN_WEIGHTS="x/y/z/model_final_280758.pkl"
```

## Prepare data

### Download the four publicly available datasets:
- DeepFish with the extended annotations (https://github.com/tamim662/YOLO-Fish)
- MBEEC-Low-Vis (https://github.com/slopezmarcano/dataset-fish-detection-low-visibility)
- Jellytoring (https://doi.org/10.5281/zenodo.6832131)
- S-UODAC (https://github.com/mousecpn/DMC-Domain-Generalization-for-Underwater-Object-Detection).

### Convert all downloaded datasets to COCO format and perform quality control
```
conda activate enhance_analysis
jupyter notebook prepare_datasets.ipynb
```

### Make sure your data files match the structure defined in `config.py`
```
../data/resources/DeepFish/images/
../data/resources/DeepFish/DeepFish.json

../data/resources/Jellytoring/images/
../data/resources/Jellytoring/Jellytoring.json

../data/resources/GLOW_low_visibility_estuaries/images/
../data/resources/GLOW_low_visibility_estuaries/GLOW_low_visibility_estuaries.json

../data/resources/S-UODAC/images/
../data/resources/S-UODAC/S-UODAC.json
```

## Image enhancement

GPU implementation of MSRCR
```
conda activate dt2
WIDTH=1333 # depends on the specific dataset
HEIGHT=750 # depends on the specific dataset
python MSRCR_with_rescaling.py \
--img_dir "x/y/z/images" \
--out_dir "x/y/z/images_MSRCR" \
--flavour MSRCR --framework nvidia \
--resize_width ${WIDTH} --resize_height ${HEIGHT}
```

Other image enhancement methods
```
conda activate enhance_analysis
TRANSFORM="autocontrast"
python transform_images.py --img_transform "${TRANSFORM}" \
--img_dir "x/y/z/images" --output_dir "x/y/z/images_${TRANSFORM}"
# repeat for all enhancement methods
```

## Silhouette analysis

```
conda activate enhance_analysis
TRANSFORM="autocontrast"
python silhouette_analysis.py "images_${TRANSFORM}" 256
# repeat for all enhancement methods
```

## Train and test models

```
# submit as a PBS job
qsub train_and_test.pbs
# or execute locally
# bash train_and_test.pbs
```

## Evaluate results

```
conda activate enhance_analysis
jupyter notebook collect_and_eval_all_results.ipynb
jupyter notebook results_to_plots_and_tables.ipynb
jupyter notebook plot_silhouette_analysis.ipynb
jupyter notebook plot_enhancement_and_detection_examples.ipynb
```
