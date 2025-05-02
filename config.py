import os
import numpy as np

#############################################################
# MAKE SURE THE FOLLOWING DATA STRUCTURE MATCHES YOUR FILES #
#############################################################
DATA_DIR = '../data'
RESULTS_DIR = '../results'
ANNOT_ROOT = os.path.join(os.environ["HOME"], 'jellies', 'data')

_DATASET_NAMES = ['DeepFish', 'GLOW_low_visibility_estuaries', 'Jellytoring', 'S-UODAC']
_DATASET_DIRS = {d: os.path.join(DATA_DIR, 'resources', d) for d in _DATASET_NAMES}
DATASETS = {
    d: (os.path.join(_DATASET_DIRS[d], f'{d}.json'), os.path.join(_DATASET_DIRS[d], 'images')) \
    for d in _DATASET_DIRS
}

#############################################################

DATA_NAME = 'jellies'
DEVICES = ['cpu', 'cuda', 'mps', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:4']
SHORT_CFG_FN = 'config.short.yaml'
FULL_CFG_FN = 'config.yaml'
EXP_CFG_FN = 'config.exp.yaml'
# THE BELOW DOES NOT WORK, DT2 NEEDS AN ABSOLUTE PATH
# DT2_CONFIGS = '../tools/detectron2/configs'
HOME = os.getenv('HOME')
DT2_CONFIGS = os.getenv('DT2_MODEL_ZOO', default=f'{HOME}/jellies/tools/detectron2/configs')
DT2_FASTER_RCNN_WEIGHTS = os.getenv('DT2_FASTER_RCNN_WEIGHTS', default=f'{HOME}/jellies/tools/detectron2_models/faster_rcnn_R_50_FPN_3x/model_final_280758.pkl')
DT2_RETINA_WEIGHTS = os.getenv('DT2_RETINA_WEIGHTS', default=f'{HOME}/jellies/tools/detectron2_models/retinanet_R_50_3x/model_final_5bd44e.pkl')
YOLO_MODELS = os.getenv('YOLO_MODEL_ZOO', default=f'{HOME}/jellies/tools/ultralytics/models')
QUEUE_NAME = 'jelly_queue'
TYPE_JSON = 'application/json'
TYPE_JPG = 'image/jpg'

CAT_JELLY = 'Jellyfish'

BEST_WEIGHTS_FN = os.path.join('weights', 'best.pt')
BEST_WEIGHTS_FN2 = os.path.join('weights', 'best2.pt')
LAST_WEIGHTS_FN = os.path.join('weights', 'last.pt')

MODEL_NAMES = dict(dt2='Faster R-CNN', yolo8='YOLOv8')
DATASETS = {
    # dataset_name: (annot_fn, img_dir)
    d: (os.path.join(_DATASET_DIRS[d], f'{d}.json'), os.path.join(_DATASET_DIRS[d], 'images')) for d in _DATASET_DIRS
}

MAIN_DATASETS = ['DeepFish', 'GLOW_low_visibility_estuaries', 'Jellytoring']
RENAME_DATASETS = {
    'GLOW_low_visibility_estuaries': 'MBEEC-Low-Vis'
}
ALL_SEEDS = {
    'DeepFish': [2, 7, 9], 'GLOW_low_visibility_estuaries': [9, 16, 17], 'Jellytoring': [2, 7, 9], 'S-UODAC': [0, 1, 2]
}
RENAME_PPS = {
    np.nan: 'No processing', None: 'No processing', 'NONE': 'No processing', 'grey_world': 'Grey world', 'ace': 'ACE',
    'adjust_gamma_down': 'Gamma down', 'adjust_gamma_up': 'Gamma up',
    'adjust_log': 'Adjust log', 'adjust_sigmoid': 'Adjust sigmoid', 'autocontrast': 'Auto-contrast',
    'unsharp_mask': 'Unsharp mask',
    'resized_MSRCR': 'MSRCR_old', 'resized_MSRCR_15_80_250': 'MSRCR_15_80_250',
    'resized_MSRCR_15_80_250_fk': 'MSRCR (full res.)', 'resized_MSRCR_15_80_250_new': 'MSRCR'
}
ALL_PPS = [
    'unsharp_mask', 'adjust_gamma_down', 'adjust_gamma_up', 'adjust_log', 'adjust_sigmoid',
    'DCP', 'CAP', 'FUnIE-GAN',
    'CLAHE', 'autocontrast', 'grey_world', 'ace',
    'ARCR', 'MLLE',
    # 'resized_MSRCR',
    'resized_MSRCR_15_80_250_new'
]
TOP_PPS = [
    'resized_MSRCR', 'MLLE', 'grey_world', 'ARCR'
]
NO_GROUPS_PPS = [
    'resized_MSRCR', 'MLLE', 'grey_world'
]


DARK_BLUE = '#175e91'
DARK_ORANGE = '#c76100'
DARK_GREEN = '#258325'
LIGHT_BLUE = '#a1c9f4'
LIGHT_ORANGE = '#ffb482'
LIGHT_GREEN = '#8de5a1'
BLUE = '#1f77b4'
ORANGE = '#ff7f0e'
GREEN = '#2ca02c'

DARK = [DARK_BLUE, DARK_ORANGE, DARK_GREEN]
LIGHT = [LIGHT_BLUE, LIGHT_ORANGE, LIGHT_GREEN]
BASE = [BLUE, ORANGE, GREEN]

FONT_SIZE = 9
SMALL_FONT = 7
CM = 1 / 2.54
PR_CURVE_SIZE = (7 * CM, 7 * CM)

RC_PAPER = {
    "font.sans-serif": ["Helvetica"],
    "legend.title_fontsize": FONT_SIZE,
    "figure.titlesize": FONT_SIZE,

    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,

    "axes.linewidth": 0.5,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1,
    "lines.markersize": 3,
    "patch.linewidth": 0.5,

    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,

    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "xtick.minor.size": 1,
    "ytick.minor.size": 1,

    "xtick.major.pad": 2,
    "ytick.major.pad": 2,
    "xtick.minor.pad": 2,
    "ytick.minor.pad": 2,
    "axes.labelpad": 2
}


def setup_plotting(style='ticks', context='notebook', font_scale=1, font_size=None,
                   use_tex=False, rc=None):
    import seaborn as sns
    import matplotlib as mpl
    PLOT_CONTEXTS = dict(paper=0.8, notebook=1, talk=1.5, poster=2)
    assert context in PLOT_CONTEXTS
    sns.set_color_codes()
    sns.set_style(style)

    if rc is None:
        rc = {}
    if 'font.sans-serif' not in rc:
        rc['font.sans-serif'] = ['Helvetica', 'Arial', 'Liberation Sans', 'sans-serif']
    if 'legend.title_fontsize' not in rc:
        rc['legend.title_fontsize'] = (11 if font_size is None else font_size) * PLOT_CONTEXTS[context] * font_scale
    if 'figure.titlesize' not in rc:
        rc['figure.titlesize'] = (11 if font_size is None else font_size) * PLOT_CONTEXTS[context] * font_scale
    rc['text.usetex'] = use_tex
    rc['svg.fonttype'] = 'none'

    if font_size is not None:
        for font_key in ['axes.labelsize', 'axes.titlesize', 'legend.fontsize',
                         'xtick.labelsize', 'ytick.labelsize', 'font.size']:
            rc[font_key] = font_size

    context_object = sns.plotting_context(context, font_scale=font_scale, rc=rc)
    mpl.rcParams.update(rc)
    mpl.rcParams.update(context_object)

    return context_object
