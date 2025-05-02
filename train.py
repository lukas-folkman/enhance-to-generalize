#!/usr/bin/env python
# coding: utf-8

from config import DEVICES, DATASETS
import argparse
import os
from datetime import datetime
from collections import defaultdict
import cv_tools as utils

YOLO_MODELS = [
    f'{model}{ext}' for model in ['yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x',
                                  'yolov8n-seg', 'yolov8s-seg', 'yolov8m-seg', 'yolov8l-seg', 'yolov8x-seg']
    for ext in ['', '.pt', '.yaml']
]


def training_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', choices=['yolo8', 'dt2', 'retina'], required=True,
                        help='Choose if to train YOLO or Faster R-CNN or RetinaNet.')
    parser.add_argument('--base_model', choices=YOLO_MODELS + [
        'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml',
        'COCO-Detection/retinanet_R_50_FPN_3x.yaml',
        'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml',
    ],
                        help='If not specified, this will default to "yolov8l" for YOLO and "faster_rcnn_R_50_FPN_3x" '
                             'for Detectron2. These are good options for accuracy-focused applications. If speed is '
                             'more important, try one of the smaller YOLO models.')
    parser.add_argument('--retina', action='store_true')
    parser.add_argument('--weights',
                        help='Only DT2. Pretrained weights.')
    parser.add_argument('--lightly_pretrained', action='store_true',
                        help='Only DT2. Changes input format to RGB and normalizes the inputs with '
                             'MODEL.PIXEL_MEAN 123.675,116.280,103.530 and MODEL.PIXEL_STD 58.395,57.120,57.375')

    # Data
    parser.add_argument('--dataset', choices=DATASETS.keys(),
                        help='Choose one of preprocessed external datasets (not allowed with "annot_fn" and "img_dir"')
    parser.add_argument('--annot_fn', nargs='+',
                        help='JSON dataset/annotations file in the COCO format. '
                             'This file will be used for training, validation (if "val_size" specified) and testing '
                             '(if "test_size" or "n_folds" specified).')
    parser.add_argument('--img_dir', nargs='+',
                        help='Directory with images referenced in "annot_fn".')

    parser.add_argument('--test_annot_fn', nargs='+',
                        help='Test set JSON dataset/annotations file in the COCO format (mutually exclusive with '
                             '"test_size" and "n_folds").')
    parser.add_argument('--test_img_dir', nargs='+',
                        help='Directory with images referenced in "test_annot_fn".')
    parser.add_argument('--sort_cats', action='store_true',
                        help='Sort categories after reading the datasets (can help make different datasets compatible)')
    parser.add_argument('--sort_images', action='store_true',
                        help='Sort images after reading the datasets (can help make different datasets compatible)')
    parser.add_argument('--val_annot_fn', nargs='+',
                        help='Validation set JSON dataset/annotations file in the COCO format (mutually exclusive with '
                             '"val_size").')
    parser.add_argument('--val_img_dir', nargs='+',
                        help='Directory with images referenced in "val_annot_fn".')

    parser.add_argument('--merge_categories_as', help='Merge all categories into a single category of the given name')

    # Evaluation
    parser.add_argument('--fold', type=int,
                        help='Run just this fold')
    parser.add_argument('--n_folds', type=int,
                        help='Number of folds for stratified cross-validation (mutually exclusive with "test_size" '
                             'and "test_annot_fn"). If not set, a single train/val/test split will be performed.')
    parser.add_argument('--test_groups', nargs='+')
    parser.add_argument('--test_size', type=float,
                        help='Fraction or absolute value (mutually exclusive with "n_folds" and "test_annot_fn").')
    parser.add_argument('--val_size', type=float, help='Fraction or absolute value, validation dataset will be used for'
                                                       ' validation (e.g. early-stopping).')
    parser.add_argument('--multi_label', action='store_true',
                        help='Mutually exclusive with "filename_groups". If set, every split will have a similar number'
                             ' of categories (summed across all images). '
                             'If not set, every split will have a similar number of images with specific combinations '
                             'of categories.')
    parser.add_argument('--filename_groups', action='store_true',
                        help='Splitting into train/test/val or cross-validation will preserve grouping based on '
                             'filenames (mutually exclusive with "multi_label"). '
                             'This means that all images with a given prefix (up to the first underscore character: '
                             'prefix_suffix.jpg) will be always grouped together and NOT split across train/test/val '
                             'partitions.')
    parser.add_argument('--group_splitter', default='_', help='How to split if "filename_groups" set.')
    parser.add_argument('--filename_groups_test_only', action='store_true')
    parser.add_argument('--stratify_bbox_sizes', action='store_true',
                        help='Stratification is typically done based on categories but if dealing with a single '
                             'category, one can stratify based on the size of a bounding box area.')

    parser.add_argument('--extra_train_annot_fn', nargs='+',
                        help='Extra training JSON dataset/annotations file in the COCO format (this part of the dataset'
                             ' will not be subject to splitting into train/test or cross-validation).')
    parser.add_argument('--extra_train_img_dir', nargs='+',
                        help='Directory with images referenced in "extra_train_annot_fn".')
    parser.add_argument('--extra_val_size', type=float,
                        help='Option to split "extra_train_annot_fn" into train/val (e.g. early-stopping).')
    parser.add_argument('--drop_test_groups_in_extra_train', action='store_true',
                        help='Fix it if extra train overlaps with test set groups.')
    parser.add_argument('--drop_val_groups_in_extra_train', action='store_true',
                        help='Fix it if extra train overlaps with test set groups.')
    parser.add_argument('--subsample_train', type=float,
                        help='Option to subsample training dataset for experimenting with learning completeness.')
    parser.add_argument('--subsample_groups', type=float,
                        help='Option to subsample training dataset for experimenting with learning completeness.')

    # Model hyper-parameters
    parser.add_argument('--epochs', type=float,
                        help='Number of training epochs.')
    parser.add_argument('--max_iter', type=int,
                        help='Only DT2. Number of training iterations (mutually exclusive with "epochs".')
    parser.add_argument('--eval_period',
                        help='Only DT2. Determines how often to evaluate current model using the validation split. '
                             'Specify N iterations or set to "epoch" to run validation every epoch. '
                             'YOLO runs validation every epoch by default.')
    parser.add_argument('--no_early_stopping', action='store_true',
                        help='Only DT2. Do not stop early, ever.')
    parser.add_argument('--early_stopping_wait', type=int, default=10,
                        help='Only DT2. Stop early if validation loss does not improve over the median of '
                             '"early_stopping_wait" number of last evaluations.')
    parser.add_argument('--early_no_stopping_period', type=int, default=20,
                        help='Only DT2. Do not consider stopping for the first N epochs.')
    parser.add_argument('--checkpoint_period', type=int,
                        help='Only DT2. Specify N iterations to routinely save model weights (checkpoint).')
    parser.add_argument('--imgs_per_batch', type=int,
                        help='Number of images per batch. YOLO defaults to AutoBatch, which uses the maximum number of '
                             'images that fits into memory. Detectron2 defaults to 4.')
    parser.add_argument('--rpn_batch_size_per_img', type=int,
                        help='Only DT2. Number of regions to sample per image during RPN (Region Proposal Network) '
                             'training. Defaults to 256.')
    parser.add_argument('--roi_batch_size_per_img', type=int,
                        help='Only DT2. Number of regions of interest (ROIs) to sample per image during training. '
                             'Defaults to 512.')
    parser.add_argument('--dropout_cnn', type=float,
                        help='Only DT2. Dropout for the ResNet backbone. No other architecture supported.')
    parser.add_argument('--dropout_fc', type=float,
                        help='Only DT2. Dropout for the fully connected head. No other architecture supported.')
    parser.add_argument('--base_lr', type=float,
                        help='Base value for learning rate.')
    parser.add_argument('--final_lr', type=float,
                        help='Only YOLO. Final value of learning rate.')
    parser.add_argument('--lr_gamma', type=float,
                        help='Only DT2. Multiple learning rate by "lr_gamma" once reaching "gamma_epochs".')
    parser.add_argument('--gamma_epochs', type=int, nargs='+',
                        help='Only DT2. Multiple learning rate by "lr_gamma" once reaching "gamma_epochs".')

    parser.add_argument('--weight_decay', type=float,
                        help='Weight decay - L2 regularization.')
    parser.add_argument('--filter_empty', action='store_true',
                        help='If set, images without annotations will be removed from training.')
    parser.add_argument('--filter_sizes', nargs='+', choices=list(utils.MIN_MAX_AREA.keys()),
                        help='If set, annotations with specified sizes will be dropped and masked in the images.')
    parser.add_argument('--img_size', type=int,
                        help='Only YOLO. Resizing of the image for training the model. Defaults to 640.')
    parser.add_argument('--min_max_img_size', type=int, nargs='+',
                        help='Only DT2. A tuple of "min_img_size max_img_size" with "min_img_size" being the size of '
                             'the smallest side of the image and "max_img_size" being the maximum size of the side of '
                             'the image. Defaults to "800 1333"')
    parser.add_argument('--no_resizing', action='store_true',
                        help='No resizing of the images. Mutually exclusive with "img_size" and "min_max_img_size".')

    parser.add_argument('--img_transforms', nargs='+',
                        choices=[(f'{s}:{t}' if s is not None else t) for t in utils.IMG_TRANSFORMS for s in
                                 [None, 'train', 'test', 'val']],
                        help='Applies image transformation before training/validation/testing. Experimental feature.')
    parser.add_argument('--coinflip_transform', action='store_true',
                        help='Chooses if to apply the specified image transformation(s) after flipping a coin.')

    parser.add_argument('--threshold', type=float, default=0.0,
                        help='Minimum prediction score for the detection to be considered a detection (included).')
    parser.add_argument('--NMS_threshold', type=float, default=0.7,
                        help='Threshold for non-maximum suppression (NMS) postprocessing of raw predictions.')
    parser.add_argument('--vis_threshold', type=float,
                        help='Test time prediction visualization threshold (visualization only).')
    parser.add_argument('--detections_per_image', type=int, default=100,
                        help='Maximum number of detections per image.')
    parser.add_argument('--cfg_list', nargs='+',
                        help='Only DT2. A list of additional configurations for the model: "cfg_arg1 value1 ...')
    parser.add_argument('--device', choices=DEVICES,
                        help='Device to be used for computation. Defaults to "cuda:0" if cuda is available.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random number generator seed for reproducibility.')
    parser.add_argument('--resume', action='store_true',
                        help='Only DT2. Resume stopped/failed training from the last checkpoint.')

    # Optimizer
    parser.add_argument('--optimizer', choices=['SGD', 'Adam', 'AdamW', 'RMSProp'], default='SGD',
                        help='Only YOLO. Algorithm used for gradiant descent.')
    parser.add_argument('--use_sharper_scheduler', action='store_true',
                        help='Only YOLO. Experimental feature to adjust the learning rate schedule when training for '
                             'many more than 100 epochs.')
    parser.add_argument('--target_decay_epoch', type=int,
                        help='Only YOLO. Experimental feature used only with "use_sharper_scheduler".')

    # Augmentations
    parser.add_argument('--hsv_h', type=float, help='hsv_h augmentation, default=0.015')
    parser.add_argument('--hsv_s', type=float, help='hsv_s augmentation, default=0.7')
    parser.add_argument('--hsv_v', type=float, help='hsv_v augmentation, default=0.4')
    parser.add_argument('--degrees', type=float, help='degrees augmentation, default=0.0')
    parser.add_argument('--translate', type=float, help='translate augmentation, default=0.1')
    parser.add_argument('--scale', type=float, help='scale augmentation, default=0.5')
    parser.add_argument('--shear', type=float, help='shear augmentation, default=0.0')
    parser.add_argument('--perspective', type=float, help='perspective augmentation, default=0.0')
    parser.add_argument('--flipud', type=float, help='flipud augmentation, default=0.0')
    parser.add_argument('--fliplr', type=float, help='fliplr augmentation, default=0.5')
    parser.add_argument('--mosaic', type=float, help='mosaic augmentation, default=1.0')
    parser.add_argument('--close_mosaic', type=int, help='mosaic augmentation, default=10')
    parser.add_argument('--mixup', type=float, help='mixup augmentation, default=0.0')
    parser.add_argument('--copy_paste', type=float, help='copy_paste augmentation, default=0.0')

    # Outputs
    parser.add_argument('--run_predictions', nargs='+', choices=['train', 'val', 'test'],
                        help='Specify splits for which to run predictions after training finished '
                             '(mutually exclusive with "run_predictions_all").')
    parser.add_argument('--run_predictions_all', action='store_true',
                        help='Run predictions for all splits (mutually exclusive with "run_predictions").')
    parser.add_argument('--prediction_models', nargs='+', choices=['last', 'best'],
                        help='Which model(s) to use for running predictions after training finished. '
                             'If not set, both "last" and "best" will be used.')
    parser.add_argument('--do_not_save_pred_frames', action='store_true',
                        help='When running predictions after training finished, do NOT save predicted frames.')
    parser.add_argument('--do_not_save_data', action='store_true',
                        help='Do not save input data (because it was already created).')
    parser.add_argument('--output_dir', default='output',
                        help='Specify the output directory.')

    # Debugging
    parser.add_argument('--quick_debug', action='store_true',
                        help='Run only short training/predicting session for debugging purposes.')
    parser.add_argument('--skip_training', action='store_true',
                        help='Useful to recreate data and run predictions with evaluation.')
    parser.add_argument('--only_create_data', action='store_true',
                        help='Only YOLO. Create data in the YOLO format but do NOT run training and predictions.')
    parser.add_argument('--hub_format', action='store_true',
                        help='Only YOLO. Create data in the YOLO Ultralytics Hub format.')

    return parser


def main(args):

    print('\nLog for executing the following training regime:\ntrain.py', end=' ')
    for arg in vars(args):
        print(f'--{arg} {getattr(args, arg)}', end=' ')
    print('\nSTARTED:', datetime.now().strftime("%d %B %Y, %H:%M:%S"), '\n')

    if args.model == 'dt2':
        import dt2_utils
        strict = True
        train_func = dt2_utils.dt2_train
        if args.lightly_pretrained:
            strict = False
            assert args.weights is not None
            if args.cfg_list is None:
                args.cfg_list = []
            args.cfg_list.extend(
                ['MODEL.PIXEL_MEAN', '123.675,116.280,103.530',
                 'MODEL.PIXEL_STD', '58.395,57.120,57.375',
                 'INPUT.FORMAT', 'RGB']
            )

        kwargs = dict(
            weights=args.weights, max_iter=args.max_iter, save_best_checkpoint=args.eval_period is not None,
            early_stopping=not args.no_early_stopping, early_no_stopping_period=args.early_no_stopping_period,
            dropout_cnn=args.dropout_cnn, dropout_fc=args.dropout_fc,
            rpn_batch_size_per_img=args.rpn_batch_size_per_img,
            roi_batch_size_per_img=args.roi_batch_size_per_img, min_max_img_size=args.min_max_img_size,
            lr_gamma=args.lr_gamma, gamma_epochs=args.gamma_epochs, retina=args.retina,
            checkpoint_period=args.checkpoint_period, resume=args.resume, cfg_list=args.cfg_list, strict=strict,
            copy_paste=args.copy_paste is not None and args.copy_paste > 0
        )
        if args.imgs_per_batch is None or args.imgs_per_batch == -1:
            args.imgs_per_batch = 4
        if args.base_model is None:
            args.base_model = 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml' if not args.retina else 'COCO-Detection/retinanet_R_50_FPN_3x.yaml'

    elif args.model == 'yolo8':
        import yolo_utils
        train_func = yolo_utils.yolo_train
        kwargs = dict(
            final_lr=args.final_lr, optimizer=args.optimizer, img_size=args.img_size,
            use_sharper_scheduler=args.use_sharper_scheduler, target_decay_epoch=args.target_decay_epoch,
            only_create_data=args.only_create_data, hub_format=args.hub_format,

            hsv_h=args.hsv_h, hsv_s=args.hsv_s, hsv_v=args.hsv_v, degrees=args.degrees, translate=args.translate,
            scale=args.scale, shear=args.shear, perspective=args.perspective, flipud=args.flipud, fliplr=args.fliplr,
            mosaic=args.mosaic, close_mosaic=args.close_mosaic, mixup=args.mixup, copy_paste=args.copy_paste
        )

        if args.imgs_per_batch is None:
            args.imgs_per_batch = -1
        if args.base_model is None:
            args.base_model = 'yolov8l' if not args.quick_debug else 'yolov8n'

    else:
        raise ValueError

    data_filters = utils.get_default_data_filters()
    utils.set_data_filters(data_filters, which_splits='train',
                           filter_empty=args.filter_empty, filter_sizes=args.filter_sizes)

    img_transforms = utils.get_default_img_transforms()
    if args.img_transforms:
        parsed_img_transforms = defaultdict(dict)
        for t in args.img_transforms:
            if ':' in t:
                assert t.count(':') == 1
                split, t = t.split(':')
                assert split in ['train', 'val', 'test']
                parsed_img_transforms[split][t] = True
            else:
                parsed_img_transforms['global'][t] = True

        for which_splits in parsed_img_transforms:
            utils.set_img_transforms(
                img_transforms=img_transforms,
                which_splits=which_splits if which_splits != 'global' else ['train', 'val', 'test'],
                **parsed_img_transforms[which_splits]
            )

    print(f'Expanding {args.annot_fn} into:')
    if len(args.annot_fn) == 1 and os.path.isdir(args.annot_fn[0]):
        excl_list = [os.path.abspath(fn) for fn in (args.test_annot_fn if args.test_annot_fn else []) +
                     (args.val_annot_fn if args.val_annot_fn else []) +
                     (args.extra_train_annot_fn if args.extra_train_annot_fn else [])]
        args.annot_fn = sorted([os.path.join(args.annot_fn[0], fn) for fn in os.listdir(args.annot_fn[0])
                         if fn.lower().endswith('.json')
                         and os.path.abspath(os.path.join(args.annot_fn[0], fn)) not in excl_list])
    print('\n'.join(args.annot_fn))

    shared_kwargs = dict(
        args=args, data_json_fns=args.annot_fn, data_img_dirs=args.img_dir,
        test_json_fns=args.test_annot_fn, test_img_dirs=args.test_img_dir, val_json_fns=args.val_annot_fn, val_img_dirs=args.val_img_dir, n_folds=args.n_folds, fold=args.fold,
        epochs=args.epochs, base_model=args.base_model, test_size=args.test_size, test_groups=args.test_groups,
        sort_cats=args.sort_cats, merge_categories_as=args.merge_categories_as, sort_images=args.sort_images,
        val_size=args.val_size, multi_label=args.multi_label, filename_groups=args.filename_groups,
        filename_groups_test_only=args.filename_groups_test_only, group_splitter=args.group_splitter,
        extra_train_json_fns=args.extra_train_annot_fn, extra_train_img_dirs=args.extra_train_img_dir,
        drop_test_groups_in_extra_train=args.drop_test_groups_in_extra_train,
        drop_val_groups_in_extra_train=args.drop_val_groups_in_extra_train,
        extra_val_size=args.extra_val_size, subsample_train=args.subsample_train, subsample_groups=args.subsample_groups,
        imgs_per_batch=args.imgs_per_batch, base_lr=args.base_lr,
        weight_decay=args.weight_decay, eval_period=args.eval_period,
        threshold=args.threshold, NMS_threshold=args.NMS_threshold, detections_per_image=args.detections_per_image,
        data_filters=data_filters, img_transforms=img_transforms, coinflip_transform=args.coinflip_transform,
        vis_threshold=args.vis_threshold if args.vis_threshold != 0 else None, device=args.device, seed=args.seed,
        run_predictions=args.run_predictions if args.run_predictions else args.run_predictions_all,
        prediction_models=args.prediction_models,
        save_pred_frames=not args.do_not_save_pred_frames, output_dir=args.output_dir, do_not_save_data=args.do_not_save_data,
        stratify_bbox_sizes=args.stratify_bbox_sizes, skip_training=args.skip_training, quick_debug=args.quick_debug
    )

    data_fn_prefix = train_func(**shared_kwargs, **kwargs)

    print('FINISHED:', datetime.now().strftime("%d %B %Y, %H:%M:%S"))

    #for i in range(args.n_folds) if args.n_folds is not None and args.n_folds != 1 else [None]:
    #    for split in ['train', 'test', 'val']:
    #        utils.compress_file(utils.get_data_fn(prefix=data_fn_prefix, random_seed=args.seed, split=split, fold=i), force=True)


if __name__ == '__main__':

    parser = training_args_parser()
    args = parser.parse_args()

    if args.annot_fn and len(args.annot_fn) == 1 and ';' in args.annot_fn[0]:
        args.annot_fn = args.annot_fn[0].split(';')
    if args.test_annot_fn and len(args.test_annot_fn) == 1 and ';' in args.test_annot_fn[0]:
        args.test_annot_fn = args.test_annot_fn[0].split(';')
    if args.val_annot_fn and len(args.val_annot_fn) == 1 and ';' in args.val_annot_fn[0]:
        args.val_annot_fn = args.val_annot_fn[0].split(';')
    if args.extra_train_annot_fn and len(args.extra_train_annot_fn) == 1 and ';' in args.extra_train_annot_fn[0]:
        args.extra_train_annot_fn = args.extra_train_annot_fn[0].split(';')

    assert args.model in ['yolo8', 'dt2']

    if args.dataset is not None:
        assert args.annot_fn is None and args.img_dir is None
        args.annot_fn, args.img_dir = DATASETS[args.dataset]
    else:
        assert args.annot_fn is not None and args.img_dir is not None

    if args.model != 'yolo8' and \
            (args.only_create_data or args.hub_format or args.target_decay_epoch is not None or
             args.use_sharper_scheduler or args.optimizer != 'SGD' or args.img_size is not None
             or args.final_lr is not None):
        parser.error('Some arguments are only allowed for model "yolo8"')

    if args.model != 'dt2' and \
            (args.resume or args.cfg_list is not None or args.min_max_img_size is not None
             or args.roi_batch_size_per_img is not None or args.rpn_batch_size_per_img is not None
             or args.checkpoint_period is not None or args.max_iter is not None):
        parser.error('Some arguments are only allowed for model "dt2"')

    if args.test_groups:
        if args.test_size:
            parser.error('Cannot set "test_groups" and "test_size"')
        if args.test_annot_fn:
            parser.error('Cannot set "test_groups" and "test_annot_fn"')
        if args.n_folds:
            parser.error('Cannot set "test_groups" and "n_folds"')

    if not args.val_size and not args.test_size and not args.test_annot_fn and not args.val_annot_fn and \
            not args.test_groups and (args.n_folds is None or args.n_folds < 2):
        parser.error(
            'Cannot train without validation or test dataset. Specify one of "val_size", "test_size", "test_groups", "n_folds", or "test_annot_fn".')

    if args.test_size and args.test_annot_fn:
        parser.error('Cannot set "test_size" and "test_annot_fn"')

    if args.val_size and args.val_annot_fn:
        parser.error('Cannot set "val_size" and "val_annot_fn"')

    if args.test_size and args.n_folds is not None and args.n_folds != 1:
        parser.error('Cannot set "test_size" and "n_folds"')

    if args.test_annot_fn and args.n_folds is not None and args.n_folds != 1:
        parser.error('Cannot set "test_size" and "n_folds"')

    if args.multi_label and (args.filename_groups or args.filename_groups_test_only):
        parser.error('Cannot set "multi_label" and "filename_groups"/"filename_groups_test_only"')

    if not (len(args.img_dir) == 1 or len(args.img_dir) == len(args.annot_fn)):
        parser.error('list only 1 "img_dir" or list "img_dir" for every "annot_fn"')
    if not (args.test_img_dir is None or len(args.test_img_dir) == 1 or len(args.test_img_dir) == len(
            args.test_annot_fn)):
        parser.error('list only 1 "test_img_dir" or list "test_img_dir" for every "test_annot_fn"')
    if not (args.val_img_dir is None or len(args.val_img_dir) == 1 or len(args.val_img_dir) == len(
            args.val_annot_fn)):
        parser.error('list only 1 "val_img_dir" or list "val_img_dir" for every "val_annot_fn"')
    if not (args.extra_train_img_dir is None or len(args.extra_train_img_dir) == 1 or len(
            args.extra_train_img_dir) == len(args.extra_train_annot_fn)):
        parser.error('list only 1 "extra_train_img_dir" or list "extra_train_img_dir" for every "extra_train_annot_fn"')

    if args.test_annot_fn and args.test_img_dir is None and len(args.img_dir) != 1:
        parser.error('"test_annot_fn" missing "test_img_dir"')
    if args.val_annot_fn and args.val_img_dir is None and len(args.img_dir) != 1:
        parser.error('"val_annot_fn" missing "val_img_dir"')
    if args.extra_train_annot_fn and args.extra_train_img_dir is None and len(args.img_dir) != 1:
        parser.error('"extra_train_annot_fn" missing "extra_train_img_dir"')

    if args.coinflip_transform and args.n_folds is not None and args.n_folds != 1:
        parser.error('cannot do "coinflip_transform" with K-fold cross-validation')

    for annot_fn in args.annot_fn:
        if not (os.path.isfile(annot_fn) or (os.path.isdir(annot_fn) and len(args.annot_fn) == 1)):
            parser.error(f'"annot_fn" must be an existing file: {annot_fn}')

    for img_dir in args.img_dir:
        if not os.path.isdir(img_dir):
            parser.error(f'"img_dir" must be an existing directory: {img_dir}')

    if args.test_annot_fn:
        for test_annot_fn in args.test_annot_fn:
            if not os.path.isfile(test_annot_fn):
                parser.error(f'"test_annot_fn" must be an existing file: {test_annot_fn}')

    if args.val_annot_fn:
        for val_annot_fn in args.val_annot_fn:
            if not os.path.isfile(val_annot_fn):
                parser.error(f'"val_annot_fn" must be an existing file: {val_annot_fn}')

    if args.test_img_dir:
        for test_img_dir in args.test_img_dir:
            if not os.path.isdir(test_img_dir):
                parser.error(f'"test_img_dir" must be an existing directory: {test_img_dir}')

    if args.val_img_dir:
        for val_img_dir in args.val_img_dir:
            if not os.path.isdir(val_img_dir):
                parser.error(f'"val_img_dir" must be an existing directory: {val_img_dir}')

    if args.extra_train_annot_fn:
        for extra_train_annot_fn in args.extra_train_annot_fn:
            if not os.path.isfile(extra_train_annot_fn):
                parser.error(f'"extra_train_annot_fn" must be an existing file: {extra_train_annot_fn}')

    if args.extra_train_img_dir:
        for extra_train_img_dir in args.extra_train_img_dir:
            if not os.path.isdir(extra_train_img_dir):
                parser.error(f'"extra_train_img_dir" must be an existing directory: {extra_train_img_dir}')

    if args.run_predictions and args.run_predictions_all:
        parser.error('use just one option: "run_predictions" or "run_predictions_all"')

    if args.epochs is not None and args.max_iter is not None:
        parser.error('Set only one of "epochs" or "max_iter"')

    # if args.model != 'yolo8' and args.eval_period is not None and args.eval_period != 'epoch':
    #     parser.error('for model "yolo8", "eval_period" must be "epoch"')

    if args.eval_period is not None and args.eval_period.lower() != 'epoch':
        try:
            args.eval_period = int(args.eval_period)
        except ValueError:
            parser.error('"eval_period" must be "epoch" or a number (integer)')

    if args.min_max_img_size is not None and len(args.min_max_img_size) != 2:
        parser.error('"min_max_img_size" must be a tuple "min_size max_size"')

    if args.no_resizing:
        if args.model == 'dt2':
            if args.min_max_img_size is not None and args.min_max_img_size != [-1, -1]:
                parser.error('when "no_resizing" is set, "min_max_img_size" cannot be set')
            else:
                args.min_max_img_size = [-1, -1]
        else:
            if args.img_size is not None:
                parser.error('when "no_resizing" is set, "img_size" cannot be set')
            else:
                args.img_size = -1

    if args.cfg_list is not None and len(args.cfg_list) % 2 != 0:
        parser.error('"cfg_list" must be a list of pairs')

    main(args)
