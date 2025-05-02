import os
import shutil
import sys
from ultralytics import YOLO
import re
import copy
from types import SimpleNamespace
import itertools
import cv2
import torch
import numpy as np
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.utils.callbacks import tensorboard as yolo_tb
from torch.optim import lr_scheduler
import cv_tools as utils


def load_model(model_name=None, size='n', version='8', anchorfree='auto', load=True):
    if model_name is None:
        if anchorfree == 'auto':
            anchorfree = True
        assert (anchorfree and version in ['5', '8']) or (not anchorfree and version == '5')
        model_name = f'yolov{version}{size}{"u" if anchorfree and version == "5" else ""}.pt'
    else:
        if anchorfree == 'auto':
            anchorfree = not ('v5' in model_name and re.search(r"v8|v5.u", model_name.lower()) is None)
        if not (model_name.endswith('.pt') or model_name.endswith('.yaml')):
            model_name = f'{model_name}.pt'

    if os.path.exists(model_name):
        model_fn = model_name
    elif any([model_name == f'yolov8{x}.yaml' for x in ['n', 's', 'm', 'l', 'x']]):
        model_fn = model_name
    else:
        from config import YOLO_MODELS
        model_fn = os.path.join(YOLO_MODELS, model_name.lower())
        assert os.path.exists(model_fn)
    print(f'Loading model: {model_fn}')

    if anchorfree:
        model = YOLO(model_fn)
    else:
        import yolov5
        model = yolov5.load(model_fn)

    if load:
        return model
    else:
        return model_fn


def yolo_train(
        args, data_json_fns, data_img_dirs, epochs, test_json_fns=None, test_img_dirs=None, val_json_fns=None, val_img_dirs=None,
        base_model=None, n_folds=None, fold=None, test_size=None, test_groups=None, val_size=None, eval_period='epoch', sort_cats=False, merge_categories_as=None,
        multi_label=False, filename_groups=False, filename_groups_test_only=False, group_splitter='_',
        extra_train_json_fns=None, extra_train_img_dirs=None,
        drop_test_groups_in_extra_train=False, drop_val_groups_in_extra_train=False,
        extra_val_size=None, subsample_train=None, subsample_groups=None,
        imgs_per_batch=None, base_lr=None, final_lr=None, weight_decay=None, optimizer='SGD',
        threshold=None, NMS_threshold=None, detections_per_image=None, data_filters=None, img_transforms=None, coinflip_transform=False,
        img_size=None, vis_threshold=None, stratify_bbox_sizes=False, use_sharper_scheduler=False,
        target_decay_epoch=None, device=None, seed=None, run_predictions=True, prediction_models=None, save_pred_frames=True,
        only_create_data=False, do_not_save_data=False, hub_format=False, output_dir='output', skip_training=False, quick_debug=False,
        hsv_h=None, hsv_s=None, hsv_v=None, degrees=None, translate=None,
        scale=None, shear=None, perspective=None, flipud=None, fliplr=None,
        mosaic=None, close_mosaic=None, mixup=None, copy_paste=None, sort_images=False
):
    """
    imgsz=640

    albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01), CLAHE(p=0.01, clip_limit=(1, 4.0), tile_grid_size=(8, 8))

    YOLOv8 AUGMENTATIONS:

        hsv_h: 0.015
        hsv_s: 0.7
        hsv_v: 0.4
        degrees: 0.0
        translate: 0.1
        scale: 0.5
        shear: 0.0
        perspective: 0.0
        flipud: 0.0
        fliplr: 0.5
        mosaic: 1.0
        mixup: 0.0
        copy_paste: 0.0

        def v8_transforms(dataset, imgsz, hyp):
            pre_transform = Compose([
                Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic, border=[-imgsz // 2, -imgsz // 2]),
                CopyPaste(p=hyp.copy_paste),
                RandomPerspective(
                    degrees=hyp.degrees,
                    translate=hyp.translate,
                    scale=hyp.scale,
                    shear=hyp.shear,
                    perspective=hyp.perspective,
                    pre_transform=LetterBox(new_shape=(imgsz, imgsz)),
                ),])
            return Compose([
                pre_transform,
                MixUp(dataset, pre_transform=pre_transform, p=hyp.mixup),
                Albumentations(p=1.0),
                RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
                RandomFlip(direction='vertical', p=hyp.flipud),
                RandomFlip(direction='horizontal', p=hyp.fliplr),])  # transforms
    """

    if n_folds is None:
        n_folds = 1
    utils.assert_n_folds(n_folds)
    assert not use_sharper_scheduler or target_decay_epoch is not None
    assert isinstance(run_predictions, bool) or utils.is_iterable(run_predictions) or isinstance(run_predictions, str)
    if isinstance(run_predictions, str):
        run_predictions = [run_predictions]
    assert epochs == int(epochs)

    _model = load_model(base_model)
    assert _model.task in ['detect', 'segment']
    is_segmentation = _model.task == 'segment'
    del _model

    print('Preparing data')
    datasets, data_fn_prefix, image_root, cat_names, test_cat_names = utils.prepare_datasets(
        data_json_fns=data_json_fns, data_img_dirs=data_img_dirs, test_json_fns=test_json_fns, test_img_dirs=test_img_dirs,
        val_json_fns=val_json_fns, val_img_dirs=val_img_dirs, n_folds=n_folds, test_size=test_size, test_groups=test_groups, val_size=val_size,
        sort_cats=sort_cats, merge_categories_as=merge_categories_as, multi_label=multi_label, sort_images=sort_images,
        filename_groups=filename_groups, filename_groups_test_only=filename_groups_test_only, group_splitter=group_splitter,
        data_filters=data_filters, img_transforms=img_transforms, coinflip_transform=coinflip_transform,
        seed=seed, output_dir=output_dir, extra_train_json_fns=extra_train_json_fns,
        extra_train_img_dirs=extra_train_img_dirs, drop_test_groups_in_extra_train=drop_test_groups_in_extra_train,
        drop_val_groups_in_extra_train=drop_val_groups_in_extra_train,
        extra_val_size=extra_val_size, subsample_train=subsample_train, subsample_groups=subsample_groups,
        out_formats=['coco', f'yolo{"_hub" if hub_format else ""}'],
        copy_imgs=['auto', False], segm=is_segmentation,
        save=not do_not_save_data, stratify_bbox_sizes=stratify_bbox_sizes, quick_debug=quick_debug
    )

    if only_create_data:
        print('Data created, exiting...')
        return

    if run_predictions:
        run_predictions = run_predictions if utils.is_iterable(run_predictions) \
            else list(datasets.keys()) if n_folds == 1 else list(datasets[0].keys())

    device = utils.get_device(device, model='yolo8')
    print('Device:', device)

    for i in [None] if n_folds == 1 else range(n_folds) if fold is None else [fold]:
        if not skip_training and os.path.exists(os.path.join(output_dir, f'fold{i}')):
            continue

        data_fn = utils.get_data_name(
            prefix=data_fn_prefix, random_seed=seed, fold=i, ext='yaml')
        output_dir = output_dir.rstrip(os.path.sep)
        if n_folds == 1:
            project_dir = os.path.dirname(output_dir)
            run_dir = os.path.basename(output_dir)
        else:
            project_dir = output_dir
            run_dir = f'fold{i}'
            # to make sure that results are not overwritten
            os.makedirs(os.path.join(project_dir, run_dir), exist_ok=True)

        if not skip_training:
            assert os.path.exists(os.path.join(project_dir, run_dir)), os.path.join(project_dir, run_dir)

            kwargs = dict()
            if img_size is not None:
                kwargs['imgsz'] = img_size
            if base_lr is not None:
                kwargs['lr0'] = base_lr
            if final_lr is not None:
                kwargs['lrf'] = final_lr / base_lr
            if weight_decay is not None:
                kwargs['weight_decay'] = weight_decay
            if eval_period is None:
                kwargs['val'] = False
            else:
                assert eval_period.lower() == 'epoch'

            if hsv_h is not None:
                kwargs['hsv_h'] = hsv_h
            if hsv_s is not None:
                kwargs['hsv_s'] = hsv_s
            if hsv_v is not None:
                kwargs['hsv_v'] = hsv_v
            if degrees is not None:
                kwargs['degrees'] = degrees
            if translate is not None:
                kwargs['translate'] = translate
            if scale is not None:
                kwargs['scale'] = scale
            if shear is not None:
                kwargs['shear'] = shear
            if perspective is not None:
                kwargs['perspective'] = perspective
            if flipud is not None:
                kwargs['flipud'] = flipud
            if fliplr is not None:
                kwargs['fliplr'] = fliplr
            if mosaic is not None:
                kwargs['mosaic'] = mosaic
            if close_mosaic is not None:
                kwargs['close_mosaic'] = close_mosaic
            if mixup is not None:
                kwargs['mixup'] = mixup
            if copy_paste is not None:
                kwargs['copy_paste'] = copy_paste

            overrides = dict(
                data=data_fn, project=project_dir, name=run_dir, device=device, epochs=int(epochs),
                batch=imgs_per_batch,
                optimizer=optimizer, seed=seed, deterministic=True,
                save_crop=True, plots=True, save_conf=True, save_txt=True, save_json=True, exist_ok=True,
                **kwargs
            )

            from config import EXP_CFG_FN
            utils.experiment_config(
                filename=os.path.join(project_dir, run_dir, EXP_CFG_FN), categories=cat_names,
                data_json_fn=data_json_fns, data_img_dir=data_img_dirs,
                test_json_fn=test_json_fns, test_img_dir=test_json_fns, extra_train_json_fn=extra_train_json_fns,
                extra_train_img_dir=extra_train_img_dirs,
                n_folds=n_folds, fold=i, test_size=test_size, val_size=val_size, multi_label=multi_label,
                data_filters=data_filters, img_transforms=img_transforms,
                use_sharper_scheduler=use_sharper_scheduler, target_decay_epoch=target_decay_epoch,
                filename_groups=filename_groups, filename_groups_test_only=filename_groups_test_only
            )

            model = load_model(base_model, load=False)
            trainer = get_trainer(
                model=model, is_segmentation=is_segmentation, use_sharper_scheduler=use_sharper_scheduler,
                target_decay_epoch=target_decay_epoch, **overrides)
            sys.stdout.flush()
            trainer.train()
            shutil.rmtree(os.path.join(output_dir, f'fold{i}' if n_folds != 1 else '', 'labels'), ignore_errors=True)
            shutil.move(src=os.path.join(output_dir, f'fold{i}' if n_folds != 1 else '', 'args.yaml'), dst=os.path.join(output_dir, f'fold{i}' if n_folds != 1 else '', 'config.yaml'))
            print('Finished training')
        else:
            print('Skipping training')
            weights_dir = os.path.join(project_dir, run_dir, 'weights')
            trainer = SimpleNamespace(
                best=os.path.join(weights_dir, 'best.pt'),
                last=os.path.join(weights_dir, 'last.pt')
            )

        print(prediction_models)
        print(run_predictions)
        for model_name, model in [
            ('best', trainer.best),
            ('last', trainer.last)
        ]:
            print(model, os.path.exists(model))
            if prediction_models is None or model_name in prediction_models:
                if run_predictions and model is not None and os.path.exists(model):
                    for split in run_predictions:
                        pred_in_fn = utils.get_data_fn(prefix=data_fn_prefix, random_seed=seed, split=split if n_folds == 1 else f"fold{i}.{split}", fold=None)
                        print(pred_in_fn, os.path.exists(pred_in_fn))
                        print(pred_in_fn, os.path.exists(f'{pred_in_fn}.gz'))
                        if utils.file_or_gzip_exists(pred_in_fn):
                            pred_out_dir = os.path.join(project_dir, run_dir, f'{split}_predictions_{model_name}')
                            # safer not to remove stuff
                            # shutil.rmtree(pred_out_dir, ignore_errors=True)
                            print(f'INFERENCE with "{model_name}" on "{split}" dataset')
                            yolo_predict(YOLO(model), dataset=(pred_in_fn, image_root), output_dir=pred_out_dir,
                                         output_fn=os.path.join(
                                             pred_out_dir, f'{os.path.basename(pred_in_fn).replace("annotations", "predictions")}'),
                                         model_cat_names=cat_names, predict_cat_names=test_cat_names if split != 'train' else None,
                                         threshold=threshold, NMS_threshold=NMS_threshold, detections_per_image=detections_per_image,
                                         img_size=img_size, save_pred_frames=save_pred_frames, vis_threshold=vis_threshold,
                                         evaluate=True, warmup=False, stream=True, device=device, compress=True,
                                         iouType='segm' if is_segmentation else 'bbox', quick_debug=quick_debug,
                                         eval_log_info=f'EVALUATION with model "{model_name}" on "{split}" dataset')
                        else:
                            print(f'Cannot run "{model_name}" on "{split}" dataset, {pred_in_fn} does not exist')

        # cleanup YOLO
        if os.path.exists(data_fn):
            os.remove(data_fn)
        for split in set(run_predictions).union(['train']):
            shutil.rmtree(
                utils.get_data_name(prefix=data_fn_prefix, random_seed=seed,
                                    split=split if n_folds == 1 else f"fold{i}.{split}", fold=None),
                ignore_errors=True
            )
        utils.compress_file(os.path.join(project_dir, run_dir, 'predictions.json'), force=True)

    return data_fn_prefix


def yolo_predict(model, dataset, output_dir=None, output_fn=None, video_input=None, model_cat_names=None, predict_cat_names=None,
                 threshold=None, NMS_threshold=None, detections_per_image=None, img_size=None,
                 track=None, track_buffer=None, new_track_thr=None, track_match_thr=None, track_high_thr=None, track_low_thr=None,
                 save_pred_frames=False, vis_threshold=None, evaluate=False, warmup=False, every_n_frame=None,
                 stream=True, device=None, compress=True, iouType='bbox', eval_log_info=None,
                 tensor_RT=False, fast_processing=False, quick_debug=False):

    assert track in [None, True, False, utils.SORT, utils.BOT_SORT, utils.BYTE_TRACK, utils.DUMMY_TRACK]
    if track is True:
        track = utils.BOT_SORT
    assert stream, 'Without streaming tracking does not work with long videos'
    assert not evaluate or isinstance(dataset, tuple)
    assert not (model_cat_names is None and predict_cat_names is not None)

    if vis_threshold == 0:
        vis_threshold = None
    if save_pred_frames and vis_threshold is not None:
        from ultralytics.engine.results import Results

    assert video_input in [None, False, True]
    if every_n_frame is not None and every_n_frame != 1:
        assert isinstance(dataset, str) and ((os.path.isdir(dataset) and video_input) or utils.is_video(dataset))
    if isinstance(dataset, str) and os.path.isdir(dataset):
        vids = utils.read_videos_from_dir(dir_name=dataset, basename_only=False)
        if video_input:
            dataset = sorted(vids)
        else:
            imgs = utils.read_images_from_dir(dir_name=dataset, basename_only=False)
            if len(imgs) == 0 and len(vids) != 0:
                print(f'WARNING: Did not find any images in {dataset}, did you forget to specify "video_input=True"?')
            assert os.path.isdir(dataset)

    if isinstance(dataset, tuple):
        # This is implemented for pure predict and SORT tracking
        # It assumes COCO dataset, so it can be easily replaced by other detectors
        print('Dataset from COCO!')
        assert track not in [utils.BOT_SORT, utils.BYTE_TRACK]
        input_json_fn, img_dir = dataset
        from_coco = True

        dataset = utils.read_json(input_json_fn, only_imgs=True)
        predict_cat_names = utils.assure_consistent_cat_names(dataset, predict_cat_names=predict_cat_names)
        if predict_cat_names is None:
            predict_cat_names = model_cat_names
        dataset, img_ids = zip(*[
            (os.path.join(img_dir, img['file_name']), img['id']) for img in dataset['images']
        ])
        if quick_debug:
            dataset = dataset[:2]
            img_ids = img_ids[:2]
        assert track is None or list(dataset) == sorted(dataset)
    else:
        from_coco = False
        img_ids = None
        if not utils.is_iterable(dataset):
            dataset = [dataset]

    if predict_cat_names is None:
        predict_cat_names = model_cat_names

    if isinstance(model, str):
        if model.endswith('.engine'):
            tensor_RT = True
        model = YOLO(model)
    if not tensor_RT:
        model.fuse()
    os.makedirs(output_dir, exist_ok=True)

    if model_cat_names is not None and not tensor_RT:
        _yolo_cat_names = [name for name in model.names]
        assert list(range(len(model_cat_names))) == _yolo_cat_names or list(model_cat_names) == _yolo_cat_names, \
            (list(range(len(model_cat_names))), list(model_cat_names), _yolo_cat_names)

    if track is not None:
        if track == utils.SORT:
            from sort import sort
            assert new_track_thr is None or (new_track_thr >= 1 and new_track_thr == int(new_track_thr))
            track_cfg = {}
            if track_buffer is not None:
                track_cfg['max_age'] = track_buffer
            if new_track_thr is not None:
                track_cfg['min_hits'] = new_track_thr
            if track_match_thr is not None:
                track_cfg['iou_threshold'] = track_match_thr
            tracker = sort.Sort(**track_cfg)
        elif track != utils.DUMMY_TRACK:
            from config import YOLO_MODELS
            assert new_track_thr is None or new_track_thr <= 1
            track_cfg = utils.read_yaml(os.path.join(YOLO_MODELS, f'{track}.yaml'))

            if track_buffer is not None:
                track_cfg['track_buffer'] = track_buffer
            if new_track_thr is not None:
                track_cfg['new_track_thresh'] = new_track_thr
            if track_match_thr is not None:
                track_cfg['match_thresh'] = track_match_thr
            if track_high_thr is not None:
                track_cfg['track_high_thresh'] = track_high_thr
            if track_low_thr is not None:
                track_cfg['track_low_thresh'] = track_low_thr

            track_cfg_fn = os.path.join(output_dir, 'track_config.yaml')
            utils.save_yaml(track_cfg, track_cfg_fn)

    if model_cat_names is not None and model_cat_names != predict_cat_names:
        assert set(model_cat_names).intersection(predict_cat_names) != 0
        test_classes_idx = np.arange(len(model_cat_names), dtype=int)[[c in predict_cat_names for c in model_cat_names]]
        # 1-based mapping from model_cat_names to predict_cat_names
        remap_cat_names = {i + 1: predict_cat_names.index(c) + 1 for i, c in enumerate(model_cat_names) if c in predict_cat_names}
    else:
        test_classes_idx = None
        remap_cat_names = None

    kwargs = dict(
        project=output_dir, name='predictions',
        classes=test_classes_idx, stream=stream, device=utils.get_device(device, model='yolo8'),
        save_crop=False, save_txt=False, save_conf=False
    )
    if every_n_frame is not None and every_n_frame != 1:
        kwargs['vid_stride'] = every_n_frame
    if img_size is not None:
        kwargs['imgsz'] = img_size
    if threshold is not None:
        kwargs['conf'] = threshold
    if NMS_threshold is not None:
        kwargs['iou'] = NMS_threshold
    if detections_per_image is not None:
        kwargs['max_det'] = detections_per_image
    if warmup:
        utils.model_warm_up(model)
    yolo_pred_dir = os.path.join(output_dir, 'predictions')
    shutil.rmtree(yolo_pred_dir, ignore_errors=True)

    predictions = []
    just_videos = []
    images = None if from_coco else []
    # short_video_ids = all([isinstance(source, str) and utils.is_video(source) for source in dataset]) and \
    #                   len(dataset) == len(['.'.join(os.path.basename(source).split('.')[:-1]) for source in dataset])
    for i, source in enumerate(dataset):
        print('SOURCE', source)
        if isinstance(source, str) and utils.is_video(source):
            assert video_input is None or video_input is True, f'Found {source} but video_input is {video_input}'
            is_video = True
            print(source)
        else:
            is_video = False
        just_videos.append(is_video)
        kwargs['source'] = source

        if fast_processing:
            assert is_video
            assert not from_coco
            assert track not in [None, utils.SORT, utils.DUMMY_TRACK]
            assert not save_pred_frames
            kwargs['save'] = save_pred_frames
            kwargs['verbose'] = False
            vid_predictions = []
            n_digits_fn = 6

            def conv_xyxy_to_xywh(bbox):
                bbox[:, 2] -= bbox[:, 0]
                bbox[:, 3] -= bbox[:, 1]
                return bbox

            t0 = utils.start_timing()
            outputs = model.track(tracker=track_cfg_fn, **kwargs)
            for j, outp in enumerate(outputs):
                filename = f'{".".join(outp.path.split(".")[:-1])}.frame_{j:0{n_digits_fn}d}.jpg'
                instances = [
                    {
                        "image_id": filename,
                        "category_id": int(cls) + 1,
                        "bbox": box.tolist(),
                        "score": float(score),
                        "track_id": int(track_id) if not np.isnan(track_id) else None
                    } for box, score, cls, track_id in zip(
                        conv_xyxy_to_xywh(outp.boxes.xyxy.cpu().numpy()),
                        outp.boxes.conf.cpu().numpy(),
                        outp.boxes.cls.cpu().numpy(),
                        outp.boxes.id.cpu().numpy() if outp.boxes.id is not None else utils.nans((len(outp.boxes),))
                    )
                ]
                vid_predictions.append(dict(image_id=filename, instances=instances))
            elapsed_t = utils.elapsed_time(t0)
            print(f'ELAPSED TIME: {elapsed_t} s ({source})')
        else:
            if warmup and is_video:
                utils.model_warm_up(model)

            t0 = utils.start_timing()

            if track in [None, utils.SORT, utils.DUMMY_TRACK]:
                kwargs['save'] = save_pred_frames and (is_video or vis_threshold is None)
                outputs = model.predict(**kwargs)
            else:
                kwargs['save'] = save_pred_frames
                kwargs['verbose'] = False
                outputs = model.track(tracker=track_cfg_fn, **kwargs)

            if is_video:
                assert not from_coco
                vid_predictions = []
            for j, outp in enumerate(outputs):
                if track == utils.SORT:
                    boxes_xyxy = outp.boxes.xyxy.cpu().numpy()
                    track_ids = tracker.update(boxes_xyxy)
                    if len(track_ids) != len(boxes_xyxy):
                        print(f'WARNING: {len(track_ids)} track_ids and {len(boxes_xyxy)} boxes')
                    assert len(track_ids) <= len(boxes_xyxy)
                elif track != utils.DUMMY_TRACK:
                    track_ids = outp.boxes.id.clone().cpu().numpy() if outp.boxes.id is not None else None
                else:
                    track_ids = None
                sys.stdout.flush()

                img_id = img_ids[i] if from_coco else None
                    # f'{(".".join(os.path.basename(source).split(".")[:-1]) if short_video_ids else source) if is_video else ""}{"_" if is_video else ""}{(j + 1)}'
                filename = outp.path

                if is_video:
                    n_digits_fn = 6 if stream else max(6, int(np.ceil(np.log10(len(outputs)))))
                    filename = f'{".".join(filename.split(".")[:-1])}.frame_{j:0{n_digits_fn}d}.jpg'
                    vid_predictions.append({
                        "image_id": img_id if img_id is not None else filename,
                        "instances": utils.instances_to_coco_json(
                            outp, filename, track=track if track != utils.DUMMY_TRACK else None, track_ids=track_ids,
                            one_based_cats=True, model_cat_names=model_cat_names, remap_cat_names=remap_cat_names)
                    })
                else:
                    predictions.append({
                        "image_id": img_id if img_id is not None else filename,
                        "instances": utils.instances_to_coco_json(
                            outp, img_id if img_id is not None else filename, track=track if track != utils.DUMMY_TRACK else None, track_ids=track_ids,
                            one_based_cats=True, model_cat_names=model_cat_names, remap_cat_names=remap_cat_names)
                    })

                if images is not None:
                    images.append(dict(
                        id=img_id if img_id is not None else filename,
                        file_name=filename,
                    ))

                if save_pred_frames and not is_video:
                    if track == utils.SORT:
                        img = cv2.imread(filename)
                        for ann in predictions[-1]['instances']:
                            bbox = [ann['bbox'][0], ann['bbox'][1], ann['bbox'][2], ann['bbox'][3]]
                            label = f'ID:{ann["track_id"]} {predict_cat_names[ann["category_id"] - 1]}'
                            color = [0, 0, 0]
                            color[-ann["category_id"]] = 255
                            utils.draw_box(img=img, bbox=bbox, label=label, color=color)
                        cv2.imwrite(os.path.join(output_dir, os.path.basename(filename)), img)
                    elif vis_threshold is not None:
                        assert outp.masks is None and outp.probs is None
                        vis_mask = outp.boxes.conf > vis_threshold
                        if vis_mask.any():
                            subset_outp = Results(orig_img=cv2.imread(filename), path=outp.path, names=outp.names,
                                                  boxes=outp.boxes.data[torch.as_tensor(vis_mask)], masks=None, probs=None)
                            utils.draw_boxes_natively(model='yolo8', result=subset_outp,
                                                      filename=os.path.join(output_dir, os.path.basename(filename)))
                        else:
                            shutil.copyfile(src=filename,
                                            dst=os.path.join(output_dir, os.path.basename(filename)))
                    else:
                        # reusing YOLO plotting API (now instead I copy the files as it can be faster)
                        shutil.copyfile(src=os.path.join(yolo_pred_dir, os.path.basename(filename)),
                                        dst=os.path.join(output_dir, os.path.basename(filename)))

            elapsed_t = utils.elapsed_time(t0)
            print(f'ELAPSED TIME: {elapsed_t} s ({source})')
        sys.stdout.flush()

        if is_video:
            predictions.extend(copy.deepcopy(vid_predictions))
            pred_fn = os.path.join(output_dir, f"{os.path.basename(source[:-4] if source[-4] == '.' else source)}")
            utils.save_json(
                dict(annotations=list(itertools.chain(*[p["instances"] for p in vid_predictions]))),
                fn=f'{pred_fn}.json', only_preds=True, compress=compress
            )

        if save_pred_frames and is_video:
            video_fn = os.path.join(f'{yolo_pred_dir}{str(i + 1) if i != 0 else ""}', os.path.basename(source))
            if not os.path.exists(video_fn) and video_fn.endswith('.mp4') and os.path.exists(f'{video_fn[:-4]}.avi'):
                video_fn = f'{video_fn[:-4]}.avi'

            if os.path.exists(video_fn):
                shutil.copyfile(src=video_fn,
                                dst=os.path.join(output_dir, os.path.basename(video_fn)))
            else:
                print(f'WARNING: cannot move {video_fn}.')

    if output_fn is None:
        output_fn = os.path.join(output_dir, 'predictions.json')
    for i in range(len(dataset)):
        shutil.rmtree(f'{yolo_pred_dir}{str(i + 1) if i != 0 else ""}', ignore_errors=True)
    predictions = list(itertools.chain(*[p["instances"] for p in predictions]))
    # save always for simplicity and backwards comp.
    if True or len(just_videos) == 0 or not np.all(just_videos):
        utils.save_json(
            dict(annotations=predictions) if from_coco else dict(images=images, annotations=predictions),
            output_fn, assert_correct=from_coco, only_preds=True, compress=compress
        )

    if evaluate:
        assert from_coco
        if eval_log_info:
            print(eval_log_info)
        r = utils.evaluate(gt_coco=input_json_fn, dt_coco=output_fn, iouType=iouType, maxDets=detections_per_image,
                           areaRng=None, areaRngLbl=None, PR_curve=True, allow_zero_area_boxes=True,
                           fix_zero_ann_ids=True, verbose=True)
        print("Categories:", " ".join([str(c) for c in predict_cat_names]))
        print("AP50 for each category:", r.precision.mean(axis=0))
    print("Finished predictions")
    return output_fn, predictions


def tensors_to_results(results, threshold=None, check=True, device=None):
    rdicts = []
    for r in results:
        tensor = r.boxes.data
        if device is not None:
            tensor = tensor.to(device)
        boxes_xyxy = tensor[:, :4]
        confs = tensor[:, 4]
        classes = tensor[:, 5]
        if threshold is not None:
            thr_mask = confs >= threshold
            boxes_xyxy = boxes_xyxy[thr_mask]
            confs = confs[thr_mask]
            classes = classes[thr_mask]
        assert len(boxes_xyxy) == len(confs)
        assert len(boxes_xyxy) == len(classes)
        if check and threshold is None:
            assert boxes_xyxy.tolist() == r.boxes.xyxy.tolist()
            assert confs.tolist() == r.boxes.conf.tolist()
            assert classes.tolist() == r.boxes.cls.tolist()
        rdicts.append(dict(boxes_xyxy=boxes_xyxy, confs=confs, classes=classes))
    return rdicts


def log_learning_rate(trainer: BaseTrainer):
    yolo_tb._log_scalars(trainer.lr, step=trainer.epoch + 1)
    yolo_tb._log_scalars({f'sch_lr{i}': lr for i, lr in enumerate(trainer.scheduler.get_lr())}, step=trainer.epoch + 1)


def sharper_scheduler(trainer: BaseTrainer, target_decay_epoch=100):
    if not trainer.args.cos_lr:
        print(f'WARNING: Exchanging scheduler with target_decay_epoch={target_decay_epoch}')
        _le = trainer.scheduler.last_epoch
        trainer.lf = lambda x: max(0, (1 - x / target_decay_epoch) * (1.0 - trainer.args.lrf)) + trainer.args.lrf
        trainer.scheduler = lr_scheduler.LambdaLR(trainer.optimizer, lr_lambda=trainer.lf)
        trainer.scheduler.last_epoch = _le


def get_trainer(model, use_sharper_scheduler, target_decay_epoch=100, is_segmentation=False, **overrides):
    trainer_class = SegmentationTrainer if is_segmentation else DetectionTrainer
    trainer = trainer_class(overrides=dict(model=model, **overrides))
    if use_sharper_scheduler:
        def _sharper_scheduler(tr):
            return sharper_scheduler(trainer=tr, target_decay_epoch=target_decay_epoch)
        trainer.add_callback("on_pretrain_routine_end", _sharper_scheduler)
        # This does not work anymore...
        # trainer.add_callback("on_train_epoch_end", log_learning_rate)
    return trainer
