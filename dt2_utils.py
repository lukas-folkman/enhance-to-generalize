import copy
import os
import shutil
import sys
import itertools
import numpy as np
from types import SimpleNamespace
import cv2
import time
import torch
import torch.utils.data as torchdata
from torchvision.ops import nms
import traceback
import detectron2
import logging
from detectron2.utils import comm
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, DefaultPredictor, default_setup, launch
from detectron2.utils.events import EventStorage
from detectron2.engine.hooks import HookBase
from detectron2.data import DatasetMapper
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import Visualizer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, verify_results
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.data import build_detection_test_loader, get_detection_dataset_dicts
from detectron2.data.common import MapDataset
from detectron2.data.build import trivial_batch_collator
from detectron2.data.samplers import InferenceSampler
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.modeling.roi_heads.roi_heads import StandardROIHeads
from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Instances
from detectron2.modeling import ROI_HEADS_REGISTRY
from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads.box_head import build_box_head

from collections import defaultdict
import cv_tools as utils
from config import DT2_CONFIGS, SHORT_CFG_FN, FULL_CFG_FN, EXP_CFG_FN, LAST_WEIGHTS_FN, BEST_WEIGHTS_FN, BEST_WEIGHTS_FN2
from config import DT2_FASTER_RCNN_WEIGHTS, DT2_RETINA_WEIGHTS
from typing import Dict, List, Tuple, Union
from torch.nn import functional as F


@ROI_HEADS_REGISTRY.register()
class ClassSubsetROIHeads(StandardROIHeads):

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        # Below is the only modification: FastRCNNOutputLayers --> FastRCNNOutputClassSubsetLayers
        box_predictor = FastRCNNOutputClassSubsetLayers(cfg, box_head.output_shape)
        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }


class FastRCNNOutputClassSubsetLayers(FastRCNNOutputLayers):

    def __init__(self, cfg, input_shape):
        super(FastRCNNOutputClassSubsetLayers, self).__init__(cfg, input_shape)
        self.class_subset_mask = torch.Tensor(np.asarray(cfg.MODEL.ROI_HEADS.CLASS_SUBSET_MASK, dtype=bool))

    def predict_probs(
        self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of proposals for image i.
        """
        scores, _ = predictions
        num_inst_per_image = [len(p) for p in proposals]
        assert scores.shape[1] == self.class_subset_mask.shape[0], (scores.shape, self.class_subset_mask.shape)
        # this is clumsy:
        # scores[:, torch.logical_not(self.class_subset_mask)] = torch.min(scores, axis=1, keepdim=True).values
        probs = F.softmax(scores, dim=-1).split(num_inst_per_image, dim=0)
        # set probs of other classes to 0
        for p in probs:
            p[:, torch.logical_not(self.class_subset_mask)] = 0
        return probs


def get_trainer(cfg, eval_period=None, save_best_checkpoint=True, checkpoint_metric=('val_total_loss', False),
                early_stopping=True, early_stopping_wait=10, no_stopping=20):
    if eval_period is not None and eval_period != 0:
        trainer = TrainerWithEval(
            cfg=cfg, eval_period=eval_period, save_best_checkpoint=save_best_checkpoint,
            checkpoint_metric=checkpoint_metric, early_stopping=early_stopping, early_stopping_wait=early_stopping_wait,
            no_stopping=no_stopping
        )
    else:
        trainer = DefaultTrainer(cfg=cfg)
    return trainer


class COCOEvaluatorWithMissingClasses(COCOEvaluator):
    def _eval_predictions(self, predictions, img_ids=None):
        """
        Evaluate predictions. Fill self._results with the metrics of the tasks.
        """
        from detectron2.utils.file_io import PathManager
        import json

        self._logger.info("Preparing results for COCO format ...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))
        tasks = self._tasks or self._tasks_from_predictions(coco_results)

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            dataset_id_to_contiguous_id = self._metadata.thing_dataset_id_to_contiguous_id
            all_contiguous_ids = list(dataset_id_to_contiguous_id.values())
            num_classes = len(all_contiguous_ids)
            assert min(all_contiguous_ids) == 0 and max(all_contiguous_ids) == num_classes - 1

            reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()}
            # Subset to only evaluation classes:
            coco_results = [r for r in coco_results if r["category_id"] < num_classes]
            for result in coco_results:
                category_id = result["category_id"]
                assert category_id < num_classes, (
                    f"A prediction has class={category_id}, "
                    f"but the dataset only has {num_classes} classes and "
                    f"predicted class id should be in [0, {num_classes - 1}]."
                )
                result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info(
            "Evaluating predictions with {} COCO API...".format(
                "unofficial" if self._use_fast_impl else "official"
            )
        )
        for task in sorted(tasks):
            assert task in {"bbox", "segm", "keypoints"}, f"Got unknown task: {task}!"
            coco_eval = (
                detectron2.evaluation.coco_evaluation._evaluate_predictions_on_coco(
                    self._coco_api,
                    coco_results,
                    task,
                    kpt_oks_sigmas=self._kpt_oks_sigmas,
                    use_fast_impl=self._use_fast_impl,
                    img_ids=img_ids,
                    max_dets_per_image=self._max_dets_per_image,
                )
                if len(coco_results) > 0
                else None  # cocoapi does not handle empty results very well
            )

            res = self._derive_coco_results(
                coco_eval, task, class_names=self._metadata.get("thing_classes")
            )
            self._results[task] = res


class TrainerWithEval(DefaultTrainer):

    def __init__(self, cfg, eval_period, save_best_checkpoint=False, checkpoint_metric=('val_total_loss', False),
    early_stopping=False, early_stopping_wait=10, no_stopping=20):
        self.eval_period = eval_period
        self.save_best_checkpoint = save_best_checkpoint
        self.checkpoint_metric = checkpoint_metric
        self.early_stopping = early_stopping
        self.early_stopping_wait = early_stopping_wait
        self.no_stopping = no_stopping
        super(TrainerWithEval, self).__init__(cfg=cfg)

    def run_step(self):
        self._trainer.iter = self.iter
        if self.cfg.SOLVER.AMP.ENABLED:
            self._trainer.run_step()
        else:
            """
            taken from detectron2.engine.train_loop.SimpleTrainer.run_step():
            """
            assert self._trainer.model.training, "[SimpleTrainer] model was changed to eval mode!"
            start = time.perf_counter()
            """
            If you want to do something with the data, you can wrap the dataloader.
            """
            data = next(self._trainer._data_loader_iter)
            data_time = time.perf_counter() - start

            # IMPLEMENT VISUALISATION
            if self._trainer.iter < 3:
                import bbox_visualizer as bbv
                import seaborn as sns
                palette = (np.asarray(sns.color_palette('bright')) * 255).tolist()
                for i, d in enumerate(data):
                    img = np.moveaxis(d['image'].to('cpu').numpy(), 0, -1)
                    instances = d['instances'].to('cpu')
                    bboxes = instances.gt_boxes.tensor.numpy().astype(int)
                    labels = instances.gt_classes.numpy().astype(int)
                    fn = os.path.join(self.cfg.OUTPUT_DIR, f'train_batch{self._trainer.iter}_{i}.jpg')
                    if len(labels) != 0:
                        for label, color in zip(range(max(labels) + 1), palette):
                            mask = [ll == label for ll in labels]
                            img = bbv.draw_multiple_rectangles(
                                img=img, bboxes=bboxes[mask].tolist(), bbox_color=color,
                                thickness=3, is_opaque=False, alpha=0.5)
                            img = bbv.add_multiple_labels(
                                img=img, labels=labels[mask].astype(str).tolist(), bboxes=bboxes[mask].tolist(),
                                text_bg_color=color, text_color=(255, 255, 255), top=True)
                    utils.write_image(img, fn)

            if self._trainer.zero_grad_before_forward:
                """
                If you need to accumulate gradients or do something similar, you can
                wrap the optimizer with your custom `zero_grad()` method.
                """
                self._trainer.optimizer.zero_grad()

            """
            If you want to do something with the losses, you can wrap the model.
            """
            loss_dict = self._trainer.model(data)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())
            if not self._trainer.zero_grad_before_forward:
                """
                If you need to accumulate gradients or do something similar, you can
                wrap the optimizer with your custom `zero_grad()` method.
                """
                self._trainer.optimizer.zero_grad()
            losses.backward()

            self._trainer.after_backward()

            if self._trainer.async_write_metrics:
                # write metrics asynchronically
                self._trainer.concurrent_executor.submit(
                    self._trainer._write_metrics, loss_dict, data_time, iter=self._trainer.iter
                )
            else:
                self._trainer._write_metrics(loss_dict, data_time)

            """
            If you need gradient clipping/scaling or other processing, you can
            wrap the optimizer with your custom `step()` method. But it is
            suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
            """
            self._trainer.optimizer.step()

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        # This is automatically run every cfg.TEST.EVAL_PERIOD
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "validation")
        eval_class = COCOEvaluatorWithMissingClasses if cfg.TEST.EVAL_CLASS_SUBSET else COCOEvaluator
        return eval_class(dataset_name=dataset_name, tasks=('segm','bbox') if cfg.MODEL.MASK_ON else ('bbox',),
                          output_dir=output_folder, max_dets_per_image=cfg.TEST.DETECTIONS_PER_IMAGE)

    def build_hooks(self):
        # This is for validation loss
        hooks = super(TrainerWithEval, self).build_hooks()
        checkpoint_best_value_fn = os.path.join(self.cfg.OUTPUT_DIR, "model_best_metric.txt")
        hooks.insert(-1, LossEvalHook(
            eval_period=self.eval_period,
            model=self.model,
            data_loader=build_detection_test_loader(
                cfg=self.cfg,
                dataset_name=self.cfg.DATASETS.TEST[0],
                mapper=DatasetMapper(self.cfg, True)
            ),
            save_best_checkpoint=self.save_best_checkpoint,
            checkpoint_metric=self.checkpoint_metric[0],
            checkpoint_max_best=self.checkpoint_metric[1],
            resume_best_value=self.checkpointer.has_checkpoint() and os.path.exists(checkpoint_best_value_fn),
            checkpoint_best_value_fn=checkpoint_best_value_fn,
            early_stopping=self.early_stopping,
            early_stopping_wait=self.early_stopping_wait,
            no_stopping_iter=self.no_stopping * self.eval_period
        ))
        return hooks

    def train(self):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(self.start_iter))

        self.iter = self.start_iter

        with EventStorage(self.start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(self.start_iter, self.max_iter):
                    # from time import time, ctime
                    # print('before_step:', ctime(time()))
                    self.before_step()
                    # print('run_step:', ctime(time()))
                    self.run_step()
                    # print('after_step:', ctime(time()))
                    self.after_step()
                    # print('finished:', ctime(time()))

                    if self.iter == self.max_iter:
                        # someone changed iter to max_iter to implement early stopping
                        break

                # self.iter == self.max_iter can be used by `after_train` to
                # tell whether the training successfully finished or failed
                # due to exceptions.
                self.iter += 1
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

        if len(self.cfg.TEST.EXPECTED_RESULTS) and comm.is_main_process():
            assert hasattr(
                self, "_last_eval_results"
            ), "No evaluation results obtained during training!"
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results


class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader,
                 save_best_checkpoint, checkpoint_metric, checkpoint_max_best,
                 resume_best_value, checkpoint_best_value_fn,
                 early_stopping, early_stopping_wait, no_stopping_iter):
        self.eval_period = eval_period
        self.model = model
        self.data_loader = data_loader

        self.save_best_checkpoint = save_best_checkpoint
        self.checkpoint_metric = checkpoint_metric
        self.checkpoint_max_best = checkpoint_max_best
        self.checkpoint_best_value = None
        self.checkpoint_best_value_fn = checkpoint_best_value_fn
        self.checkpoint_metric2 = None
        self.checkpoint_max_best2 = True
        self.checkpoint_best_value2 = None
        self.checkpoint_best_value_fn2 = f'{self.checkpoint_best_value_fn}2'
        self.min_epochs = 20
        self.ready_to_stop = False
        if resume_best_value:
            with open(self.checkpoint_best_value_fn, 'r') as f:
                self.checkpoint_best_value = float(f.readlines()[-1].strip())
            print(f'INFO (val): checkpoint_best_value loaded from {self.checkpoint_best_value_fn}')
        print(f'INFO (val): checkpoint_best_value: {self.checkpoint_best_value}')

        self.early_stopping = early_stopping
        self.early_stopping_wait = early_stopping_wait
        self.no_stopping_iter = no_stopping_iter

    def after_step(self):
        EVAL_EVERY_ITER = False
        # eval at every step --> helps with smoothing
        # but is TERRIBLY slow
        if EVAL_EVERY_ITER:
            self._do_loss_eval()

        if (self.trainer.iter + 1 == self.trainer.max_iter) or \
                (self.eval_period > 0 and (self.trainer.iter + 1) % self.eval_period == 0):

            if not EVAL_EVERY_ITER:
                self._do_loss_eval()
            # current metric without smoothing:
            if self.checkpoint_metric in self.trainer.storage.latest():
                current_metric = self.trainer.storage.latest()[self.checkpoint_metric][0]
                direction = -1 if self.checkpoint_max_best else 1
                current_metric2 = self.trainer.storage.latest()[self.checkpoint_metric2][0] \
                    if self.checkpoint_metric2 else None
                direction2 = -1 if self.checkpoint_max_best2 else 1

                if self.save_best_checkpoint:
                    if self.checkpoint_best_value is None or \
                            current_metric * direction < self.checkpoint_best_value * direction:
                        print(f'INFO (val, iter {self.trainer.iter + 1}): This is the best model so far: '
                              f'{self.checkpoint_metric} of {current_metric} (previously {self.checkpoint_best_value})')
                        self.checkpoint_best_value = current_metric
                        self.trainer.checkpointer.save('model_best')
                        with open(self.checkpoint_best_value_fn, 'w') as f:
                            print(self.checkpoint_best_value, file=f)

                    if self.checkpoint_metric2 and (self.checkpoint_best_value2 is None or \
                            current_metric2 * direction2 < self.checkpoint_best_value2 * direction2):
                        print(f'INFO (val, iter {self.trainer.iter + 1}): This is the best model so far: '
                              f'{self.checkpoint_metric2} of {current_metric2} (previously {self.checkpoint_best_value2})')
                        self.checkpoint_best_value2 = current_metric2
                        self.trainer.checkpointer.save('model_best2')
                        with open(self.checkpoint_best_value_fn2, 'w') as f:
                            print(self.checkpoint_best_value2, file=f)

                    if self.early_stopping and self.trainer.iter + 1 >= self.no_stopping_iter:
                        history = [x[0] for x in self.trainer.storage.histories()[self.checkpoint_metric].values()[-self.early_stopping_wait:]]
                        margin = self.checkpoint_best_value * direction * 0.005
                        if np.all([(x - margin) * direction > self.checkpoint_best_value * direction for x in history]):
                            print(f'INFO (val, iter {self.trainer.iter + 1}): Stopping training: '
                            f'{self.checkpoint_metric} of {self.checkpoint_best_value} not improved for '
                            f'{self.early_stopping_wait} epochs (mean of {np.mean(history)})')
                            print(f'INFO (val, iter {self.trainer.iter + 1}): History: {history}')
                            # This will hack the end of training
                            self.trainer.checkpointer.save('model_final')
                            self.trainer.iter = self.trainer.max_iter

    def _do_loss_eval(self):
        losses = defaultdict(lambda: [])
        for i, inputs in enumerate(self.data_loader):
            metrics_dict = self._get_loss(inputs)
            for loss in metrics_dict:
                losses[loss].append(metrics_dict[loss])
        for loss in losses:
            self.trainer.storage.put_scalar(f'val_{loss}', np.mean(losses[loss]), smoothing_hint=False)
        return losses

    def _get_loss(self, data):
        # How loss is calculated on train_loop
        metrics_dict = self.model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v) for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        metrics_dict['total_loss'] = total_losses_reduced
        return metrics_dict


def read_dt2_config(fn, strict=True):
    cfg = detectron2.config.get_cfg()
    cfg.TEST['EVAL_CLASS_SUBSET'] = None
    cfg.MODEL.RESNETS['DROPOUT'] = None
    cfg.MODEL.ROI_BOX_HEAD['DROPOUT'] = None
    cfg.INPUT['COPY_PASTE'] = detectron2.config.config.CfgNode(
        {'ENABLED': False, 'TYPE': None, 'N_OBJECTS': None, 'MAX_IOA': None, 'JSON_FN': None, 'IMG_DIR': None})
    cfg.merge_from_file(fn)
    assert not strict or cfg.INPUT.FORMAT == utils.BGR_FORMAT, f'C.INPUT.FORMAT must be "BGR", not {cfg.INPUT.FORMAT}'
    return cfg


def make_dt2_config(cfg, short_cfg_fn=SHORT_CFG_FN, long_cfg_fn=FULL_CFG_FN, strict=True):
    full_cfg = detectron2.config.get_cfg()
    if comm.is_main_process():
        output_dir = cfg.get('OUTPUT_DIR', full_cfg.OUTPUT_DIR)
        os.makedirs(output_dir, exist_ok=True)
        utils.save_yacs(cfg, os.path.join(output_dir, short_cfg_fn))
    if 'NAME' in cfg.MODEL.ROI_HEADS and cfg.MODEL.ROI_HEADS.NAME == 'ClassSubsetROIHeads':
        full_cfg.MODEL.ROI_HEADS['CLASS_SUBSET_MASK'] = None
    full_cfg.TEST['EVAL_CLASS_SUBSET'] = None
    full_cfg.MODEL.RESNETS['DROPOUT'] = None
    full_cfg.MODEL.ROI_BOX_HEAD['DROPOUT'] = None
    full_cfg.INPUT['COPY_PASTE'] = detectron2.config.config.CfgNode(
        {'ENABLED': False, 'TYPE': None, 'N_OBJECTS': None, 'MAX_IOA': None, 'JSON_FN': None, 'IMG_DIR': None})
    full_cfg.merge_from_file(os.path.join(output_dir, short_cfg_fn))
    assert not strict or full_cfg.INPUT.FORMAT == utils.BGR_FORMAT, f'C.INPUT.FORMAT must be "BGR", not {cfg.INPUT.FORMAT}'
    if comm.is_main_process():
        utils.save_yacs(full_cfg, os.path.join(output_dir, long_cfg_fn), sort_keys=True)
    return full_cfg


def set_subset_class_ROI_head(cfg, test_class_subset_mask):
    cfg.MODEL.ROI_HEADS.NAME = 'ClassSubsetROIHeads'
    cfg.MODEL.ROI_HEADS['CLASS_SUBSET_MASK'] = list(test_class_subset_mask)
    cfg.MODEL.ROI_HEADS['CLASS_SUBSET_MASK'].append(True)  # background class


def dt2_config(
        base_model, train_data, num_classes, device, max_iter, weights=None,
        test_data=None, eval_period=None,
        rpn_batch_size_per_img=None, roi_batch_size_per_img=None,
        imgs_per_batch=None, weight_decay=None, base_lr=None, lr_gamma=None, gamma_steps=None,
        mixed_precision=None, filter_empty=None,
        min_max_img_size=None, checkpoint_period=None, threshold=None, NMS_threshold=None, detections_per_image=None,
        test_class_subset_mask=None, output_dir=None, seed=None, eval_class_subset=False, dropout_cnn=None, dropout_fc=None,
        cfg_list=None, strict=True, copy_paste=False
):

    from yacs.config import CfgNode as CN
    from detectron2 import model_zoo

    C = CN()
    assert base_model is not None
    C._BASE_ = os.path.join(DT2_CONFIGS, base_model)

    assert train_data is not None
    C.DATASETS = CN()
    C.DATASETS.TRAIN = (train_data,) if not utils.is_iterable(train_data) else train_data
    C.DATASETS.TEST = (test_data,) if not utils.is_iterable(test_data) else test_data

    assert device is not None
    C.MODEL = CN()
    C.MODEL.DEVICE = device

    if weights is not None:
        C.MODEL.WEIGHTS = weights
    else:
        C.MODEL.WEIGHTS = DT2_FASTER_RCNN_WEIGHTS if base_model == 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml' \
            else DT2_RETINA_WEIGHTS if base_model == 'COCO-Detection/retinanet_R_50_FPN_3x.yaml' else \
            model_zoo.get_checkpoint_url(base_model)

    C.MODEL.RESNETS = CN()
    C.MODEL.RESNETS['DROPOUT'] = dropout_cnn
    C.MODEL.ROI_BOX_HEAD = CN()
    C.MODEL.ROI_BOX_HEAD['DROPOUT'] = dropout_fc

    assert num_classes is not None
    C.MODEL.ROI_HEADS = CN()
    C.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    C.MODEL.RETINANET = CN()
    C.MODEL.RETINANET.NUM_CLASSES = num_classes
    if test_class_subset_mask is not None:
        set_subset_class_ROI_head(C, test_class_subset_mask)

    if threshold is not None:
        C.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        C.MODEL.RETINANET.SCORE_THRESH_TEST = threshold

    if NMS_threshold is not None:
        C.MODEL.ROI_HEADS.NMS_THRESH_TEST = NMS_threshold
        C.MODEL.RETINANET.NMS_THRESH_TEST = NMS_threshold

    C.TEST = CN()
    C.TEST.EVAL_PERIOD = eval_period if eval_period is not None else 0
    C.TEST['EVAL_CLASS_SUBSET'] = eval_class_subset
    if detections_per_image is not None:
        C.TEST.DETECTIONS_PER_IMAGE = detections_per_image

    if roi_batch_size_per_img is not None:
        C.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = roi_batch_size_per_img

    if rpn_batch_size_per_img is not None:
        C.MODEL.RPN = CN()
        C.MODEL.RPN.BATCH_SIZE_PER_IMAGE = rpn_batch_size_per_img

    assert max_iter is not None
    C.SOLVER = CN()
    C.SOLVER.MAX_ITER = max_iter
    C.SOLVER.CHECKPOINT_PERIOD = checkpoint_period if checkpoint_period is not None else (max_iter + 1)
    # C.SOLVER.LR_SCHEDULER_NAME = 'WarmupCosineLR'

    if imgs_per_batch is not None:
        C.SOLVER.IMS_PER_BATCH = imgs_per_batch
    if base_lr is not None:
        C.SOLVER.BASE_LR = base_lr
    if lr_gamma is not None:
        C.SOLVER.GAMMA = lr_gamma
    if gamma_steps is not None:
        assert utils.is_iterable(gamma_steps)
        C.SOLVER.STEPS = gamma_steps

    if mixed_precision is not None:
        C.SOLVER.AMP = CN({"ENABLED": mixed_precision})
    if weight_decay is not None:
        C.SOLVER.WEIGHT_DECAY = weight_decay

    if filter_empty is not None:
        C.DATALOADER = CN()
        C.DATALOADER.FILTER_EMPTY_ANNOTATIONS = filter_empty

    if min_max_img_size is not None or copy_paste:
        C.INPUT = CN()
        if min_max_img_size is not None:
            C.INPUT.MIN_SIZE_TEST = min_max_img_size[0] if min_max_img_size[0] > 0 else 0
            C.INPUT.MAX_SIZE_TEST = min_max_img_size[1] if min_max_img_size[1] > 0 else int(1e4)
            C.INPUT.MIN_SIZE_TRAIN = (
                min_max_img_size[0] - 64,
                min_max_img_size[0] - 32,
                min_max_img_size[0],
                min_max_img_size[0] + 32,
                min_max_img_size[0] + 64
            ) if min_max_img_size[0] > 0 else (0,)
            C.INPUT.MAX_SIZE_TRAIN = min_max_img_size[1] if min_max_img_size[1] > 0 else int(1e4)
        if copy_paste:
            C.INPUT.COPY_PASTE = CN()
            C.INPUT.COPY_PASTE.ENABLED = copy_paste
            C.INPUT.COPY_PASTE.TYPE = 'closed_balance'
            C.INPUT.COPY_PASTE.MAX_IOA = 0.3
            C.INPUT.COPY_PASTE.N_OBJECTS = 10
            for path in [
                os.path.join(os.getenv('HOME'), 'lscratch', 'jellies'),
                os.path.join(os.getenv('HOME'), 'scratch', 'jellies'),
                os.path.join(os.getenv('HOME'), 'jellies')
            ]:
                if os.path.exists(path):
                    break
            JELLIES_DIR = path
            print(f'CPA: using {JELLIES_DIR}')
            WHICH_TRAIN = os.getenv('WHICH_TRAIN')
            C.INPUT.COPY_PASTE.JSON_FN = os.path.join(JELLIES_DIR, 'data', 'annot', 'fish_welfare', 'after_QC', f'after_QC.merged.spuriousCleaned.LFQC.segments.QC.train{WHICH_TRAIN}.json')
            C.INPUT.COPY_PASTE.IMG_DIR = os.path.join(JELLIES_DIR, 'data', 'frames', 'fish_welfare', 'Tassal_1280_blurred.LFQC')

    if output_dir is not None:
        C.OUTPUT_DIR = output_dir
    if seed is not None:
        C.SEED = seed

    if cfg_list:
        print('\n', cfg_list, '\n')
        assert len(cfg_list) % 2 == 0, "cfg_list must be a list of pairs"
        # Insert new keys -- not supported by API
        for full_key in cfg_list[0::2]:
            key_list = full_key.split(".")
            d = C
            for subkey in key_list[:-1]:
                if subkey not in d:
                    d[subkey] = CN()
                d = d[subkey]
            subkey = key_list[-1]
            if subkey not in d:
                d[subkey] = None
        # Insert new values properly with API
        C.merge_from_list(cfg_list)

    cfg = make_dt2_config(C, strict=strict)
    if comm.is_main_process():
        assert os.path.exists(cfg.OUTPUT_DIR)

    return cfg


def numpy_to_dt2_input(image):
    return dict(image=torch.from_numpy(image.transpose(2, 0, 1)))


def dt2_predict(cfg, weights_fn, dataset, output_dir, output_fn=None, video_input=None, model_cat_names=None, predict_cat_names=None,
                threshold=None, NMS_threshold=None, detections_per_image=None, min_max_img_size=None, imgs_per_batch=1,
                track=None, track_buffer=None, new_track_thr=None, track_match_thr=None, track_high_thr=None, track_low_thr=None,
                save_pred_frames=False, vis_threshold=None, evaluate=False, device=None, warm_up=False, save_config=False,
                compress=True, eval_log_info=None, one_based_video_frames=False, every_n_frame=None, quick_debug=False, strict=True):
    assert every_n_frame is None or every_n_frame == 1, 'Not implemented for dt2'
    assert track in [None, True, False, utils.BOT_SORT]
    if track is True:
        track = utils.BOT_SORT

    if imgs_per_batch is None:
        imgs_per_batch = 1
    assert imgs_per_batch is not None and imgs_per_batch > 0
    assert not (model_cat_names is None and predict_cat_names is not None)
    if comm.is_main_process():
        os.makedirs(output_dir, exist_ok=True)

    assert video_input in [None, False, True]
    if isinstance(dataset, str) and os.path.isdir(dataset):
        vids = utils.read_videos_from_dir(dir_name=dataset, basename_only=False, extensions='mp4')
        if video_input:
            dataset = sorted(vids)
        else:
            imgs = utils.read_images_from_dir(dir_name=dataset, basename_only=False)
            if len(imgs) == 0 and len(vids) != 0:
                print(f'WARNING: Did not find any images in {dataset}, did you forget to specify "video_input=True"?')
            dataset = sorted(imgs)

    if utils.is_iterable(dataset):
        if isinstance(dataset, tuple) and len(dataset) == 2:
            dataset_name = 'predictions'
            json_file, img_dir = dataset
            dataset = utils.read_json(json_file, only_imgs=True)
            predict_cat_names = utils.assure_consistent_cat_names(dataset, predict_cat_names=predict_cat_names)
            if predict_cat_names is None:
                predict_cat_names = model_cat_names
            register_coco_instances(
                name=dataset_name, json_file=json_file, image_root=img_dir,
                metadata=dict(thing_classes=predict_cat_names if predict_cat_names is not None else model_cat_names)
            )
        else:
            dataset_name = None
            if all([utils.is_image(x) for x in dataset]):
                assert video_input is None or video_input is False, f'Found images but video_input is {video_input}'
                video_input = False
            elif all([utils.is_video(x) for x in dataset]):
                if not all([utils.is_video(x, extensions='mp4') for x in dataset]):
                    raise ValueError(f'Only mp4 videos are supported: {dataset}')
                assert video_input is None or video_input is True, f'Found videos but video_input is {video_input}'
                video_input = True
            else:
                raise ValueError(f'Incorrect inputs (do not mix images and videos): {dataset}')
    else:
        assert isinstance(dataset, str), dataset
        dataset_name = dataset

    if predict_cat_names is None:
        predict_cat_names = model_cat_names

    if isinstance(cfg, str):
        cfg = read_dt2_config(cfg, strict=strict)
    else:
        cfg = cfg.clone()
        cfg.defrost()

    cfg.MODEL.WEIGHTS = weights_fn
    if dataset_name is not None:
        cfg.DATASETS.TEST = (dataset_name,)
    device = utils.get_device(device, model='dt2')
    cfg.MODEL.DEVICE = device
    if threshold is not None:
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = threshold
    if NMS_threshold is not None:
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = NMS_threshold
        cfg.MODEL.RETINANET.NMS_THRESH_TEST = NMS_threshold
    if detections_per_image is not None:
        cfg.TEST.DETECTIONS_PER_IMAGE = detections_per_image
    if min_max_img_size is not None:
        cfg.INPUT.MIN_SIZE_TEST = min_max_img_size[0] if min_max_img_size[0] > 0 else 0
        cfg.INPUT.MAX_SIZE_TEST = min_max_img_size[1] if min_max_img_size[1] > 0 else int(1e4)

    if model_cat_names is not None and model_cat_names != predict_cat_names:
        assert set(model_cat_names).intersection(predict_cat_names) != 0
        test_class_subset_mask = [c in predict_cat_names for c in model_cat_names]
        # 1-based mapping from model_cat_names to predict_cat_names
        remap_cat_names = {i + 1: predict_cat_names.index(c) + 1 for i, c in enumerate(model_cat_names) if c in predict_cat_names}
        set_subset_class_ROI_head(cfg, test_class_subset_mask)
    else:
        remap_cat_names = None
    cfg.freeze()
    if save_config:
        utils.save_yacs(cfg, os.path.join(output_dir, FULL_CFG_FN))

    model = build_model(cfg)
    DetectionCheckpointer(model).load(weights_fn)
    model.eval()

    if warm_up:
        utils.model_warm_up(model, data_mapper=lambda image: [numpy_to_dt2_input(image)])

    predictions = []
    predictions_without_nms = []
    # dataset: DT2 dataset from a DT2 database
    if dataset_name is not None:
        data_loader = build_batchable_detection_test_loader(
            cfg=cfg, dataset_name=dataset_name, batch_size=imgs_per_batch)

        with torch.no_grad():
            for i, inputs in enumerate(data_loader):
                outputs = model(inputs)
                for inp, outp in zip(inputs, outputs):
                    predictions.append({
                        "image_id": inp["image_id"],
                        "instances": instances_to_coco_json(outp["instances"].to("cpu"), img_id=inp["image_id"])
                    })
                    if save_pred_frames and comm.is_main_process():
                        visualize_predictions(
                            img=cv2.imread(inp["file_name"]), det_per_img=predictions[-1]['instances'],
                            cat_names=MetadataCatalog.get(dataset_name).get('thing_classes', None),
                            vis_threshold=vis_threshold,
                            output_fn=os.path.join(output_dir, os.path.basename(inp["file_name"])),
                            one_based=False, save=True
                        )
                if quick_debug:
                    break

    # dataset: list of images or list of videos
    else:
        if output_fn is None:
            output_fn = os.path.join(output_dir, 'predictions.json')
        assert not evaluate

        if track:
            from botsort.tracker.mc_bot_sort import BoTSORT
            from sort import sort

        def get_fresh_predictor():
            if track:
                track_cfg = utils.get_track_config(
                    track_high_thr=track_high_thr, track_low_thr=track_low_thr, new_track_thr=new_track_thr,
                    track_buffer=track_buffer, track_match_thr=track_match_thr)
                tracker = BoTSORT(args=track_cfg)
            else:
                tracker = None

            predictor = DefaultPredictor(cfg)
            assert predictor.input_format == utils.BGR_FORMAT
            return predictor, tracker

        if not video_input:
            predictor, tracker = get_fresh_predictor()
            # os.makedirs(os.path.join(output_dir, 'without_nms'), exist_ok=True)
            for inp_fn in dataset:
                # use PIL, to be consistent with evaluation
                img = utils.read_image(inp_fn, format=utils.BGR_FORMAT)
                outp = predictor(img)

                # det_per_img_without_nms = instances_to_coco_json(outp["instances"].to("cpu"), img_id=inp_fn)
                # predictions_without_nms.append({"image_id": inp_fn, "instances": det_per_img_without_nms})
                # if save_pred_frames and comm.is_main_process():
                #     visualize_predictions(
                #         img=img, det_per_img=det_per_img_without_nms,
                #         cat_names=model_cat_names, vis_threshold=vis_threshold,
                #         output_fn=os.path.join(output_dir, 'without_nms', os.path.basename(inp_fn)),
                #         one_based=False, save=True
                #     )

                if track:
                    # class agnostic NMS
                    idx = nms(outp["instances"].pred_boxes.tensor, outp["instances"].scores,
                              iou_threshold=cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST)
                else:
                    idx = np.arange(len(outp["instances"]), dtype=int)

                det_per_img = instances_to_coco_json(outp["instances"][idx].to("cpu"), img_id=inp_fn)
                if track:
                    try:
                        utils.update_tracker_with_detection(
                            tracker=tracker, det_per_img=det_per_img, img=img, iou_func=sort.iou_batch)
                        det_per_img = [d for d in det_per_img if 'track_id' in d]
                    except:
                        print(f'Tracking failed for {inp_fn}')
                        traceback.print_exc()
                predictions.append({"image_id": inp_fn, "instances": det_per_img})
                if save_pred_frames and comm.is_main_process():
                    visualize_predictions(
                        img=img, det_per_img=det_per_img,
                        cat_names=model_cat_names, vis_threshold=vis_threshold,
                        output_fn=os.path.join(output_dir, os.path.basename(inp_fn)), one_based=False, save=True
                    )
        else:
            # video_input
            for inp_fn in dataset:
                predictor, tracker = get_fresh_predictor()
                print(inp_fn)
                vid_predictions = []
                assert os.path.isfile(inp_fn)
                input_video = cv2.VideoCapture(inp_fn)
                width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frames_per_second = input_video.get(cv2.CAP_PROP_FPS)
                # num_frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
                video_out_fn = os.path.join(output_dir, os.path.basename(inp_fn))
                if save_pred_frames:
                    CODEC = 'mp4v'
                    assert video_out_fn.endswith('.mp4')
                    if os.path.exists(video_out_fn):
                        os.remove(video_out_fn)
                    output_video = cv2.VideoWriter(
                        filename=video_out_fn,
                        fourcc=cv2.VideoWriter_fourcc(*CODEC),
                        fps=float(frames_per_second),
                        frameSize=(width, height),
                        isColor=True,
                    )

                for f, frame in enumerate(utils.frame_from_video(input_video)):
                    frame_fn = f"{inp_fn[:-4]}.frame_{f + (1 if one_based_video_frames else 0):06d}.jpg"
                    outp = predictor(frame)
                    if track:
                        # class agnostic NMS
                        idx = nms(outp["instances"].pred_boxes.tensor, outp["instances"].scores,
                                  iou_threshold=cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST)
                    else:
                        idx = np.arange(len(outp["instances"]), dtype=int)
                    det_per_img = instances_to_coco_json(outp["instances"][idx].to("cpu"), img_id=frame_fn)
                    if track:
                        try:
                            utils.update_tracker_with_detection(
                                tracker=tracker, det_per_img=det_per_img, img=frame, iou_func=sort.iou_batch)
                            det_per_img = [d for d in det_per_img if 'track_id' in d]
                        except:
                            print(f'Tracking failed for frame {f}')
                            traceback.print_exc()
                    vid_predictions.append({"image_id": frame_fn, "instances": det_per_img})
                    if save_pred_frames and comm.is_main_process():
                        visualize_predictions(
                            img=frame, det_per_img=det_per_img,
                            cat_names=model_cat_names, vis_threshold=vis_threshold,
                            output_video=output_video, one_based=False, save=True
                        )
                input_video.release()
                predictions.extend(copy.deepcopy(vid_predictions))
                pred_fn = os.path.join(os.path.dirname(output_fn), f"{os.path.basename(inp_fn[:-4] if inp_fn[-4] == '.' else inp_fn)}")
                save_predictions_as_json(
                    predictions=vid_predictions, output_fn=f'{pred_fn}.json', model_cat_names=model_cat_names,
                    remap_cat_names=remap_cat_names, compress=compress
                )

                if save_pred_frames:
                    output_video.release()

    # save always for simplicity and backwards comp.
    if (True or dataset_name is not None or not video_input) and comm.is_main_process():
        save_predictions_as_json(predictions=predictions, output_fn=output_fn,
                                 model_cat_names=model_cat_names, remap_cat_names=remap_cat_names, compress=compress)

        if len(predictions_without_nms) != 0:
            save_predictions_as_json(predictions=predictions_without_nms, output_fn=output_fn.replace('.json', '.without_nms.json'),
                                     model_cat_names=model_cat_names, remap_cat_names=remap_cat_names,
                                     compress=compress)

    if evaluate and dataset_name is not None:
        if eval_log_info:
            print(eval_log_info)
        r = utils.evaluate(gt_coco=MetadataCatalog.get(dataset_name).json_file, dt_coco=output_fn,
                           iouType='segm' if cfg.MODEL.MASK_ON else 'bbox', maxDets=detections_per_image,
                           areaRng=None, areaRngLbl=None, PR_curve=True, allow_zero_area_boxes=True,
                           fix_zero_ann_ids=True, verbose=True)
        print("Categories:", " ".join([str(c) for c in predict_cat_names]))
        print("AP50 for each category:", r.precision.mean(axis=0))

    print("Finished predictions")
    return output_fn, predictions


def save_predictions_as_json(predictions, output_fn, model_cat_names=None, remap_cat_names=None, compress=False):
    predictions = list(itertools.chain(*[p["instances"] for p in predictions]))
    # detectron2 is 0-based but COCO-format is 1-based
    for pred in predictions:
        cls = pred['category_id'] + 1
        assert model_cat_names is None or cls in range(1, len(model_cat_names) + 1)
        # remap to predict_cat_names
        if remap_cat_names is not None:
            cls = remap_cat_names[cls]
        pred['category_id'] = cls
    utils.save_json(dict(annotations=predictions), output_fn, only_preds=True, compress=compress)


def visualize_predictions(
    img, det_per_img, cat_names=None, vis_threshold=None, output_fn=None, output_video=None, one_based=False, save=False
):
    if vis_threshold:
        det_per_img = [d for d in det_per_img if d['score'] > vis_threshold]
    if len(det_per_img) != 0:
        vis_img = utils.plot_bboxes(
            img, det_per_img, cat_dict={cid: cat for cid, cat in enumerate(cat_names, 1 if one_based else 0)})
    else:
        vis_img = img
    if save:
        assert sum([output_fn is None, output_video is None]) == 1
        if output_fn is not None:
            assert isinstance(output_fn, str)
            utils.write_image(vis_img, output_fn)
        if output_video is not None:
            output_video.write(vis_img)
    return vis_img


def visualize_image(instances, img, input_fn, cat_names=None, vis_threshold=None, output_dir=None, save=False):
    if img is None:
        img = cv2.imread(input_fn)
    visualizer = Visualizer(img_rgb=img[:, :, ::-1], metadata=dict(thing_classes=cat_names))
    if vis_threshold is not None:
        instances = instances[instances.scores > vis_threshold]
    vis_img = visualizer.draw_instance_predictions(instances)
    if save:
        vis_img.save(os.path.join(output_dir, os.path.basename(input_fn)))
    return vis_img


def visualize_video(instances, frame, cat_names=None, vis_threshold=None, output_video=None, save=False):
    video_visualizer = VideoVisualizer(metadata=dict(thing_classes=cat_names))
    if vis_threshold is not None:
        instances = instances[instances.scores > vis_threshold]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    vis_frame = video_visualizer.draw_instance_predictions(frame, instances)
    vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
    if save:
        output_video.write(vis_frame)
    return vis_frame


def build_batchable_detection_test_loader(cfg, dataset_name, batch_size=None, num_workers=None):

    if batch_size is None or batch_size == 1:
        # the default Detectron2 approach
        data_loader = build_detection_test_loader(
            cfg=cfg, dataset_name=dataset_name)
    else:
        # replica of the above but allowing batch_size > 1
        if isinstance(dataset_name, str):
            dataset_name = [dataset_name]

        dataset = get_detection_dataset_dicts(
            names=dataset_name,
            filter_empty=False,
            proposal_files=[
                cfg.DATASETS.PROPOSAL_FILES_TEST[list(cfg.DATASETS.TEST).index(x)] for x in dataset_name
            ] if cfg.MODEL.LOAD_PROPOSALS else None
        )
        dataset = MapDataset(dataset=dataset, map_func=DatasetMapper(cfg, False))

        data_loader = torchdata.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=None if isinstance(dataset, torchdata.IterableDataset) else InferenceSampler(len(dataset)),
            num_workers=cfg.DATALOADER.NUM_WORKERS if num_workers is None else num_workers,
            collate_fn=trivial_batch_collator,
        )

    return data_loader


def prepare_and_register_datasets(
        data_json_fns, data_img_dirs, test_json_fns, test_img_dirs, val_json_fns, val_img_dirs,
        n_folds, test_size, test_groups, val_size, sort_cats, multi_label, filename_groups, filename_groups_test_only, group_splitter,
        data_filters, img_transforms, coinflip_transform, seed, do_not_save_data, output_dir, sort_images,
        drop_test_groups_in_extra_train, drop_val_groups_in_extra_train,
        extra_train_json_fns, extra_train_img_dirs, extra_val_size, subsample_train, subsample_groups,
        stratify_bbox_sizes, merge_categories_as=None, quick_debug=False):

    utils.assert_n_folds(n_folds)
    datasets, data_fn_prefix, image_root, cat_names, test_cat_names = utils.prepare_datasets(
        data_json_fns=data_json_fns, data_img_dirs=data_img_dirs, test_json_fns=test_json_fns, test_img_dirs=test_img_dirs,
        val_json_fns=val_json_fns, val_img_dirs=val_img_dirs,
        n_folds=n_folds, test_size=test_size, test_groups=test_groups, val_size=val_size, sort_cats=sort_cats, merge_categories_as=merge_categories_as, multi_label=multi_label,
        filename_groups=filename_groups, filename_groups_test_only=filename_groups_test_only, group_splitter=group_splitter,
        data_filters=data_filters, img_transforms=img_transforms, coinflip_transform=coinflip_transform,
        seed=seed, output_dir=output_dir,
        drop_test_groups_in_extra_train=drop_test_groups_in_extra_train,
        drop_val_groups_in_extra_train=drop_val_groups_in_extra_train,
        extra_train_json_fns=extra_train_json_fns, extra_train_img_dirs=extra_train_img_dirs,
        extra_val_size=extra_val_size, subsample_train=subsample_train, subsample_groups=subsample_groups, out_formats='coco', save=comm.is_main_process() and not do_not_save_data, quick_debug=quick_debug,
        stratify_bbox_sizes=stratify_bbox_sizes, sort_images=sort_images
    )
    # All process need to wait until the data files were created by the main process
    # For some reason, it gets stuck here :-(
    # comm.synchronize()

    # Each process has to register their own dataset
    data_names = {}
    for fold in [None] if n_folds == 1 else range(n_folds):
        for split in datasets if fold is None else datasets[fold]:
            metadata = dict(thing_classes=test_cat_names) if split == 'test' else dict(thing_classes=cat_names)
            fn = utils.get_data_fn(prefix=data_fn_prefix, random_seed=seed, split=split, fold=fold)
            print('Registering', fn, image_root, metadata)
            register_coco_instances(name=fn, json_file=fn, metadata=metadata, image_root=image_root)
            data_names[utils.get_data_id(split=split, fold=fold)[1:]] = fn

    return datasets, data_fn_prefix, image_root, cat_names, test_cat_names, data_names


def calculate_epoch_length(datasets, imgs_per_batch, n_gpus=1):
    assert n_gpus == 1
    if not utils.is_iterable(datasets, dict_allowed=False):
        datasets = [datasets]
    n_iters_per_epoch = np.mean([len(d['train']['images']) for d in datasets]) / imgs_per_batch
    return n_iters_per_epoch


def dt2_train(
        args, data_json_fns, data_img_dirs, epochs, max_iter=None, test_json_fns=None, test_img_dirs=None, val_json_fns=None, val_img_dirs=None,
        base_model=None, weights=None, n_folds=None, fold=None, test_size=None, test_groups=None, val_size=None, sort_cats=False, merge_categories_as=None,
        multi_label=False, filename_groups=False, filename_groups_test_only=False, group_splitter='_',
        extra_train_json_fns=None, extra_train_img_dirs=None, extra_val_size=None, subsample_train=None, subsample_groups=None,
        drop_test_groups_in_extra_train=False, drop_val_groups_in_extra_train=False, retina=False,
        imgs_per_batch=None, rpn_batch_size_per_img=None, roi_batch_size_per_img=None,
        base_lr=None, weight_decay=None, lr_gamma=None, gamma_epochs=None, threshold=None, NMS_threshold=None, detections_per_image=None, dropout_cnn=None, dropout_fc=None,
        data_filters=None, img_transforms=None, coinflip_transform=False, min_max_img_size=None,
        eval_period=None, checkpoint_period=None, checkpoint_metric=None, vis_threshold=None, stratify_bbox_sizes=False, cfg_list=None,
        device=None, seed=None, run_predictions=True, prediction_models=None, save_pred_frames=True, do_not_save_data=False, output_dir='output', skip_training=False, quick_debug=False,
        save_best_checkpoint=False, early_stopping=False, early_stopping_wait=10, early_no_stopping_period=20,
        resume=False, sort_images=False, strict=True, copy_paste=False
):
    """
    Detectron2 AUGMENTATIONS:
    MAX_SIZE_TEST: 1333
    MIN_SIZE_TEST: 800
    ResizeShortestEdge(short_edge_length=(640, 672, 704, 736, 768, 800), max_size=1333, sample_style='choice')
    RandomFlip(prob=0.5, horizontal=True, vertical=False))
    """
    if run_predictions and save_pred_frames:
        # fail early import
        import bbox_visualizer
    if n_folds is None:
        n_folds = 1
    utils.assert_n_folds(n_folds)
    assert epochs is None or max_iter is None
    assert isinstance(run_predictions, bool) or utils.is_iterable(run_predictions) or isinstance(run_predictions, str)
    if isinstance(run_predictions, str):
        run_predictions = [run_predictions]
    assert cfg_list is None or len(cfg_list) % 2 == 0, "cfg_list must be a list of pairs"
    assert min_max_img_size is None or len(min_max_img_size) == 2
    assert eval_period is None or (isinstance(eval_period, str) and eval_period.lower() == 'epoch') \
           or eval_period == int(eval_period)
    assert not save_best_checkpoint or (eval_period is not None and eval_period != 0)

    if base_model is None:
        base_model = 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml' if not retina else 'COCO-Detection/retinanet_R_50_FPN_3x.yaml'

    use_test_as_val = (val_size is None or val_size == 0) and not val_json_fns
    datasets, data_fn_prefix, image_root, cat_names, test_cat_names, data_names = prepare_and_register_datasets(
        data_json_fns=data_json_fns, data_img_dirs=data_img_dirs, test_json_fns=test_json_fns, test_img_dirs=test_img_dirs,
        val_json_fns=val_json_fns, val_img_dirs=val_img_dirs, n_folds=n_folds, test_size=test_size, test_groups=test_groups, val_size=val_size, sort_cats=sort_cats, merge_categories_as=merge_categories_as,
        multi_label=multi_label, filename_groups=filename_groups, filename_groups_test_only=filename_groups_test_only, group_splitter=group_splitter,
        data_filters=data_filters, img_transforms=img_transforms, coinflip_transform=coinflip_transform, seed=seed, output_dir=output_dir, sort_images=sort_images,
        drop_test_groups_in_extra_train=drop_test_groups_in_extra_train,
        drop_val_groups_in_extra_train=drop_val_groups_in_extra_train,
        extra_train_json_fns=extra_train_json_fns, extra_train_img_dirs=extra_train_img_dirs,
        extra_val_size=extra_val_size, subsample_train=subsample_train, subsample_groups=subsample_groups, do_not_save_data=do_not_save_data, stratify_bbox_sizes=stratify_bbox_sizes, quick_debug=quick_debug)
    print(data_names.keys())
    sys.stdout.flush()

    if run_predictions:
        run_predictions = run_predictions if utils.is_iterable(run_predictions) \
            else list(datasets.keys()) if n_folds == 1 else list(datasets[0].keys())

    for i in [None] if n_folds == 1 else range(n_folds) if fold is None else [fold]:
        if not skip_training and os.path.exists(os.path.join(output_dir, f'fold{i}')) and not resume:
            continue

        n_iters_per_epoch = calculate_epoch_length(
            datasets=datasets[i] if i is not None else datasets, imgs_per_batch=imgs_per_batch)

        if max_iter is None:
            _max_iter = int(np.ceil(epochs * n_iters_per_epoch))
            print(f"EPOCH is approximately: {n_iters_per_epoch} iterations")
            print(f"MAX_ITER set to {epochs} epochs: {_max_iter} iterations")
        else:
            _max_iter = max_iter

        if isinstance(eval_period, str) and eval_period.lower() == 'epoch':
            _eval_period = int(np.ceil(n_iters_per_epoch))
            print(f"EVAL_PERIOD set to 1 epoch: {_eval_period} iterations")
        else:
            _eval_period = eval_period

        cfg_kwargs = dict(
            base_model=base_model, weights=weights,
            num_classes=len(cat_names), device=utils.get_device(device, model='dt2'),
            max_iter=_max_iter, eval_period=_eval_period, checkpoint_period=int(checkpoint_period * n_iters_per_epoch) if checkpoint_period is not None else None,
            rpn_batch_size_per_img=rpn_batch_size_per_img, roi_batch_size_per_img=roi_batch_size_per_img,
            threshold=threshold, NMS_threshold=NMS_threshold, detections_per_image=detections_per_image,
            filter_empty=data_filters['TRAIN_RMV_EMPTY'],
            min_max_img_size=min_max_img_size, imgs_per_batch=imgs_per_batch, base_lr=base_lr, weight_decay=weight_decay,
            lr_gamma=lr_gamma, gamma_steps=(np.asarray(gamma_epochs) * n_iters_per_epoch).tolist() if gamma_epochs else None,
            output_dir=output_dir if n_folds == 1 else os.path.join(output_dir, f'fold{i}'),
            seed=seed, eval_class_subset=use_test_as_val and cat_names != test_cat_names, dropout_cnn=dropout_cnn, dropout_fc=dropout_fc,
            copy_paste=copy_paste, cfg_list=cfg_list
        )

        train_cfg = dt2_config(
            strict=strict,
            train_data=data_names['train' if n_folds == 1 else f'fold{i}.train'],
            test_data=data_names[('test' if use_test_as_val else 'val') if n_folds == 1 else f"fold{i}.{'test' if use_test_as_val else 'val'}"],
            **cfg_kwargs
        )
        train_cfg.freeze()

        if not skip_training:
            # This is extremely important - it will properly initialise seeds for all GPUs
            default_setup(train_cfg, args)

            # Only the main process will save the experiment config
            if detectron2.utils.comm.is_main_process():
                utils.experiment_config(
                    filename=os.path.join(train_cfg.OUTPUT_DIR, EXP_CFG_FN), categories=cat_names,
                    data_json_fn=data_json_fns, data_img_dir=data_img_dirs,
                    test_json_fn=test_json_fns, test_img_dir=test_json_fns, extra_train_json_fn=extra_train_json_fns,
                    extra_train_img_dir=extra_train_img_dirs,
                    n_folds=n_folds, fold=i, test_size=test_size, val_size=val_size, multi_label=multi_label,
                    data_filters=data_filters, img_transforms=img_transforms,
                    filename_groups=filename_groups, filename_groups_test_only=filename_groups_test_only
                )

            if checkpoint_metric is None:
                checkpoint_metric = ('segm/AP', True) if train_cfg.MODEL.MASK_ON else ('bbox/AP', True)

            trainer = get_trainer(cfg=train_cfg, eval_period=_eval_period, save_best_checkpoint=save_best_checkpoint,
                                  early_stopping=early_stopping, early_stopping_wait=early_stopping_wait,
                                  checkpoint_metric=checkpoint_metric, no_stopping=early_no_stopping_period)
            trainer.resume_or_load(resume=resume)
            trainer.train()

            for old_fn, new_fn in [
                (os.path.join(train_cfg.OUTPUT_DIR, 'model_best.pth'), os.path.join(train_cfg.OUTPUT_DIR, BEST_WEIGHTS_FN)),
                (os.path.join(train_cfg.OUTPUT_DIR, 'model_final.pth'), os.path.join(train_cfg.OUTPUT_DIR, LAST_WEIGHTS_FN)),
                (os.path.join(train_cfg.OUTPUT_DIR, 'model_best2.pth'), os.path.join(train_cfg.OUTPUT_DIR, BEST_WEIGHTS_FN2))
            ]:
                if os.path.exists(old_fn):
                    os.makedirs(os.path.dirname(new_fn), exist_ok=True)
                    shutil.move(src=old_fn, dst=new_fn)
            print('Finished training')
        else:
            print('Skipping training')

        for model_name, weights_fn in [
            ('best', BEST_WEIGHTS_FN if save_best_checkpoint else None),
            ('last', LAST_WEIGHTS_FN)
        ]:
            if prediction_models is None or model_name in prediction_models:
                if run_predictions and weights_fn is not None and \
                        os.path.exists(os.path.join(train_cfg.OUTPUT_DIR, weights_fn)):
                    for split in run_predictions:
                        dataset = data_names.get(split if n_folds == 1 else f"fold{i}.{split}")
                        if dataset is not None:
                            pred_out_dir = os.path.join(train_cfg.OUTPUT_DIR, f'{split}_predictions_{model_name}')
                            print(f'INFERENCE with "{model_name}" on "{split}" dataset')
                            dt2_predict(
                                cfg=train_cfg, weights_fn=os.path.join(train_cfg.OUTPUT_DIR, weights_fn), dataset=dataset,
                                model_cat_names=cat_names, predict_cat_names=test_cat_names if split != 'train' else None,
                                output_dir=pred_out_dir, output_fn=os.path.join(
                                    pred_out_dir, f'{os.path.basename(data_names[split if n_folds == 1 else f"fold{i}.{split}"]).replace("annotations", "predictions")}'),
                                save_pred_frames=save_pred_frames, vis_threshold=vis_threshold, evaluate=True,
                                eval_log_info=f'EVALUATION with model "{model_name}" on "{split}" dataset', strict=strict
                            )
        utils.compress_file(os.path.join(train_cfg.OUTPUT_DIR, 'validation', 'coco_instances_results.json'), force=True)

    return data_fn_prefix


def dt2_launch(main_func, args, num_gpus_per_machine=1, num_machines=1, machine_rank=0, dist_url='auto'):
    launch(
            main_func=main_func,
            num_gpus_per_machine=num_gpus_per_machine,
            num_machines=num_machines,
            machine_rank=machine_rank,
            dist_url=dist_url,
            args=args
    )


def get_default_predictor(config_fn, weights_fn=None, threshold=None, device=None):

    cfg = get_cfg()
    cfg.merge_from_file(config_fn)
    device = utils.get_device(device, model='dt2')
    cfg.MODEL.DEVICE = device
    if weights_fn is not None:
        cfg.MODEL.WEIGHTS = weights_fn
    if threshold is not None:
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = threshold
    cfg.freeze()

    return DefaultPredictor(cfg)
