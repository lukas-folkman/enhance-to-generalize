import json
import sys
import io
import warnings
from types import SimpleNamespace
from itertools import product, tee
import base64
import yaml
import gzip
import tempfile
import re
import os
import numpy as np
import copy
import time
import itertools
import datetime
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
from typing import Collection, Union, List
from six import string_types
# import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from collections.abc import Iterable
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, StratifiedGroupKFold, GroupShuffleSplit
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import check_random_state
from collections import defaultdict
import shutil
import pickle
import math
from contextlib import redirect_stdout
import cv2
from typing import Tuple

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

BGR_FORMAT = 'BGR'
RGB_FORMAT = 'RGB'
FAULTY_FPS_THR = 250
NEWLINE = '\n'

SORT, BOT_SORT, BYTE_TRACK, DUMMY_TRACK = 'sort', 'botsort', 'bytetrack', 'dummy'

TRAIN_RMV_EMPTY = 'TRAIN_RMV_EMPTY'
VAL_RMV_EMPTY = 'VAL_RMV_EMPTY'
TEST_RMV_EMPTY = 'TEST_RMV_EMPTY'

FAKE_CATEGORY = '#FAKE_CATEGORY'

X_SMALL, SMALL, S_MEDIUM, MEDIUM, LARGE, X_LARGE, XX_LARGE = 'x-small', 'small', 's-medium', 'medium', 'large', 'x-large', 'xx-large'

BOX_X_LENGTH = {
    X_SMALL: 16,
    SMALL: 32,
    S_MEDIUM: 64,
    MEDIUM: 96,
    LARGE: 128,
    X_LARGE: 256,
    XX_LARGE: 512
}

MIN_MAX_AREA = {
    X_SMALL: (0, BOX_X_LENGTH[X_SMALL] ** 2),
    SMALL: (BOX_X_LENGTH[X_SMALL] ** 2, BOX_X_LENGTH[SMALL] ** 2),
    S_MEDIUM: (BOX_X_LENGTH[SMALL] ** 2, BOX_X_LENGTH[S_MEDIUM] ** 2),
    MEDIUM: (BOX_X_LENGTH[S_MEDIUM] ** 2, BOX_X_LENGTH[MEDIUM] ** 2),
    LARGE: (BOX_X_LENGTH[MEDIUM] ** 2, BOX_X_LENGTH[LARGE] ** 2),
    X_LARGE: (BOX_X_LENGTH[LARGE] ** 2, BOX_X_LENGTH[X_LARGE] ** 2),
    XX_LARGE: (BOX_X_LENGTH[X_LARGE] ** 2, BOX_X_LENGTH[XX_LARGE] ** 2),
}


class Rectangle:
    __slots__ = '__x1', '__y1', '__x2', '__y2'

    def __init__(self, x1, y1, x2, y2):
        self.__setstate__((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))

    def xyxy(self):
        return [self.x1, self.y1, self.x2, self.y2]

    def xywh(self):
        return [self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1]

    def __repr__(self):
        return '{}({})'.format(type(self).__name__, ', '.join(map(repr, self)))

    def __eq__(self, other):
        return self.data == other.data

    def __ne__(self, other):
        return self.data != other.data

    def __hash__(self):
        return hash(self.data)

    def __len__(self):
        return 4

    def __getitem__(self, key):
        return self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __and__(self, other):
        x1, y1, x2, y2 = max(self.x1, other.x1), max(self.y1, other.y1), \
                         min(self.x2, other.x2), min(self.y2, other.y2)
        if x1 < x2 and y1 < y2:
            return type(self)(x1, y1, x2, y2)

    def __sub__(self, other):
        intersection = self & other
        if intersection is None:
            yield self
        else:
            x, y = {self.x1, self.x2}, {self.y1, self.y2}
            if self.x1 < other.x1 < self.x2:
                x.add(other.x1)
            if self.y1 < other.y1 < self.y2:
                y.add(other.y1)
            if self.x1 < other.x2 < self.x2:
                x.add(other.x2)
            if self.y1 < other.y2 < self.y2:
                y.add(other.y2)
            for (x1, x2), (y1, y2) in product(Rectangle.pairwise(sorted(x)),
                                              Rectangle.pairwise(sorted(y))):
                instance = type(self)(x1, y1, x2, y2)
                if instance != intersection:
                    yield instance

    def __getstate__(self):
        return self.x1, self.y1, self.x2, self.y2

    def __setstate__(self, state):
        self.__x1, self.__y1, self.__x2, self.__y2 = state

    @property
    def x1(self):
        return self.__x1

    @property
    def y1(self):
        return self.__y1

    @property
    def x2(self):
        return self.__x2

    @property
    def y2(self):
        return self.__y2

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    intersection = __and__

    difference = __sub__

    data = property(__getstate__)

    @staticmethod
    def pairwise(iterable):
        "s -> (s0, s1), (s1, s2), (s2, s3), ..."
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)

    @staticmethod
    def test():
        print('Example 1:')
        a = Rectangle(1, 1, 5, 5)
        b = Rectangle(3, 3, 7, 7)
        print(a & b)
        print(list(a - b))
        ##########################
        print('Example 2:')
        b = Rectangle(3, 2, 7, 4)
        print(a & b)
        print(list(a - b))
        ##########################
        print('Example 3:')
        b = Rectangle(2, 2, 4, 4)
        print(a & b)
        print(list(a - b))
        ##########################
        print('Example 4:')
        b = Rectangle(6, 2, 10, 6)
        print(a & b)
        print(list(a - b))
        ##########################
        print('Example 5:')
        b = Rectangle(0, 0, 6, 6)
        print(a & b)
        print(list(a - b))
        ##########################
        print('Example 6:')
        b = Rectangle(2, 0, 4, 6)
        print(a & b)
        print(list(a - b))


def count_items(arr):
    return dict(zip(*np.unique(arr, return_counts=True)))


def compress_file(fn, text_or_bin='t', force=False):
    if os.path.exists(fn):
        if os.path.exists(f'{fn}.gz'):
            if force:
                os.remove(f'{fn}.gz')
            else:
                raise ValueError(f'{fn}.gz already exists')
        with open(fn, mode=f'r{text_or_bin}') as f_in, gzip.open(f'{fn}.gz', mode=f'w{text_or_bin}') as f_out:
            f_out.writelines(f_in)
        assert os.path.exists(f'{fn}.gz')
        os.remove(fn)


def file_or_gzip_exists(fn):
    return fn is not None and \
           (os.path.exists(fn) or
            (not fn.endswith('.gz') and os.path.exists(f'{fn}.gz')) or
            (fn.endswith('.gz') and os.path.exists(fn[:-3])))


def exec_without_stdout(quiet, func, **kwargs):
    if quiet:
        with open(os.devnull, 'w') as f:
            with redirect_stdout(f):
                return func(**kwargs)
    else:
        return func(**kwargs)


def create_gt_COCO(fn_or_dataset, fix_zero_ann_ids=False, verbose=False):
    gt_coco = COCO()
    if isinstance(fn_or_dataset, str):
        gt_fn = fn_or_dataset
        gt_coco.dataset = read_json(fn=fn_or_dataset, fix_zero_ann_ids=fix_zero_ann_ids, verbose=verbose)
    else:
        gt_fn = None
        assert_dataset_correct(fn_or_dataset)
        gt_coco.dataset = copy.deepcopy(fn_or_dataset)
    exec_without_stdout(quiet=True, func=gt_coco.createIndex)
    gt_coco.dataset.pop('licenses', None)
    [img.pop('license', None) for img in gt_coco.dataset['images']]
    return gt_coco, gt_fn


def iou(bbox1, bbox2, format='XYWH'):
    '''
    https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    '''
    assert format.lower() in ['xywh', 'xyxy']
    bbox1 = from_XYWH_to_XYXY(bbox1)
    bbox2 = from_XYWH_to_XYXY(bbox2)

    # determine the (x, y)-coordinates of the intersection rectangle
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    # compute the area of intersection rectangle
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # compute the area of both the prediction and ground-truth rectangles
    bbox1_area = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    bbox2_area = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)

    iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
    return iou


class COCOResults:
    def __init__(self, gt_coco: Union[str, dict], dt_coco: Union[str, dict, np.ndarray],
                 fix_zero_ann_ids=False, allow_zero_area_boxes='warn', filename_group=None,
                 class_agnostic_nms=None, verbose=False):
        assert isinstance(gt_coco, (str, dict))
        assert isinstance(dt_coco, (str, dict, np.ndarray))
        assert not isinstance(gt_coco, str) or file_or_gzip_exists(gt_coco)
        assert not isinstance(dt_coco, str) or file_or_gzip_exists(dt_coco), dt_coco
        assert not isinstance(gt_coco, dict) or (
                'annotations' in gt_coco and 'images' in gt_coco and 'categories' in gt_coco)
        assert not isinstance(dt_coco, dict) or ('annotations' in dt_coco)
        # array shape must be (N, 7) with the following columns: (imageID, x, y, w, h, score, class)
        assert not isinstance(dt_coco, np.ndarray) or dt_coco.shape[1] == 7
        assert allow_zero_area_boxes in ['warn', 'raise', True, False]

        self.gt_coco = gt_coco
        self.dt_coco = dt_coco
        self.gt_fn = None
        self.dt_fn = None
        self.eval = None
        self.precision = None
        self.PRs = None
        self.stats = None
        self.allow_zero_area_boxes = allow_zero_area_boxes
        self.fix_zero_ann_ids = fix_zero_ann_ids
        self.class_agnostic_nms = class_agnostic_nms
        self.verbose = verbose

        self.gt_coco, self.gt_fn = create_gt_COCO(gt_coco, fix_zero_ann_ids=self.fix_zero_ann_ids, verbose=self.verbose)

        if isinstance(dt_coco, str):
            self.dt_fn = dt_coco
            self.dt_coco = \
                read_json(fn=self.dt_fn, only_preds=True, fix_zero_ann_ids=self.fix_zero_ann_ids, verbose=self.verbose)[
                    'annotations']
        elif isinstance(dt_coco, dict):
            assert_dataset_correct(dt_coco, only_preds=True)
            self.dt_coco = dt_coco['annotations']
        else:
            self.dt_coco = dt_coco
        assert isinstance(self.dt_coco, list) and all([isinstance(x, dict) for x in self.dt_coco])

        if self.class_agnostic_nms is not None:
            assert 0 <= self.class_agnostic_nms <= 1
            from torchvision.ops import nms
            import torch
            boxes = torch.tensor([ann['bbox'] for ann in self.dt_coco])
            assert boxes.shape == ((len(self.dt_coco), 4)), boxes.shape
            scores = torch.tensor([ann['score'] for ann in self.dt_coco])
            assert scores.shape == (len(self.dt_coco),), scores.shape
            idx = nms(boxes=boxes, scores=scores, iou_threshold=self.class_agnostic_nms)
            # print('dt_coco before NMS', len(self.dt_coco))
            self.dt_coco = [ann for i, ann in enumerate(self.dt_coco) if i in idx]
            # print('dt_coco after NMS', len(self.dt_coco))

        self.index_dt_coco()

        if filename_group:
            group_splitter = ".frame_"
            groups = get_filename_groups(self.gt_coco.dataset['images'], split_char=group_splitter, n_parts=1)
            drop = [img['id'] for img, grp in zip(self.gt_coco.dataset['images'], groups) if grp != filename_group]
            self.gt_coco.dataset['annotations'] = [ann for ann in self.gt_coco.dataset['annotations'] if
                                                   ann['image_id'] not in drop]
            self.dt_coco.dataset['annotations'] = [ann for ann in self.dt_coco.dataset['annotations'] if
                                                   ann['image_id'] not in drop]
            exec_without_stdout(quiet=not self.verbose, func=self.gt_coco.createIndex)
            exec_without_stdout(quiet=not self.verbose, func=self.dt_coco.createIndex)
        self.assert_results_correct()

    def assert_results_correct(self):
        assert_dataset_correct(self.gt_coco.dataset)
        assert_dataset_correct(self.dt_coco.dataset, allow_zero_area_boxes=self.allow_zero_area_boxes)
        assert self.gt_coco.dataset['images'] == self.dt_coco.dataset['images']

    def index_dt_coco(self):
        assert isinstance(self.dt_coco, (list, np.ndarray))
        if len(self.dt_coco) != 0:
            self.dt_coco = exec_without_stdout(quiet=not self.verbose, func=self.gt_coco.loadRes, resFile=self.dt_coco)
        else:
            self.dt_coco = COCO()
            self.dt_coco.dataset['images'] = [img for img in self.gt_coco.dataset['images']]
            self.dt_coco.dataset['categories'] = copy.deepcopy(self.gt_coco.dataset['categories'])
            self.dt_coco.dataset['annotations'] = []
            exec_without_stdout(quiet=not self.verbose, func=self.dt_coco.createIndex)

    def reload(self):
        if self.gt_fn is not None:
            self.gt_coco, _ = create_gt_COCO(self.gt_fn, fix_zero_ann_ids=self.fix_zero_ann_ids, verbose=self.verbose)
            self.gt_coco.dataset.pop('licenses', None)
            [img.pop('license', None) for img in self.gt_coco.dataset['images']]
        else:
            print('WARNING: cannot reload gt_coco, gt_fn is None', file=sys.stderr)
        if self.dt_fn is not None:
            self.dt_coco = \
                read_json(fn=self.dt_fn, only_preds=True, fix_zero_ann_ids=self.fix_zero_ann_ids, verbose=self.verbose)[
                    'annotations']
            self.index_dt_coco()
        else:
            print('WARNING: cannot reload dt_coco, dt_fn is None', file=sys.stderr)
        if self.gt_coco is not None and self.dt_coco is not None:
            self.assert_results_correct()
        else:
            print('WARNING: incomplete results, gt_coco is None or dt_coco is None', file=sys.stderr)

    def evaluate(self, conf_thr=None, iouType='bbox', useCats=True, imgIds=None, catIds=None, iouThrs=None,
                 recThrs=None, areaRng=None, areaRngLbl=None, maxDets=None, PR_curve=True,
                 conf_thresholds=(0.05, 0.1, 0.2, 0.25, 0.5, 0.75), verbose=None):
        if verbose is None:
            verbose = self.verbose
        if areaRng is not None:
            assert areaRngLbl is not None and len(areaRng) == len(areaRngLbl)
        else:
            assert areaRngLbl is None
            areaRng = []  # list(MIN_MAX_AREA.values())
            areaRngLbl = []  # list(MIN_MAX_AREA.keys())

        if conf_thr:
            dt_coco = copy.deepcopy(self.dt_coco)
            dt_coco.dataset['annotations'] = [ann for ann in dt_coco.dataset['annotations'] if ann['score'] >= conf_thr]
            exec_without_stdout(quiet=not self.verbose, func=dt_coco.createIndex)

        self.eval = COCOEvalPlus(
            cocoGt=self.gt_coco, cocoDt=dt_coco if conf_thr else self.dt_coco, iouType=iouType, useCats=useCats,
            imgIds=imgIds, catIds=catIds, iouThrs=iouThrs, recThrs=recThrs, areaRng=areaRng, areaRngLbl=areaRngLbl,
            maxDets=maxDets
        )
        exec_without_stdout(quiet=not verbose, func=self.eval.evaluate)
        exec_without_stdout(quiet=not verbose, func=self.eval.accumulate)
        exec_without_stdout(quiet=not verbose, func=self.eval.summarize, verbose=verbose)
        self.stats = copy.copy(self.eval.stats)

        _iouThr = 0.5 if not iouThrs else iouThrs if not is_iterable(iouThrs) else iouThrs[0]
        _areaRng = 'all' if not areaRngLbl else areaRngLbl if not is_iterable(areaRngLbl) else areaRngLbl[0]
        _maxDets = 100 if not maxDets else maxDets if not is_iterable(maxDets) else maxDets[0]

        if PR_curve:
            self.precision = self.eval.calc_PR_curve(iouThr=_iouThr, areaRng=_areaRng, maxDets=_maxDets, catIds=catIds,
                                                     verbose=verbose)
        if conf_thresholds:
            if not is_iterable(conf_thresholds):
                conf_thresholds = [conf_thresholds]
            self.PRs = {
                thr: self.eval.precision_recall_at_conf_thr(
                    conf_thr=thr, iouThr=_iouThr, areaRng=_areaRng, maxDets=_maxDets) for thr in conf_thresholds
            }

    def __repr__(self):
        if self.gt_coco is not None and self.dt_coco is not None:
            return "{}images: {}\nground truth annotations: {}\ndetections: {}\ncategories: {} ({})".format(
                f"{self.gt_coco.dataset['info']}\n" if 'info' in self.gt_coco.dataset else '',
                len(self.gt_coco.dataset['images']), len(self.gt_coco.dataset['annotations']),
                len(self.dt_coco.dataset['annotations']),
                len(self.gt_coco.dataset['categories']), ', '.join(get_category_names(self.gt_coco.dataset))
            )
        else:
            return f"{self.gt_fn}\n{self.dt_fn}"

    # def __iter__(self):
    #     return self.box_results.__iter__()
    #
    # def __getitem__(self, index):
    #     return self.box_results[index]


def evaluate(gt_coco, dt_coco, conf_thr=None, iouType='bbox', useCats=True, imgIds=None, catIds=None, iouThrs=None,
             recThrs=None, areaRng=None, areaRngLbl=None, maxDets=None, PR_curve=True, allow_zero_area_boxes=True,
             fix_zero_ann_ids=False, filename_group=None, class_agnostic_nms=None, verbose=True):
    results = COCOResults(gt_coco=gt_coco, dt_coco=dt_coco, allow_zero_area_boxes=allow_zero_area_boxes,
                          fix_zero_ann_ids=fix_zero_ann_ids, filename_group=filename_group, class_agnostic_nms=class_agnostic_nms, verbose=verbose)
    results.evaluate(conf_thr=conf_thr, iouType=iouType, useCats=useCats, imgIds=imgIds, catIds=catIds, iouThrs=iouThrs,
                     recThrs=recThrs, areaRng=areaRng, areaRngLbl=areaRngLbl, maxDets=maxDets, PR_curve=PR_curve,
                     verbose=verbose)
    return results


class COCOEvalPlus(COCOeval):

    def __init__(
            self, cocoGt=None, cocoDt=None, iouType='bbox', useCats=True,
            imgIds=None, catIds=None, iouThrs=None, recThrs=None, areaRng=None, areaRngLbl=None, maxDets=None
    ):
        """
        :param cocoGt: ground truth
        :param cocoDt: detections
        :param iouType: bbox or segm
        :param useCats: True if true use category labels for evaluation
        :param imgIds: img ids to use for evaluation
        :param catIds: cat ids to use for evaluation
        :param iouThrs: [.5:.05:.95] T=10 IoU thresholds for evaluation
        :param recThrs: [0:.01:1] R=101 recall thresholds for evaluation
        :param areaRng: object area ranges for evaluation
        :param areaRngLbl:object area names for evaluation
        :param maxDets: thresholds on max detections per image
        """
        assert (areaRng is None and areaRngLbl is None) or (areaRng is not None and areaRngLbl is not None)
        assert areaRng is None or len(areaRng) == len(areaRngLbl)

        if not iouType:
            print('WARNING: iouType not specified, using default iouType "bbox"')
        if iouType != 'bbox':
            print(
                f'WARNING: iouType "{iouType}" specified, implementation for "{iouType}" is not thoroughly tested')
        super().__init__(cocoGt=cocoGt, cocoDt=cocoDt, iouType=iouType)

        self.params.useCats = useCats
        if imgIds is not None:
            self.params.imgIds = imgIds if is_iterable(imgIds) else [imgIds]
        # if not useCats:
        #     self.params.catIds = [-1]
        # else:
        #     if catIds is not None:
        #         self.params.catIds = catIds if is_iterable(catIds) else [catIds]
        if iouThrs is not None:
            self.params.iouThrs = iouThrs if is_iterable(iouThrs) else [iouThrs]
        if recThrs is not None:
            self.params.recThrs = recThrs if is_iterable(recThrs) else [recThrs]
        if areaRng is not None:
            self.params.areaRng = list(areaRng) if is_iterable(areaRng) else [areaRng]
        if areaRngLbl is not None:
            self.params.areaRngLbl = list(areaRngLbl) if is_iterable(areaRngLbl) else [areaRngLbl]
        if maxDets is not None:
            self.params.maxDets = maxDets if is_iterable(maxDets) else [maxDets]

        if 'all' not in self.params.areaRngLbl:
            self.params.areaRngLbl.insert(0, 'all')
            self.params.areaRng.insert(0, [0, 1e5 ** 2])

    # def TPs_and_FPs(self, areaRng='all'):
    #
    #     areaRng = self.params.areaRng[self.params.areaRngLbl.index(areaRng)]
    #
    #     FPs, TPs = [], []
    #     for img in self.evalImgs:
    #         if img is not None and img['aRng'] == areaRng:
    #             fp = np.asarray(img['dtIds'])[img['dtMatches'].reshape((-1,)) == 0]
    #             tp = np.asarray(img['dtIds'])[img['dtMatches'].reshape((-1,)) != 0]
    #             if len(fp) != 0:
    #                 FPs.append(fp)
    #             if len(tp) != 0:
    #                 TPs.append(tp)
    #
    #     if len(TPs) != 0:
    #         TPs = np.concatenate(TPs)
    #     assert len(TPs) == len(set(TPs))
    #
    #     if len(FPs) != 0:
    #         FPs = np.concatenate(FPs)
    #     assert len(FPs) == len(set(FPs))
    #
    #     print('Here')
    #     _TPs, _FPs, _ = self.TPs_FPs_FNs(concat=True)
    #     print(sorted(_TPs) == sorted(TPs))
    #     print(sorted(_FPs) == sorted(FPs))
    #     assert sorted(_TPs) == sorted(TPs)
    #     assert sorted(_FPs) == sorted(FPs)
    #
    #     return TPs, FPs

    def precision_recall_at_conf_thr(self, conf_thr=None, iouThr=0.5, areaRng='all', maxDets=100):
        TPs, FPs, FNs = self.TPs_FPs_FNs(iouThr=iouThr, areaRng=areaRng, maxDets=maxDets, conf_thr=conf_thr)
        precision = np.asarray([TPs[k] / (TPs[k] + FPs[k]) for k in range(len(self._paramsEval.catIds))])
        recall = np.asarray([TPs[k] / (TPs[k] + FNs[k]) for k in range(len(self._paramsEval.catIds))])
        return precision, recall

    def TPs_FPs_FNs(self, iouThr=0.5, areaRng='all', maxDets=100, conf_thr=None, counts_per_cat=False, concat=False,
                    ids_per_cat=False):
        if not counts_per_cat and not concat and not ids_per_cat:
            counts_per_cat = True
        assert sum([ids_per_cat, counts_per_cat, concat]) == 1

        if isinstance(areaRng, str):
            areaRng = self._paramsEval.areaRng[self._paramsEval.areaRngLbl.index(areaRng)]
        iouThr_idx = [i for i, t in enumerate(self._paramsEval.iouThrs) if t == iouThr]
        assert len(iouThr_idx) == 1
        iouThr_idx = iouThr_idx[0]

        FPs, TPs, FNs = defaultdict(list), defaultdict(list), defaultdict(lambda: 0)
        for img in self.evalImgs:
            if img is not None:
                assert img['aRng'] == areaRng
                assert img['maxDet'] == maxDets
                assert not np.any(img['gtIgnore'])
                assert not np.any(img['dtIgnore'])
                dtIds = np.asarray(img['dtIds'])
                gtIds = np.asarray(img['gtIds'])
                assert 0 not in dtIds
                assert 0 not in gtIds
                dtMatches = img['dtMatches'][iouThr_idx]
                gtMatches = img['gtMatches'][iouThr_idx]
                assert all([m == 0 or m in dtIds for m in gtMatches]), img
                dtScores = np.asarray(img['dtScores'])
                if conf_thr is not None:
                    conf_mask = dtScores >= conf_thr
                    dtIds = dtIds[conf_mask]
                    dtMatches = dtMatches[conf_mask]
                    dtScores = dtScores[conf_mask]
                    gtMatches = np.asarray([0 if m not in dtIds else m for m in gtMatches])

                fp = dtIds[dtMatches == 0]
                tp = dtIds[dtMatches != 0]
                fn = sum(gtMatches == 0)

                if len(fp) != 0:
                    FPs[img['category_id']].append(fp)
                if len(tp) != 0:
                    TPs[img['category_id']].append(tp)
                FNs[img['category_id']] += fn

        for cat_id in FPs:
            FPs[cat_id] = np.concatenate(FPs[cat_id])
            assert len(FPs[cat_id]) == len(set(FPs[cat_id]))
        for cat_id in TPs:
            TPs[cat_id] = np.concatenate(TPs[cat_id])
            assert len(TPs[cat_id]) == len(set(TPs[cat_id]))
            assert len(set(FPs[cat_id]).intersection(TPs[cat_id])) == 0

        if ids_per_cat:
            FPs = np.asarray([FPs[c] for c in self._paramsEval.catIds], dtype=object)
            TPs = np.asarray([TPs[c] for c in self._paramsEval.catIds], dtype=object)
            FNs = np.asarray([FNs[c] for c in self._paramsEval.catIds], dtype=object)
        elif counts_per_cat:
            FPs = np.asarray([len(FPs[c]) for c in self._paramsEval.catIds])
            TPs = np.asarray([len(TPs[c]) for c in self._paramsEval.catIds])
            FNs = np.asarray([FNs[c] for c in self._paramsEval.catIds])
        elif concat:
            FPs = np.concatenate([FPs[c] for c in self._paramsEval.catIds])
            TPs = np.concatenate([TPs[c] for c in self._paramsEval.catIds])
            FNs = np.sum([FNs[c] for c in self._paramsEval.catIds])

        return TPs, FPs, FNs

    def calc_PR_curve(self, iouThr=0.5, areaRng='all', maxDets=100, catIds=None, verbose=False):
        if verbose and (iouThr != 0.5 or areaRng != 'all' or maxDets != 100):
            print(
                f'Calculating PR cruve for iouThr={iouThr}, areaRng={areaRng}, maxDets={maxDets}, catIds={self._paramsEval.catIds}')
        assert self.evalImgs
        assert self._paramsEval.useCats == self.params.useCats
        # assert list(self._paramsEval.catIds) == list(self.params.catIds), \
        #     (('_paramsEval', self._paramsEval.useCats, self._paramsEval.catIds),
        #      ('params', self.params.useCats, self.params.catIds))
        assert catIds is None or (self.params.useCats == 1 and np.isin(catIds, self.params.catIds).all())
        assert not is_iterable(iouThr) and not is_iterable(areaRng) and not is_iterable(maxDets)

        precision = self.calculate(metric='precision', iouThr=iouThr, areaRng=areaRng, maxDets=maxDets,
                                   mean=False)  # .squeeze(0)
        assert precision.ndim == 2 and precision.shape[0] == len(self.params.recThrs) and precision.shape[1] == (
            len(self.params.catIds) if self.params.useCats else 1), precision.shape

        # # dimension of recall: [TxKxAxM]
        # recall = self.calculate(metric='recall', iouThr=iouThr, areaRng=areaRng, maxDets=maxDets, mean=False)#.squeeze(0)
        # assert recall.ndim == 1 and recall.shape[0] == (len(self.params.catIds) if self.params.useCats else 1), recall.shape
        #
        # # dimension of precision/scores: [TxRxKxAxM]
        # scores = self.calculate(metric='scores', iouThr=iouThr, areaRng=areaRng, maxDets=maxDets, mean=False)#.squeeze(0)
        # assert precision.shape == scores.shape, (precision.shape, scores.shape)

        return precision

    def calculate(self, metric, iouThr=None, areaRng='all', maxDets=100, mean='auto', squeeze=True):

        # conf_thr = 0.25
        # iouThr = 0.5
        # areaRng = 'all'
        # maxDets = 100
        # if isinstance(areaRng, str):
        #     areaRng = E._paramsEval.areaRng[E._paramsEval.areaRngLbl.index(areaRng)]
        #
        # assert count_items(e['params'].iouThrs)[iouThr] == 1
        # assert sum([tuple(a) == tuple(areaRng) for a in e['params'].areaRng]) == 1
        # assert count_items(e['params'].maxDets)[maxDets] == 1
        #
        # e = E.eval
        #
        # t_idx = [i for i, t in enumerate(e['params'].iouThrs) if t == iouThr][0]
        # a_idx = [i for i, a in enumerate(e['params'].areaRng) if tuple(a) == tuple(areaRng)][0]
        # m_idx = [i for i, m in enumerate(e['params'].maxDets) if m == maxDets][0]
        # S = e['scores'][t_idx, :, :, a_idx, m_idx]
        # P = e['precision'][t_idx, :, :, a_idx, m_idx]
        # R = e['recall'][t_idx, :, a_idx, m_idx]
        # assert S.shape == (e['counts'][1], e['counts'][2])
        # assert all([sorted(S[:, k])[::-1] == S[:, k].tolist() for k in range(S.shape[1])])
        # S.shape

        assert metric in ['precision', 'recall', 'scores']
        assert mean in ['auto', True, False]
        if mean == 'auto':
            mean = metric in ['precision', 'recall']

        p = self.params
        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
        tind = np.arange(len(p.iouThrs), dtype=int)
        if iouThr is not None:
            if not is_iterable(iouThr):
                iouThr = [iouThr]
            tind = tind[np.isin(p.iouThrs, iouThr)]

        values = self.eval[metric]
        if metric in ['precision', 'scores']:
            # dimension of precision/scores: [TxRxKxAxM]
            values = values[tind, :, :, aind, mind]
        elif metric == 'recall':
            # dimension of recall: [TxKxAxM]
            values = values[tind, :, aind, mind]
        else:
            raise ValueError
        assert (len(tind) == 1 and values.shape[0] == 1) or (len(tind) != 1 and values.shape[0] != 1)

        if mean:
            values = values[values > -1]
            return np.mean(values) if len(values) != 0 else -1
        else:
            return values.squeeze(0) if squeeze and len(tind) == 1 else values

    def __repr__(self):
        return f'COCOEvalPlus\ncatIds: {self.params.catIds}\nareaRngLbl: {self.params.areaRngLbl}\nmaxDets:{self.params.maxDets}'

    def __str__(self):
        return self.summarize(human_output=True, verbose=False)

    def summarize(self, human_output=False, verbose=True):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''

        def _summarize(ap=True, iouThr=None, areaRng='all', maxDets=100, human_output=False, verbose=True):
            mean_s = self.calculate(
                metric='precision' if ap else 'recall', iouThr=iouThr, areaRng=areaRng, maxDets=maxDets, mean=True)

            if human_output or verbose:
                iStr = ' {:<18} {} @[ IoU={:<9} | area={:>8s} | maxDets={:>3d} ] = {:0.3f}'
                titleStr = 'Average Precision' if ap else 'Average Recall'
                typeStr = '(AP)' if ap else '(AR)'
                iouStr = '{:0.2f}:{:0.2f}'.format(self.params.iouThrs[0], self.params.iouThrs[-1]) \
                    if iouThr is None else '{:0.2f}'.format(iouThr)
                out_str = iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)
                if verbose:
                    print(out_str)
            if human_output:
                return out_str
            else:
                name_s = f"{'mAP' if ap else 'mAR'}" \
                         f"{'' if iouThr is None else int(iouThr * 100)}" \
                         f"{'' if areaRng == 'all' else f'_{areaRng}'}" \
                         f"{'' if maxDets == 100 else f'_top{maxDets}'}"
                return name_s, mean_s

        def _summarizeDets(human_output=False, verbose=True):
            n_outputs = 3 + len(self.params.areaRngLbl) - 1 + len(self.params.maxDets) + len(self.params.areaRngLbl) - 1
            # if human_output:
            #     stats = [None] * n_outputs
            # else:
            #     stats = np.zeros((n_outputs,))
            stats = [None] * n_outputs

            maxDets = max(self.params.maxDets)
            # Precision
            stats[0] = _summarize(ap=True, iouThr=None, areaRng='all', maxDets=maxDets, human_output=human_output,
                                  verbose=verbose)
            stats[1] = _summarize(ap=True, iouThr=.5, areaRng='all', maxDets=maxDets, human_output=human_output,
                                  verbose=verbose)
            stats[2] = _summarize(ap=True, iouThr=.75, areaRng='all', maxDets=maxDets, human_output=human_output,
                                  verbose=verbose)
            offset = 3

            for i, areaRng in enumerate([a for a in self.params.areaRngLbl if a != 'all']):
                stats[i + offset] = _summarize(ap=True, iouThr=None, areaRng=areaRng, maxDets=maxDets,
                                               human_output=human_output, verbose=verbose)
            offset += len(self.params.areaRngLbl) - 1

            # Recall
            for i, nDets in enumerate(sorted(self.params.maxDets)):
                stats[i + offset] = _summarize(ap=False, iouThr=None, areaRng='all', maxDets=nDets,
                                               human_output=human_output, verbose=verbose)
            offset += len(self.params.maxDets)

            for i, areaRng in enumerate([a for a in self.params.areaRngLbl if a != 'all']):
                stats[i + offset] = _summarize(ap=False, iouThr=None, areaRng=areaRng, maxDets=maxDets,
                                               human_output=human_output, verbose=verbose)
            offset += len(self.params.areaRngLbl) - 1

            return stats

        if not self.eval:
            raise Exception('Please run accumulate() first')
        elif self.params.iouType == 'segm' or self.params.iouType == 'bbox':
            assert self._paramsEval.useCats == self.params.useCats
            if human_output:
                stats = _summarizeDets(human_output=True, verbose=verbose)
                return '\n'.join(stats)
            else:
                self.stats = dict(_summarizeDets(human_output=False, verbose=verbose))
                return self.stats
        elif self.params.iouType == 'keypoints':
            super().summarize()
            return self.stats


def precision(tp, fp):
    return tp / (tp + fp)


def recall(tp, fn):
    return tp / (tp + fn)


def F1(pr, rc):
    return 2 * pr * rc / (pr + rc)


def _assert_precision_recall(precision, recall):
    assert isinstance(recall, list) or recall.ndim == 1
    assert isinstance(precision, list) or precision.ndim == 1
    assert len(precision) == len(recall)


def best_F1(precision, recall=None):
    if recall is None:
        recall = np.arange(0, 1.01, 0.01)
    _assert_precision_recall(precision, recall)

    best_f1, best_i = -1, None
    for i in range(len(recall)):
        f1 = F1(pr=precision[i], rc=recall[i])
        if f1 > best_f1:
            best_f1 = f1
            best_i = i
    assert best_i is not None
    return best_f1, best_i


def plot_PR_curve(precision, recall=None, catIds=None, all_catIds=None, average_cats=False,
                  color=None, palette=None, ls=None, ax=None, figsize=None, label=None,
                  f1_curves=False, f1_values=None, f1_color="lightgrey", f1_annot_color=None, f1_lw=None,
                  f1_format="$F_1={:0.1f}$",
                  f1_legend=None, f1_fontsize=None):
    import seaborn as sns
    if isinstance(palette, str):
        palette = sns.color_palette(palette=palette)
    n_colors = len(palette)
    if color is not None and isinstance(color, int):
        color = palette[color % n_colors]
    if f1_values is None:
        f1_values = [0.6, 0.7, 0.8, 0.9]

    if recall is None:
        recall = np.arange(0, 1.01, 0.01)

    if average_cats:
        assert catIds is None
        precision = precision.mean(axis=1).reshape((precision.shape[0], 1))

    if all_catIds is None:
        all_catIds = range(1, precision.shape[1] + 1)

    if ax is None and figsize is not None:
        _, ax = plt.subplots(1, 1, figsize=figsize)

    if f1_curves:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=figsize)
        for f1 in f1_values:
            x = np.linspace(0.01, 1, 100)
            x = x[(2 * x - f1) != 0]
            y = f1 * x / (2 * x - f1)
            ax.plot(x[(y >= 0) & (y <= 1)], y[(y >= 0) & (y <= 1)], color=f1_color, lw=f1_lw, label=f1_legend)
            ax.annotate(f1_format.format(f1), xy=(1, y[-1] + 0.02), color=f1_annot_color, fontsize=f1_fontsize)
            f1_legend = None

    for k, cat_id in enumerate(all_catIds):
        if catIds is None or cat_id in catIds:
            _assert_precision_recall(precision[:, k], recall)
            if len(all_catIds) == 1:
                cat_info = ''
            else:
                cat_info = f' category: {cat_id}'
            ax = sns.lineplot(
                x=recall, y=precision[:, k], ax=ax, ls=ls, color=color if color is not None else palette[k % n_colors],
                label=f'{label if label is not None else ""}{cat_info}'
            )
            # ax = sns.lineplot(x=recall, y=scores[k], ax=ax)
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    return ax


def make_PR_plot(r, split, seed, label=None, title=None, show_mAP=True, show_F1=False, linestyle=None, color=None,
                 despine=True, legend='lower left', ax=None):
    import seaborn as sns
    if seed is None:
        precision = [r[split][s].eval.calc_PR_curve() for s in r]
        # stack the different seeds and average across seeds
        precision = np.stack(precision).mean(axis=0)
    else:
        precision = r[split][seed].eval.calc_PR_curve()
        # precision is [R, K] (101 recall thresholds x K categories)
    assert precision.ndim == 2 and precision.shape[0] == 101
    recall = np.arange(0, 1.01, 0.01)
    # average across categories
    best_f1, best_i = best_F1(precision.mean(axis=1), recall)

    ax = plot_PR_curve(
        precision,
        recall=recall,
        catIds=None, all_catIds=None,
        average_cats=True,
        ls=linestyle, color=color, ax=ax,
        label=f'{label if label else ""}'
              f'{" | " if label and show_mAP else ""}'
              f'{f"mAP={precision.mean():.2f}" if show_mAP else ""}'
              f'{NEWLINE if (label or show_mAP) and show_F1 else ""})'
              f'{f"best F1={best_f1:.2f} (pr={precision.mean(axis=1)[best_i]:.2f} rc={recall[best_i]:.2f}" if show_F1 else ""})'
    )
    if legend:
        ax.legend(loc=legend)
    else:
        ax.legend().set_visible(False)
    if title:
        ax.set_title(title)
    if despine:
        sns.despine(ax=ax)
    return ax


def flip_color_channels(image):
    return image[:, :, ::-1]


class FixedStratifiedGroupKFold:
    def __init__(self, n_splits=5, shuffle=False, random_state=None, max_attempts=None, warning_only=True):
        if max_attempts is None:
            max_attempts = 10000
        rng = check_random_state(random_state)
        self.max_attempts = max_attempts
        self.warning_only = warning_only
        self._splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=rng)

    def split(self, X, y, groups):
        if self.warning_only:
            splits = [(train, test) for train, test in self._splitter.split(X=X, y=y, groups=groups)]
            if not all([set(y[train]) == set(y[test]) for train, test in splits]):
                print(('WARNING: cannot split while preserving every class label in every split', np.unique(groups)))
        else:
            success = False
            for r in range(self.max_attempts):
                splits = [(train, test) for train, test in self._splitter.split(X=X, y=y, groups=groups)]
                if all([set(y[train]) == set(y[test]) for train, test in splits]):
                    success = True
                    break
            if not success:
                raise ValueError('ERROR: cannot split while preserving every class label in every split',
                                 np.unique(groups))

        for train, test in splits:
            yield train, test


class StratifiedGroupShuffleSplit:
    def __init__(self, n_splits, test_size, random_state=None, max_attempts=None, swapping_allowed=False):
        assert 0 < test_size < 1
        self.n_splits = n_splits
        self.test_size = test_size
        self.swapping_allowed = swapping_allowed
        self._splitter = FixedStratifiedGroupKFold(
            n_splits=2, shuffle=True, random_state=random_state, max_attempts=max_attempts)

    def split(self, X, y, groups):
        groups = np.asarray(groups)

        def get_fractions(labels, labelled_classes):
            return np.asarray([(labels == lb).sum() / len(labels) for lb in labelled_classes])

        def check_every_label_present(gl_matrix):
            return (gl_matrix.sum(axis=0) != 0).all()

        i = 0
        while i < self.n_splits:
            for train, test in self._splitter.split(X=X, y=y, groups=groups):
                if i < self.n_splits:
                    while len(test) / len(y) > (self.test_size if self.test_size <= 0.5 else (1 - self.test_size)):
                        uniq_groups = np.unique(groups[test])
                        uniq_labels = np.unique(y[test])
                        # assert list(uniq_labels) == list(np.unique(y[train]))

                        groups_and_labels = np.zeros((len(uniq_groups), len(uniq_labels)))
                        for group, label in zip(groups[test], y[test]):
                            groups_and_labels[uniq_groups == group, uniq_labels == label] += 1

                        train_fractions = get_fractions(y[train], uniq_labels)

                        best_drop_group, best_std = None, None
                        for drop_group in uniq_groups:
                            if check_every_label_present(groups_and_labels[uniq_groups != drop_group]):
                                drop_idx = np.arange(len(groups), dtype=int)[groups == drop_group]
                                test_without_group = [idx for idx in test if idx not in drop_idx]
                                test_fractions = get_fractions(y[test_without_group], uniq_labels)
                                std = np.std(train_fractions - test_fractions)
                                if best_drop_group is None or best_std > std:
                                    best_drop_group = drop_group
                                    best_std = std

                        if best_drop_group is not None:
                            drop_idx = np.arange(len(groups), dtype=int)[groups == best_drop_group]
                            assert len(set(drop_idx).intersection(test)) == len(drop_idx)
                            assert len(set(drop_idx).intersection(train)) == 0
                            if abs(self.test_size - (len(test) / len(y))) \
                                    > abs(self.test_size - ((len(test) - len(drop_idx)) / len(y))):
                                test = [idx for idx in test if idx not in drop_idx]
                                train = np.concatenate([train, drop_idx])
                            else:
                                print('INFO: cannot drop any more groups without making split too small')
                                break
                        else:
                            print('INFO: cannot drop any more groups without losing label category')
                            break

                    if self.swapping_allowed:
                        if (self.test_size < 0.5 and len(test) > len(train)) or \
                                (1 > self.test_size > 0.5 and len(test) < len(train)):
                            print('Swapping to adjust test set size vs train set size')
                            swap = train
                            train = test
                            test = swap

                        if set(y[test]).issuperset(y[train]) and not set(y[train]).issuperset(y[test]):
                            print('Swapping to have all labels in the training set')
                            swap = train
                            train = test
                            test = swap

                    print('test', len(test), 'train', len(train))
                    i += 1
                    yield train, test


def convert_predictions(outputs: Collection, model, return_type='numpy'):
    assert model in ['dt2', 'yolo8']
    assert return_type in ['tensor', 'numpy', 'list']
    assert is_iterable(outputs, dict_allowed=False)

    def _convert_prediction(output, model, return_type):
        if model == 'dt2':
            boxes_xywh = from_XYXY_to_XYWH(output['instances'].pred_boxes.tensor.clone())
            scores = output['instances'].scores.clone()
            classes = output['instances'].pred_classes.clone()
        elif model == 'yolo8':
            boxes_xywh = from_XYXY_to_XYWH(output.boxes.xyxy.clone())
            scores = output.boxes.conf.clone()
            classes = output.boxes.cls.clone()
        else:
            raise ValueError

        if return_type == 'tensor':
            return boxes_xywh, scores, classes
        elif return_type == 'numpy':
            return boxes_xywh.cpu().numpy(), scores.cpu().numpy(), classes.cpu().numpy()
        elif return_type == 'list':
            return boxes_xywh.cpu().tolist(), scores.cpu().tolist(), classes.cpu().tolist()
        else:
            raise ValueError

    # batch predictions (list of instances for many images)
    for output in outputs:
        yield _convert_prediction(output, model=model, return_type=return_type)


def instances_to_coco_json(instances, img_id, track=None, track_ids=None, one_based_cats=True,
                           model_cat_names=None, remap_cat_names=None):
    if len(instances) == 0:
        results = []
    else:
        results = []
        for boxes_scores_classes in convert_predictions([instances], model='yolo8', return_type='numpy'):
            boxes, scores, classes = boxes_scores_classes

            if track_ids is not None:
                if track == SORT:
                    if len(track_ids) != 0:
                        from sort import sort
                        iou = sort.iou_batch(from_XYWH_to_XYXY(copy.copy(boxes)), track_ids[:, :4])
                        track_map = np.argmax(iou, axis=0)
                        boxes, scores, classes = boxes[track_map], scores[track_map], classes[track_map]
                    track_ids = track_ids[:, 4]
            else:
                track_ids = nans((len(boxes),))

            for box, score, cls, track_id in zip(boxes, scores, classes, track_ids):
                cls = int(cls) + (1 if one_based_cats else 0)
                assert model_cat_names is None or cls in range(1, len(model_cat_names) + 1)
                if remap_cat_names is not None:
                    cls = remap_cat_names[cls]
                result = {
                    "image_id": img_id,
                    "category_id": cls,
                    "bbox": box.tolist(),
                    "score": float(score)
                }
                if track is not None:
                    result["track_id"] = int(track_id) if not np.isnan(track_id) else None
                results.append(result)
    return results


def safe_int(x):
    assert int(x) == x
    return int(x)


def bbox_to_string(box, in_format='xywh', out_format='xyxy'):
    assert in_format in ['xywh', 'xyxy']
    assert out_format == 'xyxy'
    if in_format == 'xywh':
        box = copy.copy(box)
        box[2] += box[0]
        box[3] += box[1]
    # assert all([int(x) == x for x in box]), box
    return '-'.join(map(lambda x: f'{x:.5f}', box))


def convert_boxes(bbox, from_format, to_format):
    '''
    :param bbox: torch.tensor, np.ndarray, list, tuple
    :param from_format: ['xyxy', 'xywh']
    :param to_format: ['xyxy', 'xywh']
    :return: converted bbox
    '''

    assert from_format.lower() in ['xyxy', 'xywh']
    assert to_format.lower() in ['xyxy', 'xywh']

    if from_format.lower() == to_format.lower():
        return bbox
    else:
        original_type = type(bbox)
        single_box = not is_iterable(bbox[0])

        if single_box and original_type in [list, tuple, np.ndarray]:
            bbox = np.asarray([bbox])
        elif isinstance(bbox, np.ndarray):
            bbox = bbox.copy()
        elif isinstance(bbox, list) and isinstance(bbox[0], (list, tuple)):
            bbox = np.asarray(bbox)
        else:
            import torch
            if isinstance(bbox, torch.Tensor):
                bbox = bbox.clone()
            else:
                raise ValueError(f'Unexpected data structure: {bbox}')

        assert len(bbox.shape) == 2, (bbox.shape, bbox)
        assert bbox.shape[1] == 4, (bbox.shape, bbox)

        if from_format.lower() == 'xyxy' and to_format.lower() == 'xywh':
            bbox[:, 2] -= bbox[:, 0]
            bbox[:, 3] -= bbox[:, 1]
        elif from_format.lower() == 'xywh' and to_format.lower() == 'xyxy':
            bbox[:, 2] += bbox[:, 0]
            bbox[:, 3] += bbox[:, 1]
        else:
            raise ValueError

        if single_box:
            if original_type in [list, tuple]:
                bbox = original_type(bbox.flatten().tolist())
            else:
                bbox = bbox.flatten()

        return bbox


def from_XYXY_to_XYWH(boxes):
    '''
    :param boxes: torch.tensor, np.ndarray, list, tuple
    '''
    return convert_boxes(bbox=boxes, from_format='xyxy', to_format='xywh')


def from_XYWH_to_XYXY(boxes):
    '''
    :param boxes: torch.tensor, np.ndarray, list, tuple
    '''
    return convert_boxes(bbox=boxes, from_format='xywh', to_format='xyxy')


def mask_to_xyxy(masks: np.ndarray) -> np.ndarray:
    """
    Copyright (c) 2022 Roboflow
    MIT License
    https://github.com/roboflow/supervision

    Converts a 3D `np.array` of 2D bool masks into a 2D `np.array` of bounding boxes.

    Parameters:
        masks (np.ndarray): A 3D `np.array` of shape `(N, W, H)`
            containing 2D bool masks

    Returns:
        np.ndarray: A 2D `np.array` of shape `(N, 4)` containing the bounding boxes
            `(x_min, y_min, x_max, y_max)` for each mask
    """
    n = masks.shape[0]
    xyxy = np.zeros((n, 4), dtype=int)

    for i, mask in enumerate(masks):
        rows, cols = np.where(mask)

        if len(rows) > 0 and len(cols) > 0:
            x_min, x_max = np.min(cols), np.max(cols)
            y_min, y_max = np.min(rows), np.max(rows)
            xyxy[i, :] = [x_min, y_min, x_max, y_max]

    return xyxy


def mask_to_polygons(mask: np.ndarray, min_polygon_point_count=3) -> List[np.ndarray]:
    """
    Copyright (c) 2022 Roboflow
    MIT License
    https://github.com/roboflow/supervision
    Converts a binary mask to a list of polygons.

    Parameters:
        mask (np.ndarray): A binary mask represented as a 2D NumPy array of
            shape `(H, W)`, where H and W are the height and width of
            the mask, respectively.

    Returns:
        List[np.ndarray]: A list of polygons, where each polygon is represented by a
            NumPy array of shape `(N, 2)`, containing the `x`, `y` coordinates
            of the points. Polygons with fewer points than `MIN_POLYGON_POINT_COUNT = 3`
            are excluded from the output.
    """

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    return [
        np.squeeze(contour, axis=1)
        for contour in contours
        if contour.shape[0] >= min_polygon_point_count
    ]


def nans_like(arr):
    return np.ones_like(arr) * np.nan


def nans(shape):
    return np.ones(shape) * np.nan


def pickle_dump(obj, fn):
    with open(fn, 'wb') as f:
        pickle.dump(obj, f)


def pickle_load(fn):
    with open(fn, 'rb') as f:
        obj = pickle.load(f)
    return obj


def start_timing():
    return time.perf_counter()


def elapsed_time(start):
    return time.perf_counter() - start


def is_iterable(o, dict_allowed=True):
    return isinstance(o, Iterable) and not isinstance(o, string_types) and (dict_allowed or not isinstance(o, dict))


def bool2int(a):
    return a.dot(1 << np.arange(a.shape[-1]))


def get_device(device=None, model=None):
    assert model in ['yolo8', 'dt2']
    if device is None:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # mps overwhelms my Mac
        # else 'mps' if torch.backends.mps.is_available() \

    if device.startswith('cuda'):
        cuda_id = str(int(device.split(':')[1]) if ':' in device else 0)
    else:
        cuda_id = None

    if model == 'yolo8':
        if cuda_id is not None:
            device = cuda_id
        assert device in ['cpu', 'mps'] or str(int(device)) == device
    elif model == 'dt2':
        if cuda_id is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = cuda_id
            device = 'cuda'
        assert device in ['cuda', 'cpu', 'mps']

    return device


def load_model_cat_names_from_file(fn):
    dataset = read_json(fn, assert_correct=False, verbose=False)
    assert_categories_correct(dataset, strict=False)
    return get_category_names(dataset)


def savefig(filename, fig=None, bbox_inches='tight', format=None, *args, **kwargs):
    assert format is None

    if filename[-4:].lower() not in ['.png', '.jpg', '.pdf', '.eps', '.svg'] and filename[-3:].lower() not in ['.ps']:
        extensions = ['.pdf', '.png', '.svg']
    else:
        extensions = ['']

    for ext in extensions:
        filename_with_ext = '{}{}'.format(filename, ext)
        if fig is None:
            import matplotlib.pylab as plt
            plt.savefig(filename_with_ext, bbox_inches=bbox_inches, format=None, *args, **kwargs)
        else:
            fig.savefig(filename_with_ext, bbox_inches=bbox_inches, format=None, *args, **kwargs)


def is_image(fn, extensions=None):
    if extensions is None:
        extensions = ['jpg', 'jpeg', 'png', 'tiff', 'tif', 'gif', 'bmp']
    if isinstance(extensions, str):
        extensions = [extensions]
    return any([fn.lower().endswith(f'.{ext}') for ext in extensions])


def is_video(fn, extensions=None):
    if extensions is None:
        extensions = ['mp4', 'asf', 'mov', 'avi']
    if isinstance(extensions, str):
        extensions = [extensions]
    return any([fn.lower().endswith(f'.{ext}') for ext in extensions])


def read_images_from_dir(dir_name, basename_only=False, extensions=None):
    filter_func = is_image if extensions is None else lambda x: is_image(x, extensions=extensions)
    return read_files_from_dir(dir_name, filter_func=filter_func, basename_only=basename_only)


def read_videos_from_dir(dir_name, basename_only=False, extensions=None):
    filter_func = is_video if extensions is None else lambda x: is_video(x, extensions=extensions)
    return read_files_from_dir(dir_name, filter_func=filter_func, basename_only=basename_only)


def read_files_from_dir(dir_name, filter_func=None, basename_only=False, skip_hidden_files=True):
    return [
        (fn if basename_only else os.path.join(dir_name, fn)) for fn in sorted(os.listdir(dir_name))
        if os.path.isfile(os.path.join(dir_name, fn)) and ((not fn.startswith('.')) if skip_hidden_files else True) and
           (filter_func(os.path.join(dir_name, fn)) if filter_func is not None else True)
    ]


def subset_dataset_to_imgs(dataset, *imgs):
    if len(imgs) == 1 and os.path.isdir(imgs[0]):
        imgs = read_images_from_dir(imgs[0], basename_only=True)

    dataset['images'] = [img for img in dataset['images'] if img['file_name'] in imgs]
    dataset['annotations'] = filter_annotations_with_images(dataset)
    return dataset


def empty_coco_dataset(info=None, license_id=None):
    assert info is None or isinstance(info, (dict, str))
    dataset = {}
    if info is not None:
        if isinstance(info, dict):
            dataset['info'] = info
        elif isinstance(info, str):
            dataset['info'] = dict(description=info)

    if license_id is not None:
        dataset['licenses'] = [dict(id=license_id, name='', url='')]

    dataset['images'] = []
    dataset['annotations'] = []
    dataset['categories'] = []

    return dataset


def create_coco_dataset(*imgs, cat_names=None, license_id=None):
    from PIL import Image
    if len(imgs) == 1 and os.path.isdir(imgs[0]):
        imgs = read_images_from_dir(imgs[0])

    dataset = empty_coco_dataset(license_id=license_id)

    for i, fn in enumerate(imgs):
        with Image.open(fn) as img:
            dataset['images'].append(
                dict(id=i + 1, file_name=os.path.basename(fn), width=img.width, height=img.height,
                     license=license_id, flickr_url='', coco_url='', date_captured='')
            )

    if cat_names is not None:
        dataset['categories'] = [dict(id=i + 1, name=cat, supercategory='') for i, cat in enumerate(cat_names)]

    return dataset


def basename_without_ext(fn):
    return filename_without_ext(os.path.basename(fn))


def filename_without_ext(fn):
    return '.'.join(fn.split('.')[:-1])


def get_video_metadata(filename, manual_counting=False):
    assert os.path.isfile(filename), filename

    def count_manually(capture):
        total_frames = 0
        while capture.read()[0]:
            total_frames += 1
        return total_frames

    cap = cv2.VideoCapture(filename)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0 or fps > FAULTY_FPS_THR:
        print(f'WARNING: {os.path.basename(filename)} metadata suggests {int(fps)} frames per second. '
              'I will count the number of frames manually.')
        total_seconds = None
        duration = None
        fps = None
        total_frames = count_manually(cap)
    else:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not manual_counting else count_manually(cap)
        total_seconds = total_frames / fps
        duration = datetime.timedelta(seconds=total_seconds)
        fps = int(round(fps))
        total_frames = int(round(total_frames))

    return dict(
        total_frames=total_frames,
        total_seconds=total_seconds,
        fps=int(round(fps)) if fps is not None else fps,
        resolution=(width, height),
        duration=duration
    )


def frame_from_video(video):
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break


def frames_to_video(filenames, output_fn, fps, width, height, fourcc=None):
    if fourcc is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_fn, fourcc=fourcc, fps=fps, frameSize=(width, height))
    for fn in filenames:
        img = cv2.imread(fn)
        out.write(img)
    out.release()


def extract_frames(filename, output_dir, every_n_sec=None, every_n_frame=None, start_at=None, stop_at=None,
                   after_n_extracted=None, jump_after_n_extracted=None, frame_ids=None, zero_based=False,
                   fast_forward='auto', n_digits_fn=6, verbose=True):
    '''
    :param fast_forward: some videos do not support forwarding so a slower, frame-by-frame, approach is default
    '''
    if every_n_sec is not None:
        assert every_n_sec > 0
        assert every_n_frame is None
        assert start_at is None and stop_at is None
        assert not frame_ids
    if every_n_frame is not None:
        assert every_n_frame > 0
        assert every_n_sec is None
        assert not frame_ids
    if frame_ids:
        assert is_iterable(frame_ids)
    assert start_at is None or start_at >= 1
    assert stop_at is None or stop_at >= 1
    assert start_at is None or stop_at is None or stop_at >= start_at
    if after_n_extracted is None:
        after_n_extracted = -1
    if jump_after_n_extracted is None:
        jump_after_n_extracted = -1

    # Because internally everything starts at 0
    if start_at is not None:
        start_at -= 1
    else:
        start_at = 0
    # Because internally everything starts at 0
    if stop_at is not None:
        stop_at -= 1

    out_prefix = os.path.join(output_dir, basename_without_ext(filename))
    cap = cv2.VideoCapture(filename)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fast_forward == 'auto':
        if fps > FAULTY_FPS_THR:
            fast_forward = False
            print(f'WARNING: {os.path.basename(filename)} metadata suggests {int(fps)} frames per second. '
                  'I am disabling "fast_forward" to be safe.')
        else:
            fast_forward = True

    if jump_after_n_extracted > 0:
        assert fast_forward and every_n_frame is not None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_seconds = total_frames / fps

    # if fps is reasonable and there is more frames than n_digits_fn can handle, adjust
    if fps <= FAULTY_FPS_THR and total_frames > 10 ** n_digits_fn:
        new_n_digits_fn = int(np.ceil(np.log10(total_frames)))
        assert new_n_digits_fn > n_digits_fn
        n_digits_fn = new_n_digits_fn

    if every_n_sec is None and every_n_frame is None:
        every_n_frame = 1

    if not fast_forward and every_n_frame is None:
        # the slow mode only supports moving by frames
        every_n_frame = int(round(fps * every_n_sec))
        print('WARNING: "fast_forward" disabled, consider using "every_n_frame" instead of "every_n_sec"')

    if frame_ids:
        assert every_n_frame == 1
        assert every_n_sec is None
        assert after_n_extracted == -1 and jump_after_n_extracted == -1

    # skip_n_frames is 0 if no skipping is done
    i = start_at if fast_forward else 0
    f, success, skip = -1, True, False
    out_files = []
    if verbose:
        print('Extracting and saving the frames...')
    n_extracted = 0
    while success and (stop_at is None or f + (every_n_frame if fast_forward else 1) <= stop_at):
        if fast_forward:
            if every_n_sec is not None:
                cap.set(cv2.CAP_PROP_POS_MSEC, (i * 1000))
                f = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                i += every_n_sec
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                f = i
                i += every_n_frame
                if jump_after_n_extracted > 0 and n_extracted > 0 and n_extracted % after_n_extracted == 0:
                    i += jump_after_n_extracted
        else:
            # skip_n_frames is 0 if no skipping is done
            skip = ((i - start_at) % every_n_frame) != 0 or i < start_at
            f += 1
            i += 1

        success, img = cap.read()

        if success and not skip and (frame_ids is None or (f + (0 if zero_based else 1)) in frame_ids):
            n_extracted += 1
            # from 1 November 2023 start numbering frames with 000001 (instead of 000000)
            out_files.append(f"{out_prefix}.frame_{f + (0 if zero_based else 1):0{n_digits_fn}d}.jpg")
            cv2.imwrite(out_files[-1], img)
            if len(out_files) % 100 == 0 and verbose:
                print(f'Saved {len(out_files)} frames so far...')

    if not fast_forward and total_frames != i - 1:
        # No need to raise this warning:
        # print(f'WARNING: {os.path.basename(fn)} metadata suggests {total_frames} '
        #       f'frames in the video but only {i - 1} frames present')
        if every_n_sec is not None:
            raise ValueError(f'Because of corrupted metadata use "every_n_frame" instead of "every_n_sec"')
        total_frames = i - 1
        total_seconds = None
        fps = None
    else:
        if every_n_frame is not None:
            if len(out_files) != np.ceil(total_frames / float(every_n_frame)):
                print(
                    f'WARNING: Saved {len(out_files)} frames but was expecting {int(np.ceil(total_frames / float(every_n_frame)))} frames in total.')
        if every_n_sec is not None:
            if abs(len(out_files) - (total_seconds / float(every_n_sec))) >= 1:
                print(
                    f'WARNING: Saved {len(out_files)} frames but was expecting approx. {total_seconds / float(every_n_sec)} frames in total.')
    if verbose:
        print(f'{len(out_files)} frames saved')
    metadata = dict(total_frames=total_frames, total_seconds=total_seconds,
                    fps=int(round(fps)) if fps is not None else fps)
    return out_files, metadata


def extract_frames_for_dataset(images, video_fn, output_dir, copy_to_tmp=False, overwrite=False, zero_based=False,
                               verbose=True):
    os.makedirs(os.path.join(output_dir), exist_ok=True)
    frames_ids, frame_fns = [], []
    for img in images:
        assert img['file_name'] == os.path.basename(img['file_name']), \
            f'Paths not allowed in image filenames: {img["file_name"]}'
        if overwrite or not os.path.exists(os.path.join(output_dir, img['file_name'])):
            frame_fns.append(img['file_name'])
            frame_str = get_frame_string_id(img['file_name'])
            frame_num = int(frame_str.split('_')[1])
            assert zero_based or frame_num != 0, 'Frame 0 found, not supported'
            frames_ids.append(frame_num)

    if len(frames_ids) != 0:
        with tempfile.TemporaryDirectory(dir=output_dir) as tmp_dir:
            if copy_to_tmp:
                print(f'Copying {video_fn} to a temporary location for processing...')
                temp_video_fn = os.path.join(tmp_dir, os.path.basename(video_fn))
                shutil.copy(video_fn, temp_video_fn)
                video_fn = temp_video_fn

            out_files, _ = extract_frames(
                filename=video_fn, output_dir=output_dir, frame_ids=frames_ids, zero_based=zero_based,
                fast_forward=False, start_at=min(frames_ids) + 1, stop_at=max(frames_ids) + 1, n_digits_fn=6,
                verbose=verbose > 0
            )
            assert sorted(frame_fns) == sorted([os.path.basename(f) for f in out_files]), \
                f'Extracted images have different names than JSON annotation file: {len(frame_fns)} {len([os.path.basename(f) for f in out_files])}'
            if verbose:
                print(f'EXTRACTED {len(out_files)} frames\n')
    else:
        if verbose:
            print(f'Nothing to do, all required frames exist\n')

    return frames_ids, frame_fns


def get_frame_string_id(filename):
    re_match = re.search(r'(^|\.)frame_[0-9]+\.', filename)
    assert re_match is not None
    frame_str = re_match.group().strip('.')
    assert re.match(r'^frame_[0-9]+$', frame_str)
    return frame_str


def fill_in_width_height(images, load=True, h=None, w=None):
    for img in images:
        if load:
            assert not h and not w
            image = read_image(img['file_name'])
            img['height'], img['width'] = image.shape[:2]
        else:
            img['height'], img['width'] = h, w


def fill_in_areas(annotations, force=False, fill_in_ids=False, fill_in_iscrowd=False):
    for i, ann in enumerate(annotations):
        assert len(ann['bbox']) == 4
        if force or 'area' not in ann:
            ann['area'] = ann['bbox'][2] * ann['bbox'][3]
        if fill_in_ids:
            assert 'id' not in ann, f'When fill_in_ids=True there must be no "id" field: {ann}'
            ann['id'] = i + 1
        if fill_in_iscrowd:
            assert 'iscrowd' not in ann, f'When fill_in_iscrowd=True there must be no "iscrowd" crowd {ann}'
            ann['iscrowd'] = False


def maybe_create_gunzipped_filename(fn):
    if fn.endswith('.gz'):
        if os.path.exists(fn[:-3]):
            return fn[:-3]
        else:
            tmp = tempfile.NamedTemporaryFile(mode='w+')
            with gzip.open(fn, mode='rt', encoding='UTF-8') as f:
                shutil.copyfileobj(f, tmp)
            return tmp
    else:
        return fn


def open_maybe_gzipped(fn):
    if not os.path.exists(fn):
        if fn.endswith('.gz') and os.path.exists(fn[:-3]):
            fn = fn[:-3]
        elif not fn.endswith('.gz') and os.path.exists(f'{fn}.gz'):
            fn = f'{fn}.gz'
    open_func = gzip.open if fn.endswith('.gz') else open
    return fn, open_func


def file_or_gzip_exists(fn):
    return os.path.isfile(fn) or os.path.isfile(f'{fn}.gz') or (fn.endswith('.gz') and os.path.isfile(fn[:-3]))


def json_datasets_equal(dataset1, dataset2, verbose=True, **read_kwargs):
    if isinstance(dataset1, str):
        dataset1 = read_json(dataset1, verbose=False, **read_kwargs)
    if isinstance(dataset2, str):
        dataset2 = read_json(dataset2, verbose=False, **read_kwargs)

    datasets_equal = True
    for field in ['images', 'categories', 'annotations']:
        match = sorted(dataset1[field], key=lambda x: x['id']) == \
                sorted(dataset2[field], key=lambda x: x['id'])
        datasets_equal = datasets_equal and match
        if verbose and not match:
            print(f'INFO: {field} do not match ({len(dataset1[field])} vs {len(dataset2[field])} {field}).')

    return datasets_equal


def read_json(fn, assert_correct=True, only_preds=False, only_imgs=False, add_missing_areas=False, fill_in_images=False,
              fix_zero_ann_ids=False, allow_zero_area_boxes=False, fix_to_basenames=False, fix_image_ids=False,
              change_to_zero_based_frames=False, fps_mod=None, verbose=True):
    fn, open_func = open_maybe_gzipped(fn)
    with open_func(fn, mode='rt', encoding='UTF-8') as f:
        dataset = json.load(f)

    if only_preds and isinstance(dataset, list):
        dataset = dict(annotations=dataset)
    if add_missing_areas:
        if not only_imgs or 'annotations' in dataset:
            fill_in_areas(dataset['annotations'])
    if fix_zero_ann_ids and any([ann['id'] == 0 for ann in dataset['annotations'] if 'id' in ann]):
        print(f'Fixing zero annotation IDs {fn}')
        for ann in dataset['annotations']:
            ann['id'] += 1
        assert all([ann['id'] >= 1 for ann in dataset['annotations']])
    if fill_in_images and 'images' not in dataset:
        for ann in dataset['annotations']:
            ann['image_id'] = ann['image_id'].replace('.mp4.frame_', '.frame_')
        filenames = sorted(set([ann['image_id'] for ann in dataset['annotations']]))
        assert all([isinstance(fn, str) for fn in filenames])
        dataset['images'] = [dict(id=fn, file_name=fn) for fn in filenames]
    if fix_to_basenames:
        use_basenames(dataset['images'])
    if fix_image_ids:
        reid_images(dataset)
    if change_to_zero_based_frames:
        anns = get_annotations_dict(dataset)
        for img in dataset['images']:
            assert img['id'] == img['file_name']
            spl_main = img['file_name'].split('.frame_')
            spl_ext = spl_main[-1].split('.')
            frame_id = spl_ext[0]
            spl_main = '.frame_'.join(spl_main[:-1])
            spl_ext = '.'.join(spl_ext[1:])
            assert f'{spl_main}.frame_{int(frame_id):0{len(frame_id)}d}.{spl_ext}' == img['file_name']
            assert int(frame_id) > 0, f"Cannot change to 0-based: {img['file_name']}"
            new_file_name = f'{spl_main}.frame_{int(frame_id) - 1:0{len(frame_id)}d}.{spl_ext}'
            for ann in anns[img['id']]:
                ann['image_id'] = new_file_name
            img['id'] = new_file_name
            img['file_name'] = new_file_name
    if fps_mod is not None and fps_mod != 1:
        imgs = []
        for img in dataset['images']:
            fn = img['file_name']
            assert fn.count('.frame_') == 1
            assert fn.lower().endswith('.jpg') or fn.lower().endswith('.png')
            if int(fn.split('.frame_')[1][:-4]) % fps_mod == 0:
                imgs.append(img)
        dataset['images'] = imgs
        dataset['annotations'] = filter_annotations_with_images(dataset)
    if assert_correct:
        assert_dataset_correct(dataset, only_preds=only_preds, only_imgs=only_imgs,
                               allow_zero_area_boxes=allow_zero_area_boxes)
    if verbose:
        if 'images' in dataset:
            print('images:', len(dataset['images']))
        if not only_imgs or 'annotations' in dataset:
            print(f'{"annotations" if not only_preds else "predictions"}:', len(dataset['annotations']))
        if 'categories' in dataset:
            cats = get_category_ids(dataset, as_dict=True, strict=assert_correct)
            print('categories:', cats.values())
            if not only_imgs:
                categories_counts(dataset, assert_correct=assert_correct, verbose=True)
    return dataset


def read_tracking_json(fn, assert_correct=False, verbose=False, categories='ventilation', fix_track_ids=True,
                       fix_areas=True, fix_width_height_by_loading=False, fix_width=1280, fix_height=960,
                       fill_in_ids=True, fill_in_iscrowd=True, fix_to_basenames=True, fix_image_ids=True,
                       fill_in_images=True, change_to_zero_based_frames=False):
    if categories == 'ventilation':
        categories = ["open", "closed", "DJ"]
    dataset = read_json(fn, assert_correct=assert_correct, fill_in_images=fill_in_images,
                        fix_to_basenames=fix_to_basenames, fix_image_ids=fix_image_ids,
                        change_to_zero_based_frames=change_to_zero_based_frames, verbose=verbose)
    dataset['categories'] = [dict(id=i + 1, name=c, supercategory="") for i, c in enumerate(categories)]
    if fix_areas:
        fill_in_areas(dataset['annotations'], fill_in_ids=fill_in_ids, fill_in_iscrowd=fill_in_iscrowd)
    if fix_width_height_by_loading or fix_width is not None or fix_height is not None:
        fill_in_width_height(dataset['images'], load=fix_width_height_by_loading, h=fix_height, w=fix_width)
    # if fix_track_ids:
    #     for ann in dataset['annotations']:
    #         video = ann['image_id'].split('.frame_')[0]
    #         ann['track_id'] = f"{video}_{ann['track_id']}"

    return dataset


def read_imgs_and_resize(dataset, img_dir, resize, group_splitter='_', n_parts=1):
    metadata = []
    groups = get_filename_groups(dataset['images'], split_char=group_splitter, n_parts=n_parts)
    img_arr = np.zeros((len(dataset['images']), resize * resize * 3), dtype=np.uint8)
    for i, (img, group) in enumerate(zip(dataset['images'], groups)):
        _img = read_image(os.path.join(img_dir, img['file_name']))
        _img = cv2.resize(_img, (resize, resize))
        img_arr[i] = _img.flatten()
        metadata.append([img['id'], img['file_name'], group])
    return img_arr, np.asarray(metadata)


def select_fish_from_GT(dataset, ground_truth, iou_thr=0.33, allow_missing_dt_frames=False, fix_DJs=False,
                        remove_unmatched_dt_boxes=True, verbose=True):

    # SUBSET TO ONLY GROUND TRUTH IMAGES
    gt_img_fns = [img['file_name'] for img in ground_truth['images']]
    assert len(gt_img_fns) == len(set(gt_img_fns))
    dt_img_fns = [img['file_name'] for img in dataset['images']]
    assert len(dt_img_fns) == len(set(dt_img_fns))
    assert allow_missing_dt_frames or set(dt_img_fns).issuperset(gt_img_fns), set(gt_img_fns).difference(dt_img_fns)
    dataset['images'] = [img for img in dataset['images'] if img['file_name'] in gt_img_fns]
    if len(set(gt_img_fns).difference(dt_img_fns)) != 0:
        assert allow_missing_dt_frames
        if verbose:
            print('WARNING: some frames have no detections and are missing:', set(gt_img_fns).difference(dt_img_fns))
            print(len(dataset['images']))
        add = [img for img in ground_truth['images'] if img['file_name'] in set(gt_img_fns).difference(dt_img_fns)]
        for img in add:
            img.pop('license')
        dataset['images'].extend(add)
        if verbose:
            print(add)
            print(len(dataset['images']))
    assert set([img['file_name'] for img in dataset['images']]) == set(gt_img_fns)

    # SUBSET TO ANNOTATIONS IN THE GROUND TRUTH IMAGES
    dataset['annotations'] = filter_annotations_with_images(dataset)

    # EVALUATE TO MATCH GT-BOXES TO DT-BOXES
    results = evaluate(gt_coco=ground_truth, dt_coco=dataset, iouThrs=iou_thr, useCats=False, verbose=False)

    # CREATE DT-TO-GT DICTIONARY
    iouThr_idx = [i for i, t in enumerate(results.eval._paramsEval.iouThrs) if t == iou_thr][0]
    dt_to_gt = {}
    for img in results.eval.evalImgs:
        if img is not None:
            assert not np.any(img['gtIgnore'])
            assert not np.any(img['dtIgnore'])
            gtIds = np.asarray(img['gtIds'])
            gtMatches = img['gtMatches'][iouThr_idx]
            dtIds = np.asarray(img['dtIds'])
            dtMatches = img['dtMatches'][iouThr_idx]
            assert set(dtIds[dtMatches != 0]) == set(gtMatches[gtMatches != 0])
            for gt_id, dt_id in zip(gtIds[gtMatches != 0], gtMatches[gtMatches != 0]):
                dt_to_gt[dt_id] = gt_id

    # EACH GT-BOX ONLY MATCHED TO ONE DT-BOX
    assert len(list(dt_to_gt.values())) == len(set(list(dt_to_gt.values())))

    if remove_unmatched_dt_boxes:
        # KEEP ONLY ANNOTATIONS THAT MATCH A GT-BOX
        dataset['annotations'] = [ann for ann in dataset['annotations'] if ann['id'] in dt_to_gt.keys()]

    # CHANGE DJs TO OPEN (hacky imputation)
    for ann in dataset['annotations']:
        if ann['id'] in dt_to_gt:
            ann['gt_id'] = dt_to_gt[ann['id']]
        if fix_DJs:
            if ann['category_id'] not in [1, 2]:
                if verbose:
                    print('FIXING DJ TO OPEN')
                assert ann['category_id'] == 3, ann
                ann['category_id'] = 1


def update_tracker(tracker, det_per_img, image,
                   aspect_ratio_thresh=10, min_box_area=1, box_format='xywh'):
    '''

    :param tracker: BoTSORT tracker object
    :param det_per_img: (n,6) array of detections per image [[x, y, x, y, conf, cls], ...]
                        det_per_img WILL BE MODIFIED IN PLACE
    :param image: (h, w, c) array of pixels in the BGR (cv2) format
    :return: track_ids
    '''

    if len(det_per_img):
        if box_format.lower() == 'xywh':
            det_per_img[:, :4] = from_XYWH_to_XYXY(det_per_img[:, :4])
        else:
            assert box_format.lower() == 'xyxy'
    else:
        det_per_img = []

    output_stracks = tracker.update(det_per_img, image)

    # REMOVE THIS IN THE FUTURE
    for t in output_stracks:
        vertical = t.tlwh[2] / t.tlwh[3] > aspect_ratio_thresh
        too_small = t.tlwh[2] * t.tlwh[3] <= min_box_area
        assert not too_small and not vertical, det_per_img

    tracks = np.asarray(
        [t.tlwh.tolist() + [t.score, t.cls, t.track_id, t.idx] for t in output_stracks],
        dtype=np.float32
    )
    assert len(tracks) == 0 or len(tracks) == len(set(tracks[:, -1])), "Not a unique matching to detections"

    return tracks


def update_tracker_with_detection(tracker, det_per_img, img, iou_func=None, assert_iou=0.5):
    if len(det_per_img) != 0:
        tracks = update_tracker(
            tracker=tracker,
            det_per_img=np.asarray([
                np.asarray(det['bbox'] + [det['score'], -1], dtype=np.float32) for det in det_per_img
            ]),
            image=img
        )

        if len(tracks) != 0:
            # tracks: [[x, y, w, h, score, class, track_id, idx], ...]
            for idx, t in zip(tracks[:, -1].astype(int), tracks[:, :-1]):
                assert abs(det_per_img[idx]['score'] - t[4]) < 1e-3, (det_per_img[idx], t)
                if iou_func is not None:
                    iou = iou_func(np.asarray([from_XYWH_to_XYXY(det_per_img[idx]['bbox'])]),
                                   np.asarray([from_XYWH_to_XYXY(t[:4])]))
                    assert np.all(iou > assert_iou), (det_per_img[idx], t, iou)
                det_per_img[idx]['track_id'] = int(t[6])  # update id
                det_per_img[idx]['bbox'] = t[:4].tolist() # update box


def get_track_config(track_high_thr=None, track_low_thr=None, new_track_thr=None, track_buffer=None, track_match_thr=None):
    track_cfg = SimpleNamespace(
        track_high_thresh=track_high_thr if track_high_thr is not None else 0.5,
        track_low_thresh=track_low_thr if track_low_thr is not None else 0.1,
        new_track_thresh=new_track_thr if new_track_thr is not None else 0.6,
        track_buffer=track_buffer if track_buffer is not None else 30,
        match_thresh=track_match_thr if track_match_thr is not None else 0.8,
        cmc_method='sparseOptFlow',
        name='DT2',
        ablation=False,
        with_reid=False,
        proximity_thresh=0.5,
        appearance_thresh=0.25,
        mot20=False
    )
    return track_cfg


def track_GT(ground_truth, track_iou=0.1, assert_iou=0.4, img_dir=None):
    return track_GT_BOTSORT(ground_truth, track_iou=track_iou, assert_iou=assert_iou, img_dir=img_dir)


def track_GT_BOTSORT(ground_truth, img_dir, track_iou=0.5, assert_iou=0.4):
    from botsort.tracker.mc_bot_sort import BoTSORT
    from sort import sort
    track_cfg = get_track_config(new_track_thr=0, track_buffer=1, track_match_thr=track_iou)
    tracker = BoTSORT(args=track_cfg)
    annot_dict = get_annotations_dict(ground_truth)

    for img_id in annot_dict.keys():
        for ann in annot_dict[img_id]:
            ann['score'] = 0.99
        img = read_image(os.path.join(img_dir, [img['file_name'] for img in ground_truth['images'] if img['id'] == img_id][0]))
        update_tracker_with_detection(tracker=tracker, det_per_img=annot_dict[img_id], img=img, iou_func=sort.iou_batch, assert_iou=assert_iou)

    last_track_id = annot_dict[img_id][-1]['track_id']
    return last_track_id


def track_GT_SORT(ground_truth, track_iou=0.1, assert_iou=0.4, img_dir=None):
    from sort import sort
    annot_dict = get_annotations_dict(ground_truth)
    tracker = sort.Sort(max_age=1, min_hits=0, iou_threshold=track_iou)

    for img_id in annot_dict:
        boxes = np.asarray([(from_XYWH_to_XYXY(ann['bbox']) + [0.99]) for ann in annot_dict[img_id]], dtype=float)
        tracks = tracker.update(boxes)
        assert len(tracks) == len(boxes), (img_id, boxes)

        iou = sort.iou_batch(tracks[:, :4], boxes[:, :4])
        track_map = np.argmax(iou, axis=0)

        for ann, track in zip(annot_dict[img_id], tracks[track_map]):
            check_iou = sort.iou_batch(track[:4].reshape((1, 4)), np.asarray([from_XYWH_to_XYXY(ann['bbox'])]))
            assert check_iou.shape == (1, 1) and check_iou[0, 0] > assert_iou, check_iou
            assert int(track[4]) == float(track[4])
            ann['track_id'] = int(track[4])
    last_track_id = ann['track_id']
    return last_track_id


def evaluate_tracks(dataset, ground_truth):
    assert sorted([img['file_name'] for img in dataset['images']]) == sorted(
        [img['file_name'] for img in ground_truth['images']])
    assert all(['track_id' in ann.keys() and ann['track_id'] is not None for ann in dataset['annotations']])
    assert all(['track_id' in ann.keys() and ann['track_id'] is not None for ann in ground_truth['annotations']])

    gt_ann, dt_ann = {}, {}
    gt_tracks, dt_tracks = defaultdict(list), defaultdict(list)
    gt_to_dt = {}
    dt_id_to_track = {}
    gt_track_to_dt_track = {}

    for ann in ground_truth['annotations']:
        assert ann['category_id'] in [1, 2], ann
        gt_ann[ann['id']] = ann
        gt_tracks[ann['track_id']].append(ann['id'])

    n_DJs = 0
    for ann in dataset['annotations']:
        if ann['category_id'] not in [1, 2]:
            n_DJs += 1
        dt_ann[ann['id']] = ann
        dt_tracks[ann['track_id']].append(ann['id'])
        dt_id_to_track[ann['id']] = ann['track_id']
        assert ann['gt_id'] not in gt_to_dt
        gt_to_dt[ann['gt_id']] = ann['id']

    if n_DJs != 0:
        print(f'WARNING: There were {n_DJs} DJs among detections, but no DJs among ground truth')

    for gt_track_id in gt_tracks:
        gt_track_to_dt_track[gt_track_id] = set([])
        for gt_id in gt_tracks[gt_track_id]:
            if gt_id in gt_to_dt:
                dt_id = gt_to_dt[gt_id]
                dt_track_id = dt_id_to_track[dt_id]
                gt_track_to_dt_track[gt_track_id].add(dt_track_id)

        # keep just the longest detection track matching this ground truth
        if len(gt_track_to_dt_track[gt_track_id]):
            gt_track_to_dt_track[gt_track_id] = np.asarray(sorted(gt_track_to_dt_track[gt_track_id]))[
                np.argmax([len(dt_tracks[dt_track_id]) for dt_track_id in sorted(gt_track_to_dt_track[gt_track_id])])]
        else:
            del gt_track_to_dt_track[gt_track_id]

    fish_tracked = len(gt_track_to_dt_track) / len(gt_tracks)

    for gt_track_id in gt_tracks:
        if gt_track_id in gt_track_to_dt_track and len(dt_tracks[gt_track_to_dt_track[gt_track_id]]) > len(gt_tracks[gt_track_id]):
            print(f'WARNING: Test fish {gt_track_id} has more detections ({len(dt_tracks[gt_track_to_dt_track[gt_track_id]])}) than ground truth ({len(gt_tracks[gt_track_id])})')

    Q2, TPR, TNR, MD = {}, {}, {}, {}
    gt_ts, dt_ts = {}, {}
    for gt_track_id in gt_tracks:
        if gt_track_id in gt_track_to_dt_track:
            Q2[gt_track_id], TPR[gt_track_id], TNR[gt_track_id], MD[gt_track_id] = [], [], [], []
            gt_ts[gt_track_id], dt_ts[gt_track_id] = [], []
            dt_track_id = gt_track_to_dt_track[gt_track_id]
            for gt_id in gt_tracks[gt_track_id]:
                gt_ts[gt_track_id].append(gt_ann[gt_id])
                if gt_id in gt_to_dt:
                    dt_id = gt_to_dt[gt_id]
                    if dt_id in dt_tracks[dt_track_id]:
                        dt_ts[gt_track_id].append(dt_ann[dt_id])
                        MD[gt_track_id].append(False)
                        Q2[gt_track_id].append(gt_ann[gt_id]['category_id'] == dt_ann[dt_id]['category_id'])
                        if gt_ann[gt_id]['category_id'] == 1:
                            TPR[gt_track_id].append(gt_ann[gt_id]['category_id'] == dt_ann[dt_id]['category_id'])
                        if gt_ann[gt_id]['category_id'] == 2:
                            TNR[gt_track_id].append(gt_ann[gt_id]['category_id'] == dt_ann[dt_id]['category_id'])
                    else:
                        MD[gt_track_id].append(True)
                        dt_ts[gt_track_id].append(None)
                else:
                    MD[gt_track_id].append(True)
                    dt_ts[gt_track_id].append(None)

            assert len(gt_ts[gt_track_id]) == len(MD[gt_track_id])
            MD[gt_track_id] = np.sum(MD[gt_track_id]) / len(MD[gt_track_id])
            Q2[gt_track_id] = np.sum(Q2[gt_track_id]) / len(Q2[gt_track_id])
            TPR[gt_track_id] = (np.sum(TPR[gt_track_id]) / len(TPR[gt_track_id])) if len(TPR[gt_track_id]) != 0 else np.nan
            TNR[gt_track_id] = (np.sum(TNR[gt_track_id]) / len(TNR[gt_track_id])) if len(TNR[gt_track_id]) != 0 else np.nan
        else:
            MD[gt_track_id] = 1.0
            Q2[gt_track_id] = 0.0
            TPR[gt_track_id] = 0.0
            TNR[gt_track_id] = 0.0
            gt_ts[gt_track_id] = [gt_ann[gt_id] for gt_id in gt_tracks[gt_track_id]]
            dt_ts[gt_track_id] = [None] * len(gt_tracks[gt_track_id])
            print(f'WARNING: Test fish {gt_track_id} was not detected at all')

        assert len(gt_ts[gt_track_id]) == len(dt_ts[gt_track_id])

    return fish_tracked, MD, Q2, TPR, TNR, gt_ts, dt_ts


def read_CVAT_json(task_dir):
    categories = []
    task_name = None
    with open(os.path.join(task_dir, 'task.json')) as f:
        task = json.load(f)
        task_name = task['name']
        categories = [l['name'] for l in task['labels']]

    with open(os.path.join(task_dir, 'annotations.json')) as f:
        ann = json.load(f)
        assert len(ann) == 1
        for s in ann[0]['shapes']:
            s['points']
            s['frame']
            s['label']


def categories_counts(dataset, assert_correct=True, verbose=True):
    cats = get_category_ids(dataset, as_dict=True, strict=assert_correct)
    n_per_cat = {cat_id: len([ann for ann in dataset['annotations'] if ann['category_id'] == cat_id]) for cat_id in
                 cats.keys()}
    # idx_max = list(n_per_cat.keys())[(np.argmax(n_per_cat.values()))]
    ratios = {}
    for cat_id in cats.keys():
        try:
            ratios[cat_id] = n_per_cat[cat_id] / len(dataset['annotations'])
        except:
            ratios[cat_id] = None
        if verbose:
            print(f"{cats[cat_id]}: {n_per_cat[cat_id]} (ratio: {ratios[cat_id]})")
    return cats, n_per_cat, ratios


def save_json(dataset, fn, assert_correct=True, only_preds=False, only_imgs=False, indent=2, sort_keys=False,
              compress='auto'):
    if assert_correct:
        assert_dataset_correct(dataset, only_preds=only_preds, only_imgs=only_imgs)

    open_func = gzip.open if compress is True or (compress == 'auto' and fn.endswith('.gz')) else open
    if compress is True and not fn.endswith('.gz'):
        fn = f'{fn}.gz'
    with open_func(fn, mode='wt', encoding='UTF-8') as f:
        json.dump(dataset, f, indent=indent, sort_keys=sort_keys)


def save_yaml(data, fn, default_flow_style=False, sort_keys=False):
    with open(fn, 'wt') as f:
        yaml.dump(data, f, default_flow_style=default_flow_style, sort_keys=sort_keys)


def read_yaml(fn):
    with open(fn, 'r') as f:
        return yaml.safe_load(f)


def save_yacs(cfg, fn, sort_keys=False):
    with open(fn, 'wt') as f:
        print(cfg.dump(sort_keys=sort_keys), file=f, sep='')


def package_for_cvat(task_name, frames_fns, labels, use_cache, output_dir, dataset=None):
    if dataset is not None:
        assert [os.path.basename(fn) for fn in frames_fns] == sorted([img['file_name'] for img in dataset['images']])
    for data, fn in [
        [create_cvat_task(task_name=task_name, n_frames=len(frames_fns), labels=labels, use_cache=use_cache),
         os.path.join(output_dir, 'task.json')],

        [create_cvat_annotations(dataset=dataset),
         os.path.join(output_dir, 'annotations.json')],

        [create_cvat_manifest(filenames=frames_fns),
         (os.path.join(output_dir, 'data', 'manifest.jsonl'),
          os.path.join(output_dir, 'data', 'index.json'))],
    ]:
        print(fn)
        if not isinstance(fn, tuple):
            with open(fn, mode='wt', encoding='UTF-8') as f:
                json.dump(data, f, separators=(',', ':'))
        else:
            with open(fn[0], mode='wt', encoding='UTF-8') as f:
                index, i, offset = {}, 0, 0
                for json_line in data:
                    serialized = json.dumps(json_line, separators=(',', ':'))
                    print(serialized, file=f)
                    if 'name' in json_line:
                        index[str(i)] = int(offset)
                        i += 1
                    offset += len(serialized) + 1

            with open(fn[1], mode='wt', encoding='UTF-8') as f:
                json.dump(index, f, separators=(',', ':'))


def create_cvat_annotations(dataset=None):
    '''
    [{
        "version":0,
        "tags":[],
        "shapes":
          [
            {"type":"rectangle","occluded":false,"outside":false,"z_order":0,"rotation":0.0,
              "points":[702.9924843161152,212.21235574999992,761.9995749182835,266.6804393827715],
              "frame":0,"group":0,"source":"manual","attributes":[],"elements":[],"label":"J"},
            {"type":"rectangle","occluded":false,"outside":false,"z_order":0,"rotation":0.0,
              "points":[1128.5243876971417,163.41803082897604,1173.914457391118,205.403845295903],
              "frame":0,"group":0,"source":"manual","attributes":[],"elements":[],"label":"J"},
            {"type":"rectangle","occluded":false,"outside":false,"z_order":0,"rotation":0.0,
              "points":[1147.8151673170814,152.0705134054806,1198.8789957228037,203.13434181120283],
              "frame":2,"group":0,"source":"manual","attributes":[],"elements":[],"label":"J"},
            {"type":"rectangle","occluded":false,"outside":false,"z_order":0,"rotation":0.0,
              "points":[347.8151889607525,630.9357486769295,405.68752782057163,677.460570113255],
              "frame":2,"group":0,"source":"manual","attributes":[],"elements":[],"label":"J"}
          ],
        "tracks":[]
    }]
    '''

    annot_dict = {"version": 0, "tags": [], "shapes": [], "tracks": []}

    if dataset is not None:
        cat_dict = get_category_ids(dataset, as_dict=True)
        img_id_to_frame_number = {img['id']: i for i, img in enumerate(dataset['images'])}
        for annot in dataset['annotations']:
            assert annot['image_id'] > 0, 'Expected 1-based image-ids'
            annot_dict["shapes"].append(
                {"type": "rectangle", "occluded": False, "outside": False, "z_order": 0, "rotation": 0.0,
                 "points": from_XYWH_to_XYXY(annot['bbox']), "frame": img_id_to_frame_number[annot['image_id']],
                 "group": 0, "source": annot.get("source", "auto"), "attributes": [], "elements": [],
                 "label": cat_dict[annot["category_id"]]}
            )

    return [annot_dict]


def create_cvat_task(task_name, n_frames, labels, colors=None, label_type="rectangle",
                     start_frame=None, stop_frame=None, image_quality=100, chunk_size=60, use_cache=True):
    if start_frame is None:
        start_frame = 0
    if stop_frame is None:
        stop_frame = n_frames - 1
    assert start_frame < n_frames
    assert stop_frame < n_frames
    assert start_frame <= stop_frame
    if colors is None:
        from seaborn import color_palette
        colors = color_palette(palette='bright').as_hex()
        while len(colors) < len(labels):
            colors.extend(colors)

    return {
        "name": task_name,
        "bug_tracker": "",
        "status": "annotation",
        "subset": "",
        "labels": [
            {"name": label, "color": color, "attributes": [], "type": label_type, "sublabels": []}
            for label, color in zip(labels, colors)
        ],
        "version": "1.0",
        "data": {
            "chunk_size": chunk_size, "image_quality": image_quality,
            "start_frame": start_frame, "stop_frame": stop_frame,
            "storage_method": "cache" if use_cache else 'file_system', "storage": "local",
            "sorting_method": "lexicographical", "chunk_type": "imageset", "deleted_frames": []
        },
        "jobs": [
            {"start_frame": start_frame, "stop_frame": stop_frame, "status": "annotation"}
        ]
    }


def create_cvat_manifest(filenames):
    json_lines = [
        {"version": "1.1"},
        {"type": "images"}
    ]

    for fn in filenames:
        height, width = read_image(fn).shape[:2]
        json_lines.append(
            {
                "name": basename_without_ext(fn), "extension": fn.split('.')[-1],
                "width": width, "height": height, "meta": {"related_images": []}
            }
        )

    return json_lines


def get_image_ids(dataset, id_field='id'):
    return [img[id_field] for img in dataset['images']]


def sort_predictions_by_score(predictions, key='score'):
    idx = np.argsort([pred[key] for pred in predictions])[::-1]
    return list(np.asarray(predictions)[idx])


def sort_categories(dataset):
    old_id_to_name = get_category_ids(dataset, as_dict=True, strict=False)
    name_to_new_id, new_id_to_name = {}, {}
    for i, name in enumerate(sorted(old_id_to_name.values())):
        name_to_new_id[name] = i + 1
        new_id_to_name[i + 1] = name

    dataset['categories'] = [dict(id=cat_id, name=new_id_to_name[cat_id], supercategory="") for cat_id in
                             new_id_to_name]
    for ann in dataset['annotations']:
        ann['category_id'] = name_to_new_id[old_id_to_name[ann['category_id']]]


def get_annotation_ids(dataset, id_field='id'):
    return [ann[id_field] for ann in dataset['annotations']]


def get_annotation_hash_ids(dataset):
    hash_ids = []
    img_to_anns = get_annotations_dict(dataset)
    for img in dataset['images']:
        if img['id'] in img_to_anns:
            anns = img_to_anns[img['id']]
            hash_ids.extend([f'{img["file_name"]}_{"_".join(map(str, ann["bbox"]))}' for ann in anns])
    return hash_ids


def sort_dataset_images(images, which_col='file_name'):
    if isinstance(which_col, str):
        keys = [img[which_col] for img in images]
    else:
        assert is_iterable(which_col)
        dtypes = []
        for c in which_col:
            dtypes.append(f'U{np.max([len(str(img[c])) for img in images])}')
        keys = np.array([tuple(img[c] for c in which_col) for img in images], np.dtype(', '.join(dtypes)))

    return np.asarray(images)[np.argsort(keys)].tolist()


def merge_some_categories(dataset, to_merge, new_name):
    assert is_iterable(to_merge)
    assert isinstance(new_name, str)
    assert set(to_merge).issubset([c['name'] for c in dataset['categories']])
    cat_names = get_category_names(dataset, as_dict=True, strict=False)
    to_merge_ids = [cat_names[x] for x in to_merge]
    new_id = to_merge_ids[0]
    for a in dataset['annotations']:
        if a['category_id'] in to_merge_ids:
            a['category_id'] = new_id
    dataset['categories'] = [c for c in dataset['categories'] if c['name'] not in to_merge]
    dataset['categories'].append(dict(id=new_id, name=new_name, supercategory=''))
    sort_categories(dataset)


def get_annotations_dict(dataset, conf_thr=None, only_preds=True, only_this_label=None):
    if not only_preds:
        annotations = {}
        # this includes images without annotations
        for image in dataset['images']:
            annotations[image['id']] = []
    else:
        annotations = defaultdict(list)

    if only_this_label:
        if not is_iterable(only_this_label):
            only_this_label = [only_this_label]
        cat_ids = get_category_names(dataset, as_dict=True)
        cat_ids = [cat_ids[l] for l in only_this_label]

    for ann in dataset['annotations']:
        if (conf_thr is None or ann['score'] >= conf_thr) and \
                (only_this_label is None or ann['category_id'] in cat_ids):
            annotations[ann['image_id']].append(ann)

    return dict(annotations)


def get_category_names(dataset, lower_case=False, as_dict=False, strict=True):
    cat_dict = {cat['name'].lower() if lower_case else cat['name']: cat['id'] for cat in dataset['categories']}
    cat_ids = list(cat_dict.values())
    # because ML models assume this
    assert not strict or cat_ids == list(range(1, len(cat_ids) + 1)), 'Category id with wrong format'

    if as_dict:
        return cat_dict
    else:
        return list(cat_dict.keys())


def get_category_ids(dataset, with_names=False, lower_case=False, as_dict=False, strict=True):
    cat_dict = {cat['id']: cat['name'].lower() if lower_case else cat['name'] for cat in dataset['categories']}
    cat_ids = list(cat_dict.keys())
    # because ML models assume this
    assert not strict or cat_ids == list(range(1, len(cat_ids) + 1)), 'Category id with wrong format'

    if as_dict:
        return cat_dict
    elif with_names:
        return cat_ids, list(cat_dict.values())
    else:
        return cat_ids


def plot_bboxes(img, annotations, cat_dict=None, img_dir=None, palette='bright', draw_labels=True,
                thickness=3, is_opaque=False, alpha=0.5):
    import bbox_visualizer as bbv
    import seaborn as sns
    # print('img:', img.shape)
    # print('annotations:', annotations[0])
    # print('cat_dict:', cat_dict)

    def _get_id(ann):
        return f'id:{ann["track_id"]:.0f}'

    def _get_score(ann):
        return f'{ann["score"]:.2f}'

    if cat_dict is None:
        cat_dict = {ann['category_id']: ann['category_id'] for ann in annotations}
    if isinstance(palette, str):
        palette = (np.asarray(sns.color_palette(palette)) * 255).tolist()
    if isinstance(img, dict):
        img = read_image(os.path.join(img_dir, img['file_name']))

    if annotations:
        bboxes = from_XYWH_to_XYXY([ann['bbox'] for ann in annotations]).astype(int)
        labels = np.asarray([
            f"{f'{_get_id(ann)} ' if 'track_id' in ann and ann['track_id'] is not None else ''}{cat_dict.get(ann['category_id'], str(ann['category_id']))}{f' {_get_score(ann)}' if 'score' in ann else ''}"
            for ann in annotations
        ])
        categories = [ann['category_id'] for ann in annotations]

        for cat, color in zip(list(cat_dict.keys()), palette):
            mask = [c == cat for c in categories]
            if len(mask) > 0:
                img = bbv.draw_multiple_rectangles(img=img, bboxes=bboxes[mask].tolist(), bbox_color=color,
                                                   thickness=thickness, is_opaque=is_opaque, alpha=alpha)
                if draw_labels:
                    img = bbv.add_multiple_labels(img=img, labels=labels[mask].tolist(), bboxes=bboxes[mask].tolist(),
                                                  text_bg_color=color, text_color=(255, 255, 255), top=True)
    return img


def get_image_ids_from_annotations(annotations, unique=True):
    img_ids = [ann['image_id'] for ann in annotations]
    if unique:
        img_ids = np.unique(img_ids).tolist()
    return img_ids


def get_annotations_for_image(image_id, annotations, which=None, apply_func=None):
    if which is None:
        return [ann for ann in annotations if ann['image_id'] == image_id]
    else:
        if apply_func is None:
            apply_func = lambda x: x
        return [apply_func(ann[which]) for ann in annotations if ann['image_id'] == image_id]


def _area_inside(area, min_area, max_area):
    return (area > min_area if min_area is not None else True) and (area <= max_area if max_area is not None else True)


def _area_outside(area, min_area, max_area):
    return (area <= min_area if min_area is not None else True) or (area > max_area if max_area is not None else True)


def filter_annotations_by_size(annotations, min_area, max_area, keep=True, remove=False):
    assert sum([keep, remove]) == 1
    assert min_area is not None or max_area is not None
    assert min_area is None or max_area is None or min_area < max_area
    assert all([ann['area'] > 0 for ann in annotations])
    if keep:
        mask = [_area_inside(ann['area'], min_area=min_area, max_area=max_area) for ann in annotations]
    else:
        mask = [_area_outside(ann['area'], min_area=min_area, max_area=max_area) for ann in annotations]
    return np.asarray(mask, dtype=bool)


def filter_images_with_annotations(dataset, with_annotations=True, without_annotations=False):
    assert sum([with_annotations, without_annotations]) == 1
    with_ann = get_image_ids_from_annotations(dataset['annotations'])
    with_ann_mask = np.asarray([img['id'] in with_ann for img in dataset['images']], dtype=bool)
    mask = with_ann_mask if with_annotations else ~with_ann_mask
    return np.asarray(dataset['images'])[mask].tolist()


def filter_images_with_specific_label(dataset, label):
    with_ann = get_annotations_dict(dataset, only_this_label=label)
    with_ann_mask = np.asarray([img['id'] in with_ann for img in dataset['images']], dtype=bool)
    return np.asarray(dataset['images'])[with_ann_mask].tolist()


def filter_annotations_with_images(dataset, with_images=True, without_images=False):
    assert with_images or without_images
    if with_images and without_images:
        return dataset['annotations']
    else:
        image_ids = get_image_ids(dataset)
        with_img_mask = np.asarray([ann['image_id'] in image_ids for ann in dataset['annotations']], dtype=bool)
        if with_images:
            mask = with_img_mask
        elif without_images:
            mask = ~with_img_mask
        else:
            raise ValueError
        return np.asarray(dataset['annotations'])[mask].tolist()


def use_basenames(images, drop_license=False):
    for img in images:
        img['file_name'] = os.path.basename(img['file_name'])

        if drop_license and 'license' in img:
            del img['license']


def reid_images(dataset):
    rename = {}
    for img in dataset['images']:
        new_id = os.path.basename(img['file_name'])
        rename[img['id']] = new_id
        img['id'] = new_id

    for ann in dataset['annotations']:
        ann['image_id'] = rename[ann['image_id']]


def get_labels_for_images(dataset, with_names=False):
    cat_ids = get_category_ids(dataset, with_names=with_names)
    if with_names:
        cat_ids, cat_names = cat_ids
    binarizer = MultiLabelBinarizer(classes=cat_ids).fit([cat_ids])
    img_labels = binarizer.transform([
        set(get_annotations_for_image(img, dataset['annotations'], which='category_id')) for img in
        get_image_ids(dataset)
    ])
    if with_names:
        assert len(cat_names) == img_labels.shape[1]

    return (img_labels, cat_names) if with_names else img_labels


def get_annot_sizes_as_labels_for_images(dataset, with_names=False):
    size_names = list(MIN_MAX_AREA.keys())
    binarizer = MultiLabelBinarizer(classes=size_names).fit([size_names])
    img_labels = binarizer.transform([
        set(get_annotations_for_image(img, dataset['annotations'], which='area', apply_func=label_area)) for img in
        get_image_ids(dataset)
    ])
    if with_names:
        assert len(size_names) == img_labels.shape[1]
    return (img_labels, size_names) if with_names else img_labels


def label_area(area):
    label = None
    for size_label in MIN_MAX_AREA:
        if _area_inside(area, min_area=MIN_MAX_AREA[size_label][0], max_area=MIN_MAX_AREA[size_label][1]):
            assert label is None
            label = size_label
    return label


def get_split_idx(dataset, n_folds, test_size, val_size, img_labels=None, multi_label=False, groups=None,
                  groups_test_only=False, random_seed=None, swapping_allowed=True):
    '''
    :param multi_label: If True, every split will have a similar number of categories (summed across all images).
                        If False, every split will have a similar number of images with specific combinations of categories.
    '''

    if test_size is None:
        test_size = 0
    if val_size is None:
        val_size = 0
    if img_labels is None:
        img_labels = get_labels_for_images(dataset)

    assert_n_folds(n_folds)
    for s in [test_size, val_size]:
        if isinstance(s, int):
            assert s >= 0
        else:
            assert isinstance(s, float) and 0 <= s <= 1
    assert 0 < n_folds == int(n_folds)
    assert n_folds != 1 or test_size > 0
    assert not multi_label or groups is None
    assert not groups_test_only or groups is not None

    if multi_label or groups is not None:
        if test_size >= 1:
            test_size = test_size / len(dataset['images'])
        if val_size >= 1:
            val_size = val_size / len(dataset['images'])

    if 0 < val_size < 1:
        n = len(dataset['images'])
        n_val = n * val_size
        #
        # if n_folds != 1:
        #     n_train = n - (n / n_folds)
        # else:
        #     if test_size < 1:
        #         assert test_size + val_size < 1
        #         n_train = n * (1 - test_size)
        #     else:
        #         n_train = n - test_size
        #
        # val_size = n_val / n_train
        # assert 0 < val_size < 1

    if multi_label:
        assert groups is None
        '''
        CREDIT: https://github.com/akarazniewicz/cocosplit
        '''
        from skmultilearn.model_selection import IterativeStratification
        # only fractional test/val sizes for multi-label splits
        assert n_folds != 1 or test_size < 1
        _global_state = np.random.get_state()
        np.random.seed(random_seed)
        test_splitter = IterativeStratification(
            n_splits=n_folds if n_folds != 1 else 2,
            sample_distribution_per_fold=None if n_folds != 1 else [test_size, 1.0 - test_size]
        )
    else:
        cv_kwargs = {}
        if n_folds == 1:
            assert test_size < 1
            test_splitter = StratifiedGroupShuffleSplit if groups else StratifiedShuffleSplit
            cv_kwargs['test_size'] = test_size
            cv_kwargs['swapping_allowed'] = swapping_allowed
        else:
            test_splitter = FixedStratifiedGroupKFold if groups else StratifiedKFold
            cv_kwargs['shuffle'] = True
        test_splitter = test_splitter(n_splits=n_folds, random_state=random_seed, **cv_kwargs)

    # y = img_labels if multi_label else bool2int(img_labels)
    # JUST FOR THE VENTILATION RATE COMPARISONS
    if multi_label:
        y = img_labels
    else:
        cached_y = None
        cached_y_fn = f'cached_img_labels.{len(dataset["images"])}.seed{random_seed}.pckl'
        # cached_y = pickle_load(cached_y_fn) if os.path.exists(cached_y_fn) else None

        if cached_y is not None and len(cached_y) == len(img_labels):
            y = cached_y
            print(f'Using {cached_y_fn} as test split')
        else:
            y = bool2int(img_labels)
            # pickle_dump(y, f'cached_img_labels.{len(dataset["images"])}.seed{random_seed}.pckl')

    def get_val_split(X, y, groups, random_seed):
        _val_size = (n_val / len(X)) if val_size < 1 else val_size
        if multi_label:
            assert groups is None
            assert _val_size < 1
            np.random.seed(random_seed)
            val_splitter = IterativeStratification(n_splits=2,
                                                   sample_distribution_per_fold=[_val_size, 1.0 - _val_size])
        elif groups is not None:
            assert _val_size < 1
            val_splitter = StratifiedGroupShuffleSplit(n_splits=1, test_size=_val_size, random_state=random_seed,
                                                       swapping_allowed=swapping_allowed)
        else:
            val_splitter = StratifiedShuffleSplit(n_splits=1, test_size=_val_size, random_state=random_seed)
        return [next(val_splitter.split(X=X, y=y, groups=groups))]

    idx = np.arange(len(dataset['images']), dtype=int)
    if n_folds == 1:
        _train_test_splits = [next(test_splitter.split(idx, y=y, groups=groups))]
    else:
        _train_test_splits = test_splitter.split(idx, y=y, groups=groups)

    if val_size == 0:
        split_idx = [(np.sort(idx[train]), np.sort(idx[test])) for train, test in _train_test_splits]
    else:
        split_idx = [
            (np.sort(idx[train_val][train]), np.sort(idx[train_val][val]), np.sort(idx[test])) for
            i, (train_val, test) in enumerate(_train_test_splits) for
            train, val in get_val_split(
                X=idx[train_val], y=y[train_val],
                groups=np.asarray(groups)[train_val] if groups is not None and not groups_test_only else None,
                random_seed=random_seed + i)
        ]

    if multi_label:
        np.random.set_state(_global_state)

    assert len(split_idx) == n_folds
    for s in split_idx:
        _idx = list(itertools.chain(*s))
        assert len(idx) == len(_idx)
        assert len(idx) == len(set(_idx))
        assert sorted(idx) == sorted(_idx)

    return split_idx


def get_default_data_filters():
    data_filters = {
        f'{split.upper()}_RMV_{size.upper()}': False
        for size in MIN_MAX_AREA.keys()
        for split in ['global', 'train', 'val', 'test']
    }
    for split in ['train', 'val', 'test']:
        data_filters[f'{split.upper()}_RMV_EMPTY'] = False

    return data_filters


def set_data_filters(data_filters, which_splits, filter_empty=False, filter_sizes=None):
    assert filter_empty in [True, False]
    if isinstance(which_splits, str):
        which_splits = [which_splits]
    if isinstance(filter_sizes, str):
        filter_sizes = [filter_sizes]
    assert not filter_sizes or is_iterable(filter_sizes)

    for split in which_splits:
        if filter_empty:
            data_filters[f'{split.upper()}_RMV_EMPTY'] = True
        if filter_sizes:
            for size in filter_sizes:
                data_filters[f'{split.upper()}_RMV_{size.upper()}'] = True


def apply_data_filters(dataset, data_filters, n_folds=1, only_global=False, masked_bboxes_per_image=None):
    assert_n_folds(n_folds)
    if masked_bboxes_per_image is None:
        masked_bboxes_per_image = defaultdict(list)
    if data_filters is not None:
        if only_global:
            # must be non-split dataset
            assert 'images' in dataset
            for size in MIN_MAX_AREA.keys():
                if data_filters[f'GLOBAL_RMV_{size.upper()}']:
                    remove_by_size(dataset, min_max_area=MIN_MAX_AREA[size],
                                   info=f'{size} globally', masked_bboxes_per_image=masked_bboxes_per_image)
        else:
            # must be a split dataset
            assert 'images' not in dataset
            for i in range(n_folds) if n_folds != 1 else [None]:
                train_val_test_dict = dataset[i] if i is not None else dataset
                for split in ['train', 'val', 'test']:
                    if split in train_val_test_dict:
                        for size in MIN_MAX_AREA.keys():
                            if data_filters[f'{split.upper()}_RMV_{size.upper()}']:
                                remove_by_size(train_val_test_dict[split], min_max_area=MIN_MAX_AREA[size],
                                               info=f'{size} {split}', masked_bboxes_per_image=masked_bboxes_per_image)
                        # This should be last so that it takes into account previous filtrations
                        if data_filters[f'{split.upper()}_RMV_EMPTY']:
                            remove_empty(train_val_test_dict[split], info=split)
    return masked_bboxes_per_image


def remove_by_size(dataset, min_max_area, info=None, masked_bboxes_per_image=None, verbose=True):
    if masked_bboxes_per_image is None:
        masked_bboxes_per_image = defaultdict(list)
    if verbose:
        print(f'Removing {info if info is not None else ""} annotations: {len(dataset["annotations"])} -->', end=' ')
    mask = filter_annotations_by_size(
        dataset['annotations'], min_area=min_max_area[0], max_area=min_max_area[1], keep=False, remove=True)
    annots_as_array = np.asarray(dataset['annotations'])
    dataset['annotations'] = annots_as_array[mask].tolist()
    for del_ann in annots_as_array[~mask]:
        masked_bboxes_per_image[del_ann['image_id']].append(del_ann['bbox'])

    if verbose:
        print(f'{len(dataset["annotations"])} annotations')
    return masked_bboxes_per_image


def remove_empty(dataset, info=None, verbose=True):
    if verbose:
        print(f'Removing {info if info is not None else ""} images without annotations: {len(dataset["images"])} -->',
              end=' ')
    dataset['images'] = filter_images_with_annotations(dataset)
    if verbose:
        print(f'{len(dataset["images"])} images')


def set_common_image_root(datasets, image_roots, splits=None):
    assert len(datasets) == len(image_roots) and len(image_roots) >= 1
    if len(set(image_roots)) > 1:
        common_image_root = os.path.commonpath(image_roots)
        assert len(common_image_root) != 0, f'No common path exists for the two datasets {common_image_root}'
        if common_image_root.endswith(os.path.sep):
            common_image_root = common_image_root[:-1]
        print('common image root:', common_image_root)

        for dataset, image_root in zip(datasets, image_roots):
            assert image_root.startswith(common_image_root), (image_root, common_image_root)
            fn_prefix = image_root[len(common_image_root):]
            if len(fn_prefix) != 0:
                assert fn_prefix.startswith(os.path.sep), (image_root, common_image_root, fn_prefix)
                fn_prefix = fn_prefix[1:]
                for img in dataset['images']:
                    img['file_name'] = os.path.join(fn_prefix, img['file_name'])
            print('new file name prefix:', fn_prefix)

        return common_image_root
    else:
        return image_roots[0]


def concat_crossval_results(result_files, assert_correct=True, only_preds=None):
    def concat_results(files, assert_correct, only_preds, verbose=False):
        dataset = {'images': [], 'annotations': []} if not only_preds else {'annotations': []}
        for fold, fn in enumerate(files):
            try:
                d = read_json(fn, assert_correct=assert_correct, only_preds=only_preds, verbose=verbose)
            except Exception as e:
                print(f'Cannot read {fn}')
                raise e
            if not only_preds:
                if 'categories' not in dataset:
                    dataset['categories'] = copy.copy(d['categories'])
                else:
                    assert dataset['categories'] == d['categories']
            for objects in ['images', 'annotations'] if not only_preds else ['annotations']:
                for obj in d[objects]:
                    for id_key in ['id', 'image_id']:
                        if id_key in obj:
                            obj[id_key] = f'fold{fold}:{obj[id_key]}'
                        if 'license' in obj:
                            del obj['license']
                dataset[objects].extend(d[objects])

        # make integer IDs
        for i, ann in enumerate(dataset['annotations']):
            ann['id'] = i + 1

        return dataset

    assert len(result_files) > 1
    if isinstance(result_files[0], str):
        assert all([isinstance(f, str) for f in result_files])
        return concat_results(
            result_files, assert_correct=assert_correct, only_preds=False if only_preds is None else only_preds)
    elif is_iterable(result_files[0]) and len(result_files[0]) == 2:
        assert all([is_iterable(f) and len(f) == 2 for f in result_files])
        gts, dts = zip(*result_files)
        gts = concat_results(gts, assert_correct=assert_correct, only_preds=False if only_preds is None else only_preds)
        dts = concat_results(dts, assert_correct=assert_correct, only_preds=True if only_preds is None else only_preds)
        return gts, dts
    else:
        raise ValueError()


def assert_results_type(results):
    assert is_result_type(results), results


def is_result_type(results):
    return isinstance(results, COCOResults) or \
               (is_iterable(results, dict_allowed=False) and all(
                   [r is None or isinstance(r, COCOResults) for r in results]))


def clean_cached_results(results, clean_precision=False, clean_PRs=False, clean_stats=False,
                         clean_eval=True, clean_detections=True, clean_ground_truths=True):
    for train_data in results:
        for model in results[train_data]:
            for model_select in results[train_data][model]:
                for split in results[train_data][model][model_select]:
                    for seed in results[train_data][model][model_select][split]:
                        res = results[train_data][model][model_select][split][seed]
                        if is_result_type(res):
                            for r in (res if is_iterable(res) else [res]):
                                if r is not None:
                                    if clean_precision:
                                        r.precision = None
                                    if clean_PRs:
                                        r.PRs = None
                                    if clean_stats:
                                        r.stats = None
                                    if clean_eval:
                                        r.eval = None
                                    if clean_detections:
                                        r.dt_coco = None
                                    if clean_ground_truths:
                                        r.gt_coco = None


def reload_results(results):
    for train_data in results:
        for model in results[train_data]:
            for model_select in results[train_data][model]:
                for split in results[train_data][model][model_select]:
                    for seed in results[train_data][model][model_select][split]:
                        res = results[train_data][model][model_select][split][seed]
                        assert_results_type(res)
                        for r in (res if is_iterable(res) else [res]):
                            r.reload()


def reevaluate_results(results, iouType='bbox', useCats=True, imgIds=None, catIds=None, iouThrs=None, recThrs=None,
                       areaRng=None, areaRngLbl=None, maxDets=None, verbose=True):
    for train_data in results:
        for model in results[train_data]:
            for model_select in results[train_data][model]:
                for split in results[train_data][model][model_select]:
                    for seed in results[train_data][model][model_select][split]:
                        res = results[train_data][model][model_select][split][seed]
                        assert_results_type(res)
                        for r in res if is_iterable(res) else [res]:
                            r.evaluate(
                                iouType=iouType, useCats=useCats, imgIds=imgIds, catIds=catIds, iouThrs=iouThrs,
                                recThrs=recThrs, areaRng=areaRng, areaRngLbl=areaRngLbl, maxDets=maxDets,
                                verbose=verbose)


def read_and_eval_yolo3(train_datas, seeds, results_dir, fix_zero_ann_ids=False, verbose=True, results=None,
                        **eval_kwargs):
    if results is None:
        results = {}
    if not is_iterable(train_datas):
        train_datas = [train_datas]
    if not is_iterable(seeds):
        seeds = [seeds]

    for train_data in train_datas:
        if train_data not in results:
            results[train_data] = {}
            results[train_data]['yolo3'] = {}
            results[train_data]['yolo3']['last'] = {}
            results[train_data]['yolo3']['last']['test'] = {}

        for seed in seeds:
            test_dir = os.path.join(results_dir, f'yolo3.{train_data}.seed{seed}', 'test')
            gt_coco = read_json(os.path.join(test_dir, 'annotations.test.json'), fix_zero_ann_ids=fix_zero_ann_ids,
                                verbose=False)
            img_id_to_name = {}
            for img in gt_coco['images']:
                new_id = '.'.join(img['file_name'].split('.')[:-1])
                img_id_to_name[img['id']] = new_id
                img['id'] = new_id
            for ann in gt_coco['annotations']:
                ann['image_id'] = img_id_to_name[ann['image_id']]
            save_json(gt_coco, os.path.join(test_dir, 'annotations.fixed.test.json'), compress=True)

            dt_coco = read_json(os.path.join(test_dir, 'last_predictions.json'), only_preds=True,
                                fix_zero_ann_ids=fix_zero_ann_ids, verbose=False)
            for ann in dt_coco['annotations']:
                ann['category_id'] += 1
            save_json(dt_coco, os.path.join(test_dir, 'last_predictions.fixed.json'), only_preds=True, compress=True)

            r = evaluate(gt_coco=gt_coco, dt_coco=dt_coco,
                         allow_zero_area_boxes=True, verbose=verbose, **eval_kwargs)
            if r is not None:
                results[train_data]['yolo3']['last']['test'][seed] = r

        if results[train_data]['yolo3']['last']['test'] == {}:
            results[train_data]['yolo3']['last'].pop('test')

    return results


def read_and_eval_results(
        train_datas, models, model_selects, seeds, splits, results_dir, n_folds=None, concat=True, test_names=None,
        fix_zero_ann_ids=False, verbose=True, results=None, filename_group=None, class_agnostic_nms=None, **eval_kwargs):
    if results is None:
        results = {}
    if not is_iterable(train_datas):
        train_datas = [train_datas]
    if not is_iterable(models):
        models = [models]
    if not is_iterable(model_selects):
        model_selects = [model_selects]
    if not is_iterable(seeds):
        seeds = [seeds]
    if not is_iterable(splits):
        splits = [splits]

    if test_names is None:
        test_names = [None] * len(splits)
    elif not is_iterable(test_names):
        test_names = [test_names]
    assert len(test_names) == len(splits)

    for train_data in train_datas:
        if train_data not in results:
            results[train_data] = {}

        for model in models:
            if model not in results[train_data]:
                results[train_data][model] = {}

            for model_select in model_selects:
                if model_select not in results[train_data][model]:
                    results[train_data][model][model_select] = {}

                for split, test_name in zip(splits, test_names):
                    _spl = f'{split}{f"_{test_name}" if test_name else ""}'

                    for seed in seeds:
                        r = read_and_eval(model=model, model_select=model_select, train_data=train_data, split=split,
                                          random_seed=seed, results_dir=results_dir, concat=concat, n_folds=n_folds,
                                          test_name=test_name, fix_zero_ann_ids=fix_zero_ann_ids, verbose=verbose,
                                          filename_group=filename_group, class_agnostic_nms=class_agnostic_nms, **eval_kwargs)
                        if r is not None and not (is_iterable(r) and all([_r is None for _r in r])):
                            if _spl not in results[train_data][model][model_select]:
                                results[train_data][model][model_select][_spl] = {}
                            results[train_data][model][model_select][_spl][seed] = r
    return results


def read_and_eval(model, model_select, train_data, split, random_seed, results_dir, n_folds=None, concat=True,
                  test_name=None, data_dir=None, fix_zero_ann_ids=False, hack=False, verbose=True, filename_group=None,
                  class_agnostic_nms=None, **eval_kwargs):
    if n_folds is None:
        n_folds = 1
    run_dir = os.path.join(results_dir, f'{model}.{train_data}.seed{random_seed}')
    result_files = []
    for fold in [None] if n_folds == 1 else range(n_folds):
        if hack:
            assert not fix_zero_ann_ids
            # A small hack for historical reasons
            assert data_dir is not None and test_name is not None
            gt_fn = os.path.join(data_dir, 'annotations', f'{test_name}.json')
            dt_fn = os.path.join(results_dir, f'{model}.{train_data}.seed{random_seed}',
                                 f'test_predictions_{model_select}',
                                 test_name, 'predictions.json')
        elif test_name is None:
            # THIS CAN BE CONCATENATED FOR KFOLD
            gt_fn = get_data_fn(os.path.join(run_dir, 'datasets', 'annotations'),
                                random_seed=random_seed, split=split, fold=fold)
            dt_fn = get_data_fn(os.path.join(run_dir, '' if fold is None else f'fold{fold}',
                                             f'{split}_predictions_{model_select}', 'predictions'),
                                random_seed=random_seed, split=split, fold=fold)
        else:
            # THIS NEEDS TO BE AVERAGED FOR KFOLD
            assert n_folds == 1 or not concat
            test_dir = os.path.join(run_dir, '' if fold is None else f'fold{fold}',
                                    f'{split}_predictions_{model_select}', test_name)
            gt_fn = os.path.join(test_dir, f'inputs.json')
            dt_fn = os.path.join(test_dir, f'predictions.json')
        result_files.append((gt_fn, dt_fn))

    if n_folds > 1 and concat:
        assert test_name is None
        try:
            gts, dts = concat_crossval_results(result_files)
        except FileNotFoundError as e:
            print(e, file=sys.stderr)
            result_files = [(None, None)]
        else:
            os.path.join(run_dir, '' if fold is None else f'fold{fold}')
            concat_files = []
            for which, dataset, only_preds in [
                ('annotations', gts, False),
                ('predictions', dts, True)
            ]:
                fn = get_data_fn(os.path.join(run_dir, f'concat_{which}_{model_select}'),
                                 random_seed=random_seed, split=split, fold=None)
                save_json(dataset, assert_correct=False, fn=fn, only_preds=only_preds, compress=True)
                concat_files.append(fn)
            result_files = [concat_files]

    results = []
    for gt_fn, dt_fn in result_files:
        if not file_or_gzip_exists(dt_fn):
            print(f'WARNING: missing {dt_fn}', file=sys.stderr)
            r = None
        else:
            if verbose:
                print(f'\nEvaluating {dt_fn}')
            r = evaluate(gt_coco=gt_fn, dt_coco=dt_fn, allow_zero_area_boxes=True, fix_zero_ann_ids=fix_zero_ann_ids,
                         verbose=verbose, filename_group=filename_group, class_agnostic_nms=class_agnostic_nms, **eval_kwargs)
            assert len(r.dt_coco.dataset['annotations']) / len(r.gt_coco.dataset['images']) <= 100, \
                f"I expected max. of 100 detections per image {len(r.dt_coco.dataset['annotations'])}, {len(r.gt_coco.dataset['images'])}"
        results.append(r)

    return results[0] if len(results) == 1 else results


def get_pickled_results(model, dataset, eval_type, ccf, pickles_dir, only_filename=False):
    assert ccf in [True, False, None]
    fn = os.path.join(pickles_dir,
                      f'{dataset}.{eval_type}.{model}{"" if dataset == "S-UODAC" else ".CCF" if ccf else ".not_CCF"}.pckl')
    return fn if only_filename else pickle_load(fn)


def combine_annot_without_overlap(primary, secondary, conf_thr=None, iou_thr=0.5, reduce=False):
    r = COCOResults(gt_coco=primary, dt_coco=secondary)
    r.evaluate(conf_thr=conf_thr, iouType='bbox', iouThrs=[iou_thr], verbose=True)
    _, FPs, _ = r.eval.TPs_FPs_FNs(iouThr=iou_thr, concat=True)
    secondary['annotations'] = [ann for ann in secondary['annotations'] if ann['id'] in FPs]

    if reduce:
        img_to_anns = get_annotations_dict(secondary)
        remove = []
        for img in secondary['images']:
            anns = img_to_anns[img['id']]
            if len(anns) > 1:
                for i in range(len(anns)):
                    if anns[i]['id'] not in remove:
                        ious = []
                        for j in range(i + 1, len(anns)):
                            if iou(anns[i]['bbox'], anns[j]['bbox']) >= iou_thr:
                                ious.append(anns[j])
                        ann_id_list = [anns[i]['id']] + [ann['id'] for ann in ious]
                        keep_idx = np.argmax([anns[i]['score']] + [a['score'] for a in ious])
                        rm = [ann_id for idx, ann_id in enumerate(ann_id_list) if idx != keep_idx]
                        if len(rm) > 0:
                            remove.extend(rm)
        secondary['annotations'] = [ann for ann in secondary['annotations'] if ann['id'] not in remove]

    merged = merge_datasets(primary, secondary, check_licenses=False, check_optional_fields=False)
    return merged


def read_all_datasets(data_json_fns, data_img_dirs=None, sort_cats=False, quick_debug=False, random_state=None,
                      verbose=True):
    if isinstance(data_img_dirs, str):
        data_img_dirs = [data_img_dirs]
    assert data_img_dirs is None or len(data_img_dirs) == 1 or len(data_img_dirs) == len(data_json_fns)
    assert not quick_debug or isinstance(quick_debug, int)

    if len(data_json_fns) != 1:
        all_datasets = [read_json(fn, verbose=verbose) for fn in data_json_fns]
        if data_img_dirs is not None:
            if len(data_img_dirs) != 1:
                data_img_dir = set_common_image_root(all_datasets, data_img_dirs)
            else:
                assert len(data_img_dirs) == 1
                data_img_dir = data_img_dirs[0]

        for dd in all_datasets:
            for cc in dd['categories']:
                if cc['name'].lower() == 'delete':
                    cc['name'] = 'delete'

        dataset = merge_datasets(*all_datasets, verbose=verbose)
    else:
        dataset = read_json(data_json_fns[0], verbose=verbose)
        if data_img_dirs is not None:
            assert len(data_img_dirs) == 1
            data_img_dir = data_img_dirs[0]

    if quick_debug:
        subset_dataset(dataset, size=quick_debug, random_state=random_state)

    if sort_cats:
        sort_categories(dataset)

    return (dataset, data_img_dir) if data_img_dirs is not None else dataset


def subset_dataset(dataset, size, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState()
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    random_state.shuffle(dataset['images'])

    images = []
    for cat_id in get_category_ids(dataset, as_dict=True):
        dummy = copy.copy(dataset)
        select_category(dummy, category_id=cat_id, new_category_id=None)
        images.extend(filter_images_with_annotations(dummy)[:size])
    images = [images[i] for i in range(len(images)) if images[i]['id'] not in [img['id'] for img in images[:i]]]
    dataset['images'] = images
    dataset['annotations'] = filter_annotations_with_images(dataset)
    print(f'REDUCING TO {len(images)} IMAGES and {len(dataset["annotations"])} annotations')


def subsample_dataset(dataset, n_imgs=None, n_groups=None, seed=None, x=None):
    if x is None:
        x = 1
    assert n_imgs is None or n_imgs > 0
    assert n_groups is None or n_groups > 0
    assert not (n_imgs is None and n_groups is None)
    random_state = np.random.RandomState(seed * x)
    imgs = np.asarray(dataset['images'])
    random_state.shuffle(imgs)

    if n_imgs is not None:
        if n_imgs < 1:
            n_imgs *= len(imgs)
        n_imgs = int(n_imgs)

    if n_groups is not None:
        groups = np.asarray(get_filename_groups(imgs))
        if n_groups < 1:
            n_groups *= len(np.unique(groups))
        n_groups = int(n_groups)

        while True:
            idx = random_state.choice(
                np.arange(len(np.unique(groups)), dtype=int), size=n_groups, replace=False)
            mask = np.isin(groups, np.unique(groups)[idx])
            if n_imgs is None or mask.sum() >= n_imgs:
                break
        imgs = imgs[mask]

    if n_imgs is not None:
        idx = random_state.choice(
            np.arange(len(imgs), dtype=int), size=n_imgs, replace=False)
        imgs = imgs[idx]

    dataset['images'] = sorted(imgs.tolist(), key=lambda x: x['id'])
    dataset['annotations'] = filter_annotations_with_images(dataset=dataset)

    groups = np.asarray(get_filename_groups(dataset['images']))
    print('SUBSAMPLED GROUPS:', len(np.unique(groups)))
    print('SUBSAMPLED IMAGES', len(dataset['images']))

    return dataset


def prepare_datasets(data_json_fns, data_img_dirs, test_json_fns, test_img_dirs, val_json_fns, val_img_dirs,
                     n_folds, test_size, test_groups, val_size, sort_cats,
                     multi_label, filename_groups, filename_groups_test_only, group_splitter,
                     seed, output_dir, stratify_bbox_sizes=False,
                     merge_categories_as=None, data_filters=None, img_transforms=None, coinflip_transform=False,
                     extra_train_json_fns=None, extra_train_img_dirs=None,
                     drop_test_groups_in_extra_train=False, drop_val_groups_in_extra_train=False,
                     extra_val_size=0, subsample_train=None, subsample_groups=None,
                     out_formats='coco', copy_imgs='auto', segm=None, sort_images=False, save=True, quick_debug=False):
    assert_n_folds(n_folds)

    if test_size is None:
        test_size = 0

    if val_size is None:
        val_size = 0

    if extra_val_size is None:
        extra_val_size = 0

    if extra_train_img_dirs is None:
        extra_train_img_dirs = copy.copy(data_img_dirs)

    if isinstance(data_json_fns, str):
        data_json_fns = [data_json_fns]
    if isinstance(data_img_dirs, str):
        data_img_dirs = [data_img_dirs]
    if isinstance(extra_train_json_fns, str):
        extra_train_json_fns = [extra_train_json_fns]
    if isinstance(extra_train_img_dirs, str):
        extra_train_img_dirs = [extra_train_img_dirs]

    assert len(data_img_dirs) == 1 or len(data_img_dirs) == len(data_json_fns)
    assert not test_json_fns or not test_img_dirs or len(test_img_dirs) == 1 or len(test_img_dirs) == len(test_json_fns)
    assert not val_json_fns or not val_img_dirs or len(val_img_dirs) == 1 or len(val_img_dirs) == len(val_json_fns)
    assert not extra_train_json_fns or len(extra_train_img_dirs) == 1 or len(extra_train_img_dirs) == len(
        extra_train_json_fns)

    if test_size != 0 or n_folds != 1:
        assert test_img_dirs is None or len(test_img_dirs) == 1
    if val_size != 0 or n_folds != 1:
        assert val_img_dirs is None or len(val_img_dirs) == 1

    if quick_debug is True:
        quick_debug = 10

    assert all([isinstance(s, (int, float)) and s >= 0 for s in [test_size, val_size, extra_val_size]]), [test_size,
                                                                                                          val_size,
                                                                                                          extra_val_size]
    assert not test_json_fns or test_size == 0, (test_json_fns, test_size)
    assert not test_groups or not test_json_fns, (test_groups, test_json_fns)
    assert not test_groups or test_size == 0, (test_groups, test_size)
    assert not val_json_fns or (val_size == 0 and extra_val_size == 0), (val_json_fns, val_size, extra_val_size)

    # if val_size != 0:
    #     assert extra_train_json_fns is None or (extra_val_size != 0)

    print('DATASET')
    dataset, data_img_dir = read_all_datasets(data_json_fns, data_img_dirs, sort_cats=sort_cats,
                                              quick_debug=quick_debug)
    if sort_images:
        dataset['images'] = sort_dataset_images(dataset['images'])

    test_cat_names = get_category_names(dataset, strict=True)
    masked_bboxes_per_image = defaultdict(list)
    apply_data_filters(dataset, data_filters, n_folds=n_folds, only_global=True,
                       masked_bboxes_per_image=masked_bboxes_per_image)

    if extra_train_json_fns:
        print(f'EXTRA TRAIN{" AND VAL" if extra_val_size != 0 else ""} DATASET')
        extra_train_dataset, extra_train_img_dir = read_all_datasets(
            extra_train_json_fns, extra_train_img_dirs, sort_cats=sort_cats, quick_debug=quick_debug)
        extra_cat_names = get_category_names(extra_train_dataset, strict=True)
        assert len(set(test_cat_names).intersection(extra_cat_names)) == \
               len(set(map(str.lower, test_cat_names)).intersection(map(str.lower, extra_cat_names))), \
            f'Unsafe merging {test_cat_names} with {extra_cat_names}'
        cat_names = copy.copy(test_cat_names)
        cat_names.extend([c for c in extra_cat_names if c not in test_cat_names])
        apply_data_filters(extra_train_dataset, data_filters, n_folds=n_folds, only_global=True,
                           masked_bboxes_per_image=masked_bboxes_per_image)
        image_root = set_common_image_root((dataset, extra_train_dataset), (data_img_dir, extra_train_img_dir))
    elif extra_train_img_dirs and extra_train_img_dirs != data_img_dirs:
        assert len(extra_train_img_dirs) == 1
        assert os.path.isdir(extra_train_img_dirs[0])
        extra_train_img_dir = extra_train_img_dirs[0]
        extra_train_dataset = create_coco_dataset(extra_train_img_dir, cat_names=test_cat_names)
        cat_names = copy.copy(test_cat_names)
        image_root = set_common_image_root((dataset, extra_train_dataset), (data_img_dir, extra_train_img_dir))
    else:
        extra_train_dataset = None
        cat_names = copy.copy(test_cat_names)
        image_root = data_img_dir

    common_image_root_kwargs = dict(splits=[], datasets=[], image_roots=[])

    if test_json_fns:
        assert test_size == 0
        assert not test_groups
        print(f'INDEPENDENT TEST DATASET')
        if test_img_dirs is None:
            test_img_dirs = copy.copy(data_img_dirs)
        test_dataset, test_img_dir = read_all_datasets(
            test_json_fns, test_img_dirs, sort_cats=sort_cats, quick_debug=quick_debug)
        apply_data_filters(test_dataset, data_filters, n_folds=n_folds, only_global=True,
                           masked_bboxes_per_image=masked_bboxes_per_image)
        common_image_root_kwargs['datasets'].append(test_dataset)
        common_image_root_kwargs['image_roots'].append(test_img_dir)
        common_image_root_kwargs['splits'].append('test')
        test_cat_names = get_category_names(test_dataset, strict=True)
        assert set(cat_names).issuperset(test_cat_names), (cat_names, test_cat_names)

    if test_groups:
        assert n_folds == 1
        assert test_size == 0
        assert val_size == 0
        assert not test_json_fns
        print(f'PREDEFINED TEST GROUPS')
        groups = get_filename_groups(dataset['images'], split_char=group_splitter)
        test_mask = np.isin(groups, test_groups)
        assert 0 < test_mask.sum() < len(test_mask)
        test_imgs = np.asarray(dataset['images'])[test_mask].tolist()
        other_imgs = np.asarray(dataset['images'])[~test_mask].tolist()
        test_dataset = copy.deepcopy(dataset)
        dataset['images'] = other_imgs
        dataset['annotations'] = filter_annotations_with_images(dataset, with_images=True)
        test_dataset['images'] = test_imgs
        test_dataset['annotations'] = filter_annotations_with_images(test_dataset, with_images=True)

    if val_json_fns:
        assert val_size == 0
        print(f'VALIDATION DATASET')
        if val_img_dirs is None:
            val_img_dirs = copy.copy(data_img_dirs)
        val_dataset, val_img_dir = read_all_datasets(
            val_json_fns, val_img_dirs, sort_cats=sort_cats, quick_debug=quick_debug)
        apply_data_filters(val_dataset, data_filters, n_folds=n_folds, only_global=True,
                           masked_bboxes_per_image=masked_bboxes_per_image)
        common_image_root_kwargs['datasets'].append(val_dataset)
        common_image_root_kwargs['image_roots'].append(val_img_dir)
        common_image_root_kwargs['splits'].append('val')
        val_cat_names = get_category_names(val_dataset, strict=True)
        assert set(cat_names).issuperset(val_cat_names), (cat_names, val_cat_names)

    if n_folds == 1:
        print(f'Creating train{"/val" if val_size != 0 else ""}/test splits')
    else:
        print('Creating cross-validation folds')
        assert test_size == 0 and not test_json_fns and not test_groups, 'Setting "n_folds" means there will be no train/test split, set "test_size" and "test_json_fns" and "test_groups" to "None"'
        assert n_folds > 1, 'To perform cross-validation, set n_folds > 1'

    # The main process creates the training/validation/testing data files
    if test_size != 0 or val_size != 0 or n_folds != 1:
        img_labels = get_annot_sizes_as_labels_for_images(dataset) if stratify_bbox_sizes else None
        datasets = split_dataset(
            dataset=dataset, n_folds=n_folds, test_size=test_size, val_size=val_size,
            img_labels=img_labels, multi_label=multi_label,
            filename_groups=filename_groups, filename_groups_test_only=filename_groups_test_only,
            group_splitter=group_splitter,
            extra_train_dataset=extra_train_dataset, extra_val_size=extra_val_size, random_seed=seed,
            drop_test_groups_in_extra_train=drop_test_groups_in_extra_train,
            drop_val_groups_in_extra_train=drop_val_groups_in_extra_train,
        )
    else:
        datasets = {
            'train': dataset
        }

    if test_json_fns:
        for i in range(n_folds) if n_folds != 1 else [None]:
            train_val_test_dict = datasets[i] if i is not None else datasets
            assert 'test' not in train_val_test_dict
            train_val_test_dict['test'] = test_dataset

    if test_groups:
        for i in range(n_folds) if n_folds != 1 else [None]:
            train_val_test_dict = datasets[i] if i is not None else datasets
            assert 'test' not in train_val_test_dict
            train_val_test_dict['test'] = test_dataset

    if val_json_fns:
        for i in range(n_folds) if n_folds != 1 else [None]:
            train_val_test_dict = datasets[i] if i is not None else datasets
            assert 'val' not in train_val_test_dict
            train_val_test_dict['val'] = val_dataset

    for _split, _root in [
        ('train', None),
        ('test', test_img_dirs),
        ('val', val_img_dirs),
    ]:
        if _split not in common_image_root_kwargs['splits']:
            for i in range(n_folds) if n_folds != 1 else [None]:
                train_val_test_dict = datasets[i] if i is not None else datasets
                if _split in train_val_test_dict:
                    common_image_root_kwargs['datasets'].append(train_val_test_dict[_split])
                    common_image_root_kwargs['image_roots'].append(_root[0] if _root else image_root)
                    common_image_root_kwargs['splits'].append(_split)
    print(common_image_root_kwargs['image_roots'])
    image_root = set_common_image_root(**common_image_root_kwargs)

    apply_data_filters(datasets, data_filters, n_folds=n_folds, masked_bboxes_per_image=masked_bboxes_per_image)

    if subsample_train or subsample_groups:
        for i in range(n_folds) if n_folds != 1 else [None]:
            train_val_test_dict = datasets[i] if i is not None else datasets
            train_val_test_dict['train'] = subsample_dataset(
                train_val_test_dict['train'], n_imgs=subsample_train, n_groups=subsample_groups, x=3, seed=seed)

    if merge_categories_as is not None:
        for i in range(n_folds) if n_folds != 1 else [None]:
            train_val_test_dict = datasets[i] if i is not None else datasets
            for _split in train_val_test_dict:
                merge_categories(train_val_test_dict[_split], new_name=merge_categories_as)
        cat_names = [merge_categories_as]
        test_cat_names = [merge_categories_as]

    data_fn_prefix = os.path.join(output_dir, 'datasets', 'annotations')
    if save:
        if isinstance(out_formats, str):
            out_formats = [out_formats]
        if not is_iterable(copy_imgs):
            copy_imgs = [copy_imgs]
        assert len(out_formats) == len(copy_imgs)

        for _out_format, _copy_imgs in zip(out_formats, copy_imgs):
            # if we are masking annotations, then datasets and image_root are updated
            data_fn_prefix, datasets, image_root = save_dataset_splits(
                dataset_splits=datasets, n_folds=n_folds, output_dir=output_dir, segm=segm,
                cat_names=cat_names, image_root=image_root,
                out_format=_out_format, copy_imgs=_copy_imgs,
                data_filters=data_filters, img_transforms=img_transforms, coinflip_transform=coinflip_transform,
                masked_bboxes_per_image=masked_bboxes_per_image, random_seed=seed)

    return datasets, data_fn_prefix, image_root, cat_names, test_cat_names


def assert_n_folds(n_folds):
    assert n_folds is not None and isinstance(n_folds, int) and n_folds >= 1, \
        f'n_folds ({n_folds}) must be "int", set it to 1 simple train-val-test split'


def get_filename_groups(images, split_char=None, n_parts=1, split_on_frame=False, force_splits_for_SUODAC=None):
    if split_char is None:
        split_char = '.frame_' if split_on_frame else '_'

    if force_splits_for_SUODAC is not None:
        groups = [split_char.join(os.path.basename(img['file_name']).split(split_char)[:n_parts]) for img in images]
        groups = np.asarray(groups)
        groups = groups == f'type{(force_splits_for_SUODAC + 1) * 2}'
        return [str(groups == g) for g in groups]
    else:
        return [split_char.join(os.path.basename(img['file_name']).split(split_char)[:n_parts]) for img in images]


def split_dataset(dataset, n_folds, test_size=None, val_size=None, img_labels=None, multi_label=False,
                  filename_groups=False, filename_groups_test_only=False, group_splitter='_',
                  extra_train_dataset=None, extra_val_size=None,
                  drop_test_groups_in_extra_train=False, drop_val_groups_in_extra_train=False,
                  random_seed=None, verbose=True):
    assert_n_folds(n_folds)

    if test_size is None:
        test_size = 0
    if val_size is None:
        val_size = 0
    if extra_val_size is None:
        extra_val_size = 0

    assert all([isinstance(s, (int, float)) and s >= 0 for s in [test_size, val_size, extra_val_size]]), [test_size,
                                                                                                          val_size,
                                                                                                          extra_val_size]
    assert (n_folds != 1 and test_size == 0) or (n_folds == 1 and (test_size != 0 or val_size != 0))

    if n_folds == 1 and test_size == 0:
        test_val_kwargs = dict(test_size=val_size, val_size=0)
    else:
        test_val_kwargs = dict(test_size=test_size, val_size=val_size)

    groups = get_filename_groups(dataset['images'],
                                 split_char=group_splitter) if filename_groups or filename_groups_test_only else None
    swapping_allowed = extra_train_dataset is None and groups is not None  # we only need to swap if groups are used and there is no extra train data
    splits = get_split_idx(dataset, n_folds=n_folds, img_labels=img_labels, multi_label=multi_label, groups=groups,
                           groups_test_only=filename_groups_test_only, random_seed=random_seed,
                           swapping_allowed=swapping_allowed, **test_val_kwargs)

    if extra_train_dataset:
        extra_groups = get_filename_groups(extra_train_dataset['images'],
                                           split_char=group_splitter) if filename_groups else None
        if extra_val_size != 0:
            assert groups is None, \
                "Not implemented: it requires a cleanup of extra_val to eliminate overlap with train and test"
            extra_splits = get_split_idx(extra_train_dataset, n_folds=1, test_size=extra_val_size, val_size=None,
                                         multi_label=multi_label, groups=extra_groups, random_seed=random_seed,
                                         swapping_allowed=swapping_allowed)
            assert len(extra_splits) == 1 and len(extra_splits[0]) == 2
            extra_train_idx, extra_val_idx = extra_splits[0]
        else:
            extra_train_idx, extra_val_idx = np.arange(len(extra_train_dataset['images']), dtype=int), []
    else:
        extra_train_idx, extra_val_idx = None, None

    dataset_splits = {}
    for i, s in enumerate(splits):
        if val_size == 0 and (n_folds != 1 or test_size != 0):
            assert len(s) == 2
            train_idx, val_idx, test_idx = s[0], None, s[1]
        elif n_folds == 1 and test_size == 0 and val_size != 0:
            assert len(s) == 2
            train_idx, val_idx, test_idx = s[0], s[1], None
        else:
            assert len(s) == 3
            train_idx, val_idx, test_idx = s[0], s[1], s[2]

        train_val_test_dict = {}
        for idx, extra_idx, split in [
            (train_idx, extra_train_idx, 'train'),
            (val_idx, extra_val_idx if val_idx is not None else None, 'val'),
            (test_idx, extra_val_idx if val_idx is None else None, 'test'),
        ]:
            if idx is not None:
                train_val_test_dict[split] = copy.deepcopy(dataset)
                train_val_test_dict[split]['images'] = np.array(train_val_test_dict[split]['images'])[idx].tolist()
                train_val_test_dict[split]['annotations'] = filter_annotations_with_images(train_val_test_dict[split])

                if extra_idx is not None:
                    if split != 'test':
                        groups = np.asarray(groups)

                        if drop_test_groups_in_extra_train and test_idx is not None:
                            # REMOVE from extra_train and extra_val groups that are in the original test
                            extra_idx = [idx for idx in extra_idx if extra_groups[idx] not in groups[test_idx]]
                        # NOTE that we will not add anything to test, because there is no extra_test by design

                        if drop_val_groups_in_extra_train and val_idx is not None:
                            if split == 'train':
                                # REMOVE from extra_train groups that are in the original val
                                extra_idx = [idx for idx in extra_idx if extra_groups[idx] not in groups[val_idx]]
                            if split == 'val':
                                # ADD to extra_val groups that are in the original val but landed in extra_train
                                extra_idx = extra_idx + [idx for idx in extra_train_idx if
                                                         extra_groups[idx] in groups[val_idx]]

                    extra = copy.deepcopy(extra_train_dataset)
                    extra['images'] = np.array(extra['images'])[extra_idx].tolist()
                    extra['annotations'] = filter_annotations_with_images(extra)
                    train_val_test_dict[split] = merge_datasets(train_val_test_dict[split], extra)

        if n_folds != 1:
            dataset_splits[i] = train_val_test_dict
        else:
            dataset_splits = train_val_test_dict

    if n_folds != 1:
        assert len(dataset_splits) == n_folds
    for i in (dataset_splits if n_folds != 1 else [None]):
        if verbose and n_folds != 1:
            print(f'Fold {i}')
        train_val_test_dict = dataset_splits[i] if i is not None else dataset_splits
        assert_split_dataset_correct(
            split=train_val_test_dict,
            full=merge_datasets(dataset, extra_train_dataset) if extra_train_dataset else dataset,
            filename_groups=filename_groups, filename_groups_test_only=filename_groups_test_only,
            group_splitter=group_splitter, no_val=val_size == 0, no_test=test_size == 0 and n_folds == 1,
            drop_test_groups_in_extra_train=drop_test_groups_in_extra_train,
            verbose=verbose
        )
        if verbose and n_folds != 1:
            print('')

    if n_folds != 1:
        assert len(dataset_splits) == n_folds
        dataset_splits = [dataset_splits[i] for i in range(len(dataset_splits))]
    return dataset_splits


def save_dataset_splits(dataset_splits, n_folds, output_dir, out_format, segm, cat_names, image_root, copy_imgs,
                        data_filters, img_transforms, coinflip_transform, masked_bboxes_per_image, random_seed):
    print("save_dataset_splits:", out_format)
    assert_n_folds(n_folds)
    assert out_format.lower() in ['coco', 'yolo', 'yolo_hub']
    assert not out_format.lower().startswith('yolo') or image_root is not None
    assert not out_format.lower().startswith('yolo') or cat_names is not None

    if data_filters is None:
        data_filters = {}
    if img_transforms is None:
        img_transforms = {}

    _with_data_filters = any([data_filters.get(f'{split.upper()}_RMV_{size.upper()}', False)
                              for split in ['train', 'test', 'val'] for size in MIN_MAX_AREA])

    if copy_imgs == 'auto' and (_with_data_filters or any(img_transforms.values())):
        # assert n_folds == 1, '"img_transforms" or "_with_data_filters" not implemented for K-fold cross-validation'
        copy_imgs = True
    else:
        copy_imgs = False

    if copy_imgs and out_format.lower() == 'coco':
        new_image_root = os.path.join(output_dir, 'datasets', 'images')
        os.makedirs(new_image_root, exist_ok=True)
    else:
        new_image_root = None

    if out_format.lower() == 'yolo_hub':
        fn_prefix = os.path.join(output_dir, os.path.basename(output_dir))
        os.makedirs(os.path.dirname(fn_prefix), exist_ok=True)
    else:
        fn_prefix = os.path.join(output_dir, 'datasets', 'annotations')
        os.makedirs(os.path.dirname(fn_prefix), exist_ok=True)

    for i in range(n_folds) if n_folds != 1 else [None]:
        print(f'fold{i}' if n_folds != 1 else '')
        train_val_test_dict = dataset_splits[i] if i is not None else dataset_splits
        # "path" must be an absolute path, otherwise yolo searches for this path in "datasets" dir
        yolo_dict = {
            'names': list(cat_names),
            'path': '' if out_format.lower() == 'yolo_hub' else os.path.abspath(os.path.dirname(fn_prefix))
        }
        train_val_path = None
        for split in train_val_test_dict:
            masking_on = any([data_filters.get(f'{split.upper()}_RMV_{size.upper()}', False) for size in MIN_MAX_AREA])
            anns_per_image = get_annotations_dict(train_val_test_dict[split])
            assert not masking_on or masked_bboxes_per_image is not None
            if out_format.lower() == 'yolo_hub':
                out_name = get_data_name(
                    prefix=os.path.join(os.path.dirname(fn_prefix), split), random_seed=None, split=None, fold=i)
            else:
                out_name = get_data_name(
                    prefix=fn_prefix, random_seed=random_seed, split=split, fold=i)
            if out_format.lower() == 'coco':
                if copy_imgs:
                    for img in train_val_test_dict[split]['images']:
                        base_img_fn = os.path.join(f'fold{i}' if n_folds != 1 else '',
                                                   os.path.basename(img['file_name']))
                        new_fn = os.path.join(new_image_root, base_img_fn)
                        os.makedirs(os.path.dirname(new_fn), exist_ok=True)
                        shutil.copyfile(src=os.path.abspath(os.path.join(image_root, img['file_name'])), dst=new_fn)
                        img['file_name'] = base_img_fn
                        if masking_on and img['id'] in masked_bboxes_per_image:
                            clean_del_bboxes_xywh = preserve_unmasked_pixels(
                                del_bboxes_xywh=masked_bboxes_per_image[img['id']],
                                keep_bboxes_xywh=[ann['bbox'] for ann in anns_per_image[img['id']]]
                            )
                            mask_image(filename=new_fn, boxes_xywh=clean_del_bboxes_xywh)
                    if any(img_transforms.values()):
                        apply_img_transforms(
                            img_transforms=img_transforms, which_split=split, dataset=train_val_test_dict[split],
                            image_root=new_image_root, coinflip_transform=coinflip_transform)
                save_json(
                    dataset=train_val_test_dict[split],
                    fn=f'{out_name}.json',
                )
            elif out_format.lower().startswith('yolo'):
                coco_to_yolo(dataset=train_val_test_dict[split], output_dir=out_name, segm=segm)
                yolo_img_dir = os.path.join(out_name, 'images')
                remove_and_create_dir(yolo_img_dir)
                for img in train_val_test_dict[split]['images']:
                    new_fn = os.path.join(yolo_img_dir, os.path.basename(img['file_name']))
                    if copy_imgs:
                        shutil.copyfile(src=os.path.abspath(os.path.join(image_root, img['file_name'])), dst=new_fn)
                        if masking_on and img['id'] in masked_bboxes_per_image:
                            clean_del_bboxes_xywh = preserve_unmasked_pixels(
                                del_bboxes_xywh=masked_bboxes_per_image[img['id']],
                                keep_bboxes_xywh=[ann['bbox'] for ann in anns_per_image[img['id']]]
                            )
                            mask_image(filename=new_fn, boxes_xywh=clean_del_bboxes_xywh)
                        if any(img_transforms.values()):
                            apply_img_transforms(
                                img_transforms=img_transforms, which_split=split, dataset=train_val_test_dict[split],
                                image_root=yolo_img_dir, use_img_base_names=True, coinflip_transform=coinflip_transform)
                    else:
                        if not os.path.exists(new_fn):
                            os.symlink(src=os.path.abspath(os.path.join(image_root, img['file_name'])), dst=new_fn)
                        else:
                            print(f'WARNING: {os.path.basename(new_fn)} already exists (are you oversampling?)')
                if split == 'train_val':
                    raise ValueError('Not implemented')
                    train_val_path = os.path.join(os.path.basename(out_name), 'images')
                else:
                    yolo_dict[split] = os.path.join(os.path.basename(out_name), 'images')

        if out_format.lower().startswith('yolo'):
            out_name = fn_prefix if out_format.lower() == 'yolo_hub' else get_data_name(
                prefix=fn_prefix, random_seed=random_seed, split=None, fold=i)

            # yolo insists on having a "val" dataset
            if 'val' not in yolo_dict and 'test' in yolo_dict:
                yolo_dict['val'] = yolo_dict['test']
                yolo_dict.pop('test')
            save_yaml(yolo_dict, f'{out_name}.yaml')

            if train_val_path is not None:
                raise ValueError('Not implemented')
                yolo_dict['train'] = train_val_path
                yolo_dict['val'] = yolo_dict['test']
                yolo_dict.pop('test')
                save_yaml(yolo_dict, f'{out_name}.train_val.yaml')

    image_root = new_image_root if new_image_root is not None else image_root

    return fn_prefix, dataset_splits, image_root


def get_default_img_transforms():
    from .transforms import IMG_TRANSFORMS
    img_transforms = {
        f'{split.upper()}_{t.upper()}': False
        for t in IMG_TRANSFORMS
        for split in ['global', 'train', 'val', 'test']
    }
    return img_transforms


def set_img_transforms(img_transforms, which_splits, **transforms):
    from .transforms import IMG_TRANSFORMS
    assert all([transforms[t] in [True, False] for t in transforms])
    assert all([t in IMG_TRANSFORMS for t in transforms])
    if isinstance(which_splits, str):
        which_splits = [which_splits]

    for t in transforms:
        for split in which_splits:
            if transforms[t]:
                img_transforms[f'{split.upper()}_{t.upper()}'] = True


def apply_img_transforms(img_transforms, which_split, dataset, image_root, use_img_base_names=False,
                         coinflip_transform=False):
    from .transforms import IMG_TRANSFORMS
    if img_transforms is not None:
        # must be non-split dataset
        assert 'images' in dataset
        transform_funcs = {t: IMG_TRANSFORMS[t] for t in IMG_TRANSFORMS
                           if img_transforms[f'{which_split.upper()}_{t.upper()}']}
        transform_images(images=dataset['images'], image_root=image_root,
                         transform_funcs=list(transform_funcs.values()),
                         use_img_base_names=use_img_base_names,
                         info=f'{which_split}:{"|".join(list(transform_funcs.keys()))}',
                         coinflip_transform=coinflip_transform)


def transform_images(images, image_root, transform_funcs, use_img_base_names=False, info=None, coinflip_transform=False,
                     verbose=True):
    if not is_iterable(transform_funcs):
        transform_funcs = [transform_funcs]

    if len(transform_funcs) != 0:
        if verbose:
            print(f'Applying image transformation {info if info is not None else ""}')

        if coinflip_transform:
            coinflips = np.random.randint(0, 2, len(images))
        else:
            coinflips = np.ones((len(images)))
        assert len(coinflips) == len(images)
        assert np.isin(coinflips, [0, 1]).all()
        if not coinflip_transform:
            assert len(np.asarray(images)[coinflips == 1]) == len(images)

        for img in np.asarray(images)[coinflips == 1]:
            img_fn = img['file_name'] if not use_img_base_names else os.path.basename(img['file_name'])
            img_fn = os.path.join(image_root, img_fn)
            image = cv2.imread(img_fn)  # in BGR
            for f in transform_funcs:
                image = f(image)
            cv2.imwrite(img_fn, image)


def get_data_fn(prefix, random_seed, split=None, fold=None):
    return get_data_name(prefix=prefix, random_seed=random_seed, split=split, fold=fold, ext='json')


def get_data_name(prefix, random_seed, split=None, fold=None, ext=None):
    return f"{prefix}{f'.seed{random_seed}' if random_seed is not None else ''}{get_data_id(split=split, fold=fold)}{f'.{ext}' if ext is not None else ''}"


def get_data_id(split, fold=None):
    return f"{f'.fold{fold}' if fold is not None else ''}{f'.{split}' if split is not None else ''}"


def assert_split_dataset_correct(split, full, filename_groups=False, filename_groups_test_only=False,
                                 group_splitter='_',
                                 no_val=False, no_test=False, strict=False, drop_test_groups_in_extra_train=False,
                                 verbose=True):
    imgs, anns, cats, groups = {}, {}, {}, {}
    assert set(['train', 'val', 'test']).issuperset(split.keys())
    assert 'train' in split
    assert 'train_val' not in split
    if no_val:
        assert 'val' not in split  # and 'train_val' not in split
        imgs['val'], anns['val'] = [], []
    else:
        assert 'val' in split  # and 'train_val' in split
    if no_test:
        assert 'test' not in split
        imgs['test'], anns['test'] = [], []
    else:
        assert 'test' in split

    for s in split:
        assert_dataset_correct(split[s], strict=strict)
        imgs[s] = get_image_ids(split[s], id_field='file_name')
        anns[s] = get_annotation_hash_ids(split[s])
        cats[s] = set(get_annotation_ids(split[s], id_field='category_id'))
        groups[s] = set(get_filename_groups(split[s]['images'], split_char=group_splitter)) if filename_groups or (
                filename_groups_test_only and s == 'test') else set([])
        assert len(set(imgs[s])) == len(imgs[s])
        assert len(set(anns[s])) == len(anns[s])

    if not no_test:
        assert len(set(imgs['test']).intersection(imgs['train'])) == 0
        assert len(set(anns['test']).intersection(anns['train'])) == 0
    if not no_val:
        if not no_test:
            assert len(set(imgs['test']).intersection(imgs['val'])) == 0
            assert len(set(anns['test']).intersection(anns['val'])) == 0
        assert len(set(imgs['train']).intersection(imgs['val'])) == 0
        assert len(set(anns['train']).intersection(anns['val'])) == 0
        # assert set(imgs['train']).union(imgs['val']) == set(imgs['train_val'])
        # assert set(anns['train']).union(anns['val']) == set(anns['train_val'])

    if not drop_test_groups_in_extra_train:
        assert sorted(imgs['train'] + imgs['test'] + imgs['val']) == sorted(get_image_ids(full, id_field='file_name'))
        assert sorted(anns['train'] + anns['test'] + anns['val']) == sorted(get_annotation_hash_ids(full))

    # TODO
    if not no_test:
        assert cats['train'].issuperset(cats['test']), cats
        assert len(groups['train'].intersection(groups['test'])) == 0
    if not no_val:
        assert cats['train'].issuperset(cats['val']), cats
        assert len(groups['train'].intersection(groups['val'])) == 0

    if verbose:
        for labels_func in [get_labels_for_images]:  # , get_annot_sizes_as_labels_for_images]:
            print(f'Ratios based on: {labels_func}')
            for use_cat_combos in [False, True]:
                print(_format_dataset_info(full, name='full', use_cat_combos=use_cat_combos,
                                           full_n=len(full['images']), labels_func=labels_func))
                for s in ['train', 'val', 'test']:
                    if s in split:
                        print(_format_dataset_info(split[s], name=s, use_cat_combos=use_cat_combos,
                                                   full_n=len(full['images']), labels_func=labels_func))
                        # print(f'{s} domains: {len(set(groups[s]))}')
            print('')


def _format_dataset_info(dataset, name, use_cat_combos, full_n, labels_func=get_labels_for_images):
    ratios = get_class_ratios(dataset, use_cat_combos=use_cat_combos, labels_func=labels_func)
    n = len(dataset['images'])
    frac = n / full_n
    return f"{name:>5}{' (category combi sets)' if use_cat_combos else f' (N = {n:<5} {int(frac * 100):>3}%)'} --> {', '.join([f'{k}: {ratios[k]:.3f}' for k in ratios])}"


def get_class_ratios(dataset, use_cat_combos=False, labels_func=get_labels_for_images):
    if use_cat_combos:
        counts = Counter(bool2int(labels_func(dataset)))
        ratios = {
            f"set {cat_id}": counts[cat_id] / len(dataset['images']) for cat_id in sorted(list(counts.keys()))
        }
    else:
        labels, cat_names = labels_func(dataset, with_names=True)
        ratios = {
            cat_names[i]: r for i, r in enumerate(labels.sum(axis=0) / len(dataset['images']))
        }
    return ratios


def assert_categories_correct(dataset, strict=True):
    cat_ids = [cat['id'] for cat in dataset['categories']]
    cat_names = [cat['name'] for cat in dataset['categories']]
    assert cat_ids == list(range(1, len(cat_ids) + 1)), 'Category id with wrong format'
    assert len(cat_ids) == len(set([c.lower() for c in cat_names])), 'Duplicate category name (case-insensitive)'

    if 'annotations' in dataset:
        cat_ids_with_anns = set([ann['category_id'] for ann in dataset['annotations']])
        assert set(cat_ids).issuperset(
            cat_ids_with_anns), f'Unknown category id ({cat_ids_with_anns}) referenced in annotations (not in {cat_ids})'
        if strict:
            assert set(cat_ids) == cat_ids_with_anns, 'Unreferenced categories'


def assert_dataset_correct(dataset, strict=False, only_preds=False, only_imgs=False, allow_zero_area_boxes=False):
    assert allow_zero_area_boxes in ['warn', 'raise', True, False]
    assert not (only_preds and only_imgs)
    # all groups ("info", "licenses", etc.) are required
    # each group requires some fields, optional fields in comments
    COCO_FORMAT = {
        "info": [
            "description",  # "year", "version", "contributor", "url", "date_created"
        ],
        "licenses": [
            "id", "name",  # "url"
        ],
        "images": [
            "id", "file_name", "width", "height",  # "license",  # "flickr_url", "coco_url", "date_captured"
        ],
        "annotations": [
            "id", "image_id", "category_id", "bbox", "area", "iscrowd",  # "segmentation"
        ] if not only_preds else ["image_id", "category_id", "bbox", "score"],
        "categories": [
            "id", "name",  # "supercategory"
        ]
    }

    OPTIONAL = ['info', 'licenses'] + (
        ['images', 'categories'] if only_preds else ['annotations', 'categories'] if only_imgs else [])

    for group in COCO_FORMAT:
        assert group in dataset or group in OPTIONAL, f'{group} missing'
        for item in dataset.get(group, []) if group != 'info' else ([dataset[group]] if 'info' in dataset else []):
            if item is not None:
                for field in COCO_FORMAT[group]:
                    assert field in item, f'{group} {item if group != "info" else ""} missing {field}'

    annotations = dataset['annotations'] if not only_imgs or 'annotations' in dataset else []

    if not only_preds:
        anns = [ann['id'] for ann in annotations]
        assert len(set(anns)) == len(anns), f'Duplicate annotation id'
        if allow_zero_area_boxes is not True and not all([ann['area'] > 0 for ann in annotations]):
            err_msg = f"zero area boxes: {[ann['id'] for ann in annotations if ann['area'] <= 0]}"
            if allow_zero_area_boxes == 'warn':
                print(f'WARNING: {err_msg}')
            else:
                raise ValueError(err_msg)

    if not only_preds or 'images' in dataset:
        for id_field in ['id']:  # 'file_name' can be duplicated
            imgs = [img[id_field] for img in dataset['images']]
            assert len(set(imgs)) == len(imgs), f'Duplicate image {id_field}'

        imgs_with_anns = set([ann['image_id'] for ann in annotations])
        assert set(imgs).issuperset(imgs_with_anns), 'Unknown image id referenced in annotations'
        if strict:
            assert set(imgs) == imgs_with_anns, 'Unreferenced images'

    if (not only_preds and not only_imgs) or 'categories' in dataset:
        assert_categories_correct(dataset, strict=strict)

    if 'licenses' in dataset:
        for id_field in ['name', 'id']:
            lcs = [lc[id_field] for lc in dataset['licenses']]
    else:
        lcs = []
    assert len(set(lcs)) == len(lcs), f'Duplicate license {id_field}'

    if not only_preds or 'images' in dataset:
        # images without license are possible
        lcs_with_imgs = set([img.get('license') for img in dataset['images']])
        lcs_with_imgs = [lc for lc in lcs_with_imgs if lc is not None and not np.isnan(lc)]
        assert set(lcs).issuperset(
            lcs_with_imgs), f'Unknown license ({lcs_with_imgs}) referenced in images ({set(lcs)})'
        if strict:
            assert set(lcs) == lcs_with_imgs, 'Unreferenced licenses'


def get_license_name_from_id(dataset, lcs_id):
    if lcs_id is not None:
        lcs_name = [lcs['name'] for lcs in dataset['licenses'] if lcs['id'] == lcs_id]
        assert len(lcs_name) == 1, 'License ids are not unique'
        lcs_name = lcs_name[0]
    else:
        lcs_name = None
    return lcs_name


def merge_datasets(*datasets, check_licenses=True, check_optional_fields=True, duplicate_images=False, verbose=True):
    """
    Merges several COCO dataset into one dataset containing all images and annotations. To do so, it will issue
    new IDs whenever necessary.

    :param datasets: dictionaries in COCO format
    :param check_licenses: for two images with the same 'file_name', the associated license names must be equal
    :param check_optional_fields: equivalency of licenses/categories/images is determined based on the 'name'/
    'file_name' field; when check_optional_fields is True, all other fields (e.g. 'url', 'supercategory',
    'date_captured') must be also equivalent; if any of the fields is not equivalent, an exception is raised
    :return: merged dataset (a dictionary in COCO format)
    """

    new_dataset = empty_coco_dataset()

    info = None
    case_check_union = {'categories': set([]), 'licenses': set([])}
    for dataset in datasets:
        # check if dataset is in COCO format (quite strict but strict compliance makes merging easier)
        assert_dataset_correct(dataset)
        # use info from the first dataset with an info field
        if info is None:
            info = copy.copy(dataset.get('info'))
            if info is not None:
                if verbose:
                    print(f'INFO: using first available "info": {info}')
                new_dataset['info'] = info
        # merge all category and license names and check if it is safe to be case-insensitive
        for group in case_check_union:
            case_check_union[group] = case_check_union[group].union([item['name'] for item in dataset.get(group, [])])

    # case-insensitive check for category and license names
    # otherwise merging would result in two different categories such as "Person" and "person"
    for group in case_check_union:
        assert len(case_check_union[group]) == len(set([name.lower() for name in case_check_union[group]])), \
            f'Union of {group} cannot be done in a case-insensitive manner: {case_check_union[group]}'

    # dict of dicts to store old_ID-new_ID pairs
    # structure: new_ids_map[id_field][dataset_id][old_id] = [new_id]
    new_ids_map = {
        'license': defaultdict(dict),
        'category_id': defaultdict(dict),
        'image_id': defaultdict(dict)
    }

    # must be in this order because new_ids_map is being filled and later used within the same loop
    # group: licenses, categories, images, annotations
    # id_field: which field is used to reference this category (e.g. category_id for categories)
    # equal_field: which field establishes equivalency (two categories with the same name must be equal)
    # remap_id_fields: which fields needs remapping (e.g. image_id in annotations)
    for group, id_field, equal_field, remap_id_fields in [
        ('licenses', 'license', 'name', []),
        ('categories', 'category_id', 'name', []),
        ('images', 'image_id', 'file_name' if not duplicate_images else None, ['license']),
        ('annotations', None, None, ['image_id', 'category_id'])
    ]:
        next_id = 1
        merged_items = {}
        for dataset_id, dataset in enumerate(datasets):
            for item in dataset.get(group, []):  # item is an individual category/license/image/annotation
                # if equal_field is defined, only those not yet present will be added
                if equal_field is not None and item[equal_field] in merged_items:
                    new_item = merged_items[item[equal_field]]
                    assert item[equal_field] == new_item[equal_field]  # sanity check
                    if check_licenses and group == 'images':
                        # check if two images with the same name have the same license names
                        lcs_id1 = item.get('license')
                        lcs_id2 = new_item.get('license')
                        lcs_name1 = get_license_name_from_id(dataset, lcs_id1)
                        lcs_name2 = get_license_name_from_id(new_dataset, lcs_id2)
                        assert (lcs_id1 is None and lcs_id2 is None) or lcs_name1 == lcs_name2 or \
                               ((lcs_name1 is None or lcs_name1.lower() in ['', 'none']) and
                                (lcs_name2 is None or lcs_name2.lower() in ['', 'none'])), \
                            f'{group} {item[equal_field]} with different license names: {lcs_name1} vs. {lcs_name2}'
                    if check_optional_fields:
                        # I would expect also these to match (or be undefined)
                        for check_field in ['supercategory', 'url', 'width', 'height',
                                            'flickr_url', 'coco_url', 'date_captured']:
                            v1 = item.get(check_field)
                            v2 = new_item.get(check_field)
                            assert (v1 is None and v2 is None) or v1 == v2, \
                                f'{group} {item[equal_field]} with different values for {check_field}: {v1} vs. {v2}'

                # has not yet been added or there is no concept of a duplicate (e.g. annotations)
                else:
                    new_item = copy.copy(item)
                    new_item['id'] = next_id
                    # new_item_key is for storing this item in merged_items
                    # when equal_field is used, duplicates will be ignored (e.g. categories merging)
                    # when id is used, concatenation of all items is achieved (e.g. annotations merging)
                    new_item_key = new_item[equal_field if equal_field is not None else 'id']
                    assert new_item_key not in merged_items  # sanity check
                    merged_items[new_item_key] = new_item

                    # now remap old ids to new ids (license for images and image_id and category_id for annotations)
                    for f in remap_id_fields:
                        # this unfolds like this:
                        # img['license'] = remap_fields['license'][dataset_id][img['license']]
                        # ann['image_id'] = remap_fields['image_id'][dataset_id][ann['image_id']]
                        # ann['category_id'] = remap_fields['category_id'][dataset_id][ann['category_id']]
                        # the change happens in place (dictionary)
                        # this check is here because images without licenses (without license field) are allowed
                        if f in new_item:
                            if new_item[f] is not None:
                                new_item[f] = new_ids_map[f][dataset_id][new_item[f]]

                    # increment next_id
                    next_id += 1

                # if id_field is defined, then store new_ids_map for this group
                # this need to be done even if new_item was not added to merged_items
                if id_field is not None:
                    new_ids_map[id_field][dataset_id][item['id']] = new_item['id']

        new_dataset[group] = list(merged_items.values())
        if verbose:
            print(f'INFO: merged {group}: {f"{new_dataset[group]}" if group in ["licenses", "categories"] else f"{len(new_dataset[group])} record(s)"}')

        # if there were no licenses, remove the field
        if group == 'licenses' and new_dataset[group] == []:
            new_dataset.pop(group)

    return new_dataset


def voc_to_coco(img_dir, voc_dir, coco_fn, info=None,
                fix_voc_filenames=False, fix_cat_names=False, fix_cats=None, sort_cat_names=False):
    assert info is None or isinstance(info, (str, dict))

    from imgann import Convertor

    if fix_cats is None:
        fix_cats = {
            'C.capilata': 'C. capilata',
            'S.meleagris': 'S. meleagris',
            'pelagia': 'P. noctiluca',  # TODO: remove pelagia because it is really bad footage
            'C. tuberculata ': 'C. tuberculata'
        }

    if fix_voc_filenames:
        from xml.etree import ElementTree
        for fn in os.listdir(voc_dir):
            assert fn.endswith('.xml')
            tree = ElementTree.parse(os.path.join(voc_dir, fn))
            root = tree.getroot()
            filenames = [fn_element for fn_element in root.findall('filename')]
            assert len(filenames) == 1
            filenames[0].text = fn[::-1].replace('.xml'[::-1], '.jpg'[::-1], 1)[::-1]

            if fix_cat_names:
                for object in root.findall('object'):
                    names = [name for name in object.findall('name')]
                    assert len(names) == 1
                    if names[0].text.endswith('\n'):
                        names[0].text = names[0].text[:-1]
                    if names[0].text in fix_cats:
                        names[0].text = fix_cats[names[0].text]

            tree.write(os.path.join(voc_dir, fn))

    Convertor.voc2coco(dataset_dir=img_dir, voc_ann_dir=voc_dir, save_dir=coco_fn, center=False)
    voc_dataset = read_json(coco_fn, assert_correct=False, verbose=False)

    dataset = empty_coco_dataset(info=info)

    for group in dataset:
        if group in voc_dataset:
            dataset[group] = voc_dataset[group]

    for ann in dataset['annotations']:
        ann['area'] = int(ann['area'])
        ann['iscrowd'] = 0
        ann['segmentation'] = []
        if 'ignore' in ann:
            assert ann['ignore'] in [0, '0']
            ann.pop('ignore')

    if sort_cat_names:
        sort_categories(dataset)

    save_json(dataset, coco_fn, assert_correct=True)
    # final check:
    dataset = read_json(coco_fn)

    return dataset


def assure_consistent_cat_names(dataset, predict_cat_names=None):
    if 'categories' in dataset:
        _check_cat_names = get_category_names(dataset)
        if predict_cat_names is None:
            predict_cat_names = _check_cat_names
        else:
            assert list(predict_cat_names) == _check_cat_names, (list(predict_cat_names), _check_cat_names)
    return predict_cat_names


def prepare_prediction_dataset_on_disk(
        input_fn, img_dir, output_dir, every_n_frame=None, every_n_sec=None, fast_forward=True,
        cat_names=None, img_transforms=None, out_json_fn=None, use_fake_category=False
):
    assert not (input_fn is None and img_dir is None)

    if isinstance(input_fn, str):
        input_fn = [input_fn]
    if isinstance(img_dir, str):
        img_dir = [img_dir]

    if input_fn is not None and img_dir is not None:
        assert all([fn.lower().endswith('.json') or fn.lower().endswith('.json.gz') for fn in input_fn])
        assert len(img_dir) == 1 or len(img_dir) == len(input_fn)
        dataset, img_dir = read_all_datasets(input_fn, img_dir)
        assert cat_names is None or list(cat_names) == get_category_names(dataset)
        # no subsetting because img_dir can be a root
        # dataset = subset_dataset_to_imgs(dataset, img_dir)
    else:
        if input_fn is not None and len(input_fn) == 1 and is_video(input_fn[0]):
            assert img_dir is None
            input_fn = input_fn[0]
            img_dir = os.path.join(output_dir, 'frames')
            os.makedirs(img_dir, exist_ok=True)
            frames, _ = extract_frames(
                input_fn, img_dir, every_n_sec=every_n_sec, every_n_frame=every_n_frame,
                fast_forward=fast_forward)
        elif input_fn is not None and len(input_fn) == 1 and is_image(input_fn[0]):
            assert img_dir is None
            input_fn = input_fn[0]
            img_dir = os.path.dirname(input_fn)
        elif img_dir is not None and len(img_dir) == 1:
            assert input_fn is None
            img_dir = img_dir[0]
        else:
            raise ValueError(f'Wrong inputs. input_fn: {input_fn} and img_dir: {img_dir}')

        dataset = create_coco_dataset(
            *(frames if input_fn is not None and is_video(input_fn) else
              [input_fn] if input_fn is not None and is_image(input_fn) else
              [img_dir]),
            cat_names=cat_names
        )
        assert len(dataset['annotations']) == 0
        dataset.pop('annotations')
        if cat_names is None:
            assert len(dataset['categories']) == 0
            if use_fake_category:
                dataset['categories'] = [dict(id=1, name=FAKE_CATEGORY, supercategory="")]
            else:
                dataset.pop('categories')

    data_json_fn = out_json_fn if out_json_fn else os.path.join(output_dir, 'inputs.json')
    os.makedirs(os.path.dirname(data_json_fn), exist_ok=True)
    save_json(dataset, data_json_fn, only_imgs=True)

    if any(img_transforms.values()):
        new_img_dir = os.path.join(output_dir, 'img_transformed')
        for img in dataset['images']:
            new_img_fn = os.path.join(new_img_dir, img['file_name'])
            os.makedirs(os.path.dirname(new_img_fn), exist_ok=True)
            shutil.copyfile(src=os.path.join(img_dir, img['file_name']), dst=new_img_fn)
        apply_img_transforms(img_transforms=img_transforms, which_split='global', dataset=dataset,
                             image_root=new_img_dir)
        img_dir = new_img_dir

    return dataset, data_json_fn, img_dir


def experiment_config(
        filename, categories, data_json_fn, data_img_dir, test_json_fn, test_img_dir, extra_train_json_fn,
        extra_train_img_dir,
        n_folds=None, fold=None, test_size=None, val_size=None, multi_label=None, filename_groups=False,
        filename_groups_test_only=False,
        data_filters=None, img_transforms=None, use_sharper_scheduler=None, target_decay_epoch=None
):
    from yacs.config import CfgNode as CN

    C = CN()
    C.CATEGORIES = categories
    C.DATA_JSON_FN = data_json_fn
    C.DATA_IMG_DIR = data_img_dir
    C.TEST_JSON_FN = test_json_fn
    C.TEST_IMG_DIR = test_img_dir
    C.EXTRA_TRAIN_JSON_FN = extra_train_json_fn
    C.EXTRA_TRAIN_IMG_DIR = extra_train_img_dir
    C.OUTPUT_DIR = os.path.dirname(filename)

    assert n_folds is None or n_folds > 0
    C.N_FOLDS = int(n_folds) if n_folds is not None and n_folds != 1 else None
    if C.N_FOLDS is not None:
        C.FOLD = fold
    else:
        assert fold is None

    assert test_size is None or test_size >= 0
    if test_size is not None and test_size >= 1:
        test_size = int(test_size)
    C.TEST_SIZE = test_size if test_size is not None and test_size != 0 else None

    assert val_size is None or val_size >= 0
    if val_size is not None and val_size >= 1:
        val_size = int(val_size)
    C.VAL_SIZE = val_size if val_size is not None and val_size != 0 else None

    C.FILENAME_GROUPS = filename_groups if filename_groups is not None else False
    assert C.FILENAME_GROUPS in [True, False]

    C.FILENAME_GROUPS_TEST_ONLY = filename_groups_test_only if filename_groups_test_only is not None else False
    assert C.FILENAME_GROUPS_TEST_ONLY in [True, False]

    C.MULTI_LABEL = multi_label if multi_label is not None else False
    assert C.MULTI_LABEL in [True, False]

    if data_filters is not None and any(data_filters.values()):
        C['DATA_FILTERS'] = CN()
        for f_name, f_value in data_filters.items():
            if f_value:
                C['DATA_FILTERS'][f_name.replace("-", "")] = f_value

    if img_transforms is not None and any(img_transforms.values()):
        C['IMG_TRANSFORMS'] = CN()
        for f_name, f_value in img_transforms.items():
            if f_value:
                C['IMG_TRANSFORMS'][f_name.replace("-", "")] = f_value

    if use_sharper_scheduler is not None:
        assert isinstance(use_sharper_scheduler, bool)
        assert not use_sharper_scheduler or target_decay_epoch is not None
        C.SCHEDULER = CN()
        C.SCHEDULER.SHARP = use_sharper_scheduler
        C.SCHEDULER.DECAY_EPOCH = target_decay_epoch if use_sharper_scheduler else None
    else:
        assert target_decay_epoch is None

    save_yacs(C, filename)


def draw_box(img, bbox, label=None, color=[255, 0, 0]):
    start_point = (int(bbox[0]), int(bbox[1]))
    end_point = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(
        img, start_point, end_point, color=tuple(color), thickness=2
    )
    if label is not None:
        cv2.putText(
            img, label, (start_point[0], start_point[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 1, color=tuple(color), thickness=2
        )


def draw_boxes_natively(model, result, img=None, filename=None, cat_names=None, format=BGR_FORMAT):
    if model == 'yolo8':
        if cat_names is not None:
            orig_names = copy.copy(result.names)
            result.names = cat_names
        if img is None:
            img = result.plot()
        if cat_names is not None:
            result.names = orig_names
        if filename is not None:
            cv2.imwrite(filename, img)
        return img
    elif model == 'dt2':
        from detectron2.utils.visualizer import Visualizer
        assert isinstance(img, np.ndarray)
        assert format in [BGR_FORMAT, RGB_FORMAT]
        if format == BGR_FORMAT:
            img = img[:, :, ::-1]
        kwargs = dict(metadata=dict(thing_classes=cat_names)) if cat_names is not None else dict()
        visualizer = Visualizer(img_rgb=img, **kwargs)
        vis = visualizer.draw_instance_predictions(result.to("cpu"))
        if filename is not None:
            vis.save(filename)
        return vis
    else:
        raise ValueError(f'Unsupported model {model}')


def read_image(image_fn, format=BGR_FORMAT):
    assert format in [BGR_FORMAT, RGB_FORMAT]

    image = cv2.imread(image_fn)
    if image is None:
        raise ValueError(f'Cannot read {image_fn}')
    if format == RGB_FORMAT:
        image = image[:, :, ::-1]

    return image


def write_image(image, image_fn, flip_channels=False):
    if flip_channels:
        image = image[:, :, ::-1]
    cv2.imwrite(image_fn, image)


def read_image_as_b64(image_fn, format=BGR_FORMAT):
    image = read_image(image_fn=image_fn, format=format)
    _, buffer = cv2.imencode(img=image, ext=f'.{image_fn.split(".")[-1]}')
    b64_img_buffer = base64.b64encode(buffer)
    return b64_img_buffer


def convert_b64_to_numpy(b64_img_buffer, flip_channels=False):
    img_array_1d = np.frombuffer(base64.decodebytes(b64_img_buffer), dtype=np.uint8)
    # NOTE: image in the same format (RGB or BGR) as was originally encoded
    image = cv2.imdecode(img_array_1d, flags=cv2.IMREAD_COLOR)
    if flip_channels:
        image = image[:, :, ::-1]
    return image


def remove_and_create_dir(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)


def coco_to_yolo(dataset, output_dir, segm=False):
    remove_and_create_dir(os.path.join(output_dir, 'labels'))
    images = {img['id']: img for img in dataset['images']}
    annotations = get_annotations_dict(dataset)
    # Write labels file
    for img_id in annotations:
        img = images[img_id]
        labels = []
        for annot in annotations[img_id]:
            if not segm:
                coord = np.array(annot['bbox'], dtype=np.float64)
                coord[:2] += coord[2:] / 2  # xy top-left corner to center
                coord[[0, 2]] /= img['width']  # normalize x
                coord[[1, 3]] /= img['height']  # normalize y
                assert coord[2] > 0 and coord[3] > 0, annot
            else:
                if len(annot["segmentation"]) > 1:
                    coord = merge_multi_segment(annot["segmentation"])
                    coord = (np.concatenate(coord, axis=0) / np.array([img['width'], img['height']])).reshape(-1)
                else:
                    coord = [j for i in annot["segmentation"] for j in i]  # all segments concatenated
                    coord = (np.array(coord).reshape(-1, 2) / np.array([img['width'], img['height']])).reshape(-1)

            assert annot['category_id'] >= 1
            label = [annot['category_id'] - 1] + coord.tolist()
            if label not in labels:
                labels.append(label)

        assert is_image(img['file_name'])
        with open(os.path.join(
                output_dir, 'labels',
                f'{".".join(os.path.basename(img["file_name"]).split(".")[:-1])}.txt'), 'a') as f:
            for label in labels:
                label = *label,
                f.write(('%g ' * len(label)).rstrip() % label + '\n')


def fill_polygon(img, coco_segmentation, out_fn=None):
    if isinstance(img, str):
        img = read_image(img)

    if len(coco_segmentation) > 1:
        segments = [np.concatenate(merge_multi_segment(coco_segmentation), axis=0).astype(int).reshape(-1, 2)]
    else:
        assert len(coco_segmentation) == 1
        segments = [np.array(coco_segmentation[0], dtype=int).reshape(-1, 2)]

    new_img = cv2.fillPoly(img, pts=segments, color=(0, 0, 255))
    if out_fn is not None:
        write_image(new_img, out_fn)
    return new_img


def _min_index(arr1, arr2):
    """
    Copyright (c) 2024 Ultralytics
    AGPL-3.0 license
    https://github.com/ultralytics/ultralytics

    Find a pair of indexes with the shortest distance.

    Args:
        arr1: (N, 2).
        arr2: (M, 2).
    Return:
        a pair of indexes(tuple).
    """
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)


def merge_multi_segment(segments):
    """
    Copyright (c) 2024 Ultralytics
    AGPL-3.0 license
    https://github.com/ultralytics/ultralytics

    GNU AFFERO GENERAL PUBLIC LICENSE
    Merge multi segments to one list. Find the coordinates with min distance between each segment, then connect these
    coordinates with one thin line to merge all segments into one.

    Args:
        segments(List(List)): original segmentations in coco's json file.
            like [segmentation1, segmentation2,...],
            each segmentation is a list of coordinates.
    """
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    # record the indexes with min distance between each segment
    for i in range(1, len(segments)):
        idx1, idx2 = _min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # use two round to connect all the segments
    for k in range(2):
        # forward connection
        if k == 0:
            for i, idx in enumerate(idx_list):
                # middle segments have two indexes
                # reverse the index of middle segments
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]

                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                # deal with the first segment and the last one
                if i in [0, len(idx_list) - 1]:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0]: idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in [0, len(idx_list) - 1]:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s


def yolo_to_coco(dataset_dir, dataset_info):
    img_id, ann_id = 0, 0
    dataset = empty_coco_dataset(info=dataset_info)
    cat_ids = set([])
    for fn in os.listdir(os.path.join(dataset_dir, 'labels')):
        if fn.endswith('.txt'):
            img_id += 1
            img_fn = f'{fn[:-4]}.jpg'
            height, width = read_image(os.path.join(dataset_dir, 'images', img_fn)).shape[:2]
            dataset['images'].append(dict(
                id=img_id,
                file_name=img_fn,
                height=height,
                width=width
            ))
            with open(os.path.join(dataset_dir, 'labels', fn), mode='rt') as f:
                for line in f:
                    assert len(line) != 0
                    cat_id, x, y, w, h = line.strip().split()
                    cat_id = int(cat_id) + 1
                    cat_ids.add(cat_id)
                    bbox = np.asarray([float(x), float(y), float(w), float(h)])
                    bbox[[0, 2]] *= width  # unnormalize x
                    bbox[[1, 3]] *= height  # unnormalize y
                    bbox[:2] -= bbox[2:] / 2  # xy center to top-left corner
                    ann_id += 1
                    dataset['annotations'].append(dict(
                        id=ann_id,
                        image_id=img_id,
                        category_id=cat_id,
                        area=width * height,
                        bbox=bbox.tolist(),
                        iscrowd=0,
                        segmentation=[]
                    ))

    with open(os.path.join(dataset_dir, 'classes.txt'), mode='rt') as f:
        for i, line in enumerate(f):
            dataset['categories'].append(dict(
                id=i + 1,
                name=line.strip(),
                supercategory="none"
            ))
    assert set([c['id'] for c in dataset['categories']]) == cat_ids
    return dataset


def aims_to_coco(aims_fn, dataset_info, batch_key):
    img_id, ann_id = 0, 0
    dataset = empty_coco_dataset(info=dataset_info)
    categories = {}
    with open(aims_fn, mode='rt', encoding='UTF-8') as f:
        for line in f:
            img = json.loads(line.strip())

            try:
                img_id += 1
                img_fn = img['source-ref']
                assert len(img[batch_key]['image_size']) == 1, img[batch_key]['image_size']
                width = int(img[batch_key]['image_size'][0]['width'])
                height = int(img[batch_key]['image_size'][0]['height'])
                dataset['images'].append(dict(
                    id=img_id,
                    file_name=img_fn,
                    height=height,
                    width=width
                ))

                for cat_id in img[f'{batch_key}-metadata']['class-map']:
                    if int(cat_id) + 1 in categories:
                        assert categories[int(cat_id) + 1] == img[f'{batch_key}-metadata']['class-map'][cat_id]
                    else:
                        categories[int(cat_id) + 1] = img[f'{batch_key}-metadata']['class-map'][cat_id]

                for r in img[batch_key]['annotations']:
                    try:
                        cat_id = int(r['class_id']) + 1
                        assert cat_id in categories
                        x = float(r['left'])
                        y = float(r['top'])
                        w = float(r['width'])
                        h = float(r['height'])
                        ann_id += 1
                        dataset['annotations'].append(dict(
                            id=ann_id,
                            image_id=img_id,
                            category_id=cat_id,
                            area=width * height,
                            bbox=[x, y, w, h],
                            iscrowd=0,
                            segmentation=[]
                        ))
                    except Exception as e:
                        print(f'WARNING: {e} --> skipping region {r}')
            except Exception as e:
                print(f'WARNING: {e} --> skipping image {img}')

    for cat_id in categories:
        dataset['categories'].append(dict(
            id=cat_id,
            name=categories[cat_id],
            supercategory="none"
        ))

    return dataset


def aims_species_to_coco(aims_fn, dataset_info, img_dir):
    with open(aims_fn, mode='rt', encoding='UTF-8') as f:
        aims_dict = json.load(f)

    img_id, ann_id = 0, 0
    dataset = empty_coco_dataset(info=dataset_info)
    categories = []
    for img in aims_dict['_via_img_metadata'].values():
        try:
            img_id += 1
            img_fn = img['filename']
            height, width = read_image(os.path.join(img_dir, img_fn)).shape[:2]
            dataset['images'].append(dict(
                id=img_id,
                file_name=img_fn,
                height=height,
                width=width
            ))
            for r in img['regions']:
                try:
                    assert r['shape_attributes']['name'] == 'rect'
                    category = r['region_attributes']['label']
                    if category not in categories:
                        categories.append(category)
                    cat_id = categories.index(category) + 1
                    x = float(r['shape_attributes']['x'])
                    y = float(r['shape_attributes']['y'])
                    w = float(r['shape_attributes']['width'])
                    h = float(r['shape_attributes']['height'])
                    ann_id += 1
                    dataset['annotations'].append(dict(
                        id=ann_id,
                        image_id=img_id,
                        category_id=cat_id,
                        area=width * height,
                        bbox=[x, y, w, h],
                        iscrowd=0,
                        segmentation=[]
                    ))
                except Exception as e:
                    print(f'WARNING: {e} --> skipping region {r}')
        except Exception as e:
            print(f'WARNING: {e} --> skipping image {img}')

    for i, cat_name in enumerate(categories):
        dataset['categories'].append(dict(
            id=i + 1,
            name=cat_name,
            supercategory="none"
        ))

    return dataset


def model_warm_up(model, h=1024, w=1024, data_mapper=None, kwargs=None):
    print('Warming up...')
    if kwargs is None:
        kwargs = dict()
    for _ in range(5):
        rand_rbg = np.random.random_integers(low=0, high=255, size=h * w * 3).reshape((h, w, 3)).astype(np.uint8)
        if data_mapper is not None:
            rand_rbg = data_mapper(rand_rbg.copy())
        model(rand_rbg, **kwargs)


def preserve_unmasked_pixels(del_bboxes_xywh, keep_bboxes_xywh):
    clean_del_bboxes_xywh = set([])
    for del_bbox_xywh in del_bboxes_xywh:
        del_boxes = set([Rectangle(*from_XYWH_to_XYXY(del_bbox_xywh))])
        for keep_bbox_xywh in keep_bboxes_xywh:
            keep_box = Rectangle(*from_XYWH_to_XYXY(keep_bbox_xywh))
            new_del_boxes = set([])
            for del_box in del_boxes:
                if (del_box & keep_box) is not None:
                    new_del_boxes = new_del_boxes.union(list(del_box - keep_box))
                else:
                    new_del_boxes = new_del_boxes.union([del_box])
            del_boxes = new_del_boxes
        clean_del_bboxes_xywh = clean_del_bboxes_xywh.union(del_boxes)
    return [b.xywh() for b in clean_del_bboxes_xywh]


def mask_image(filename, boxes_xywh, color=None, out_filename=None, blur_radius=10):
    assert color in [None, 'average', 'blur'] or (is_iterable(color) and len(color) == 3)
    if color is None:
        color = 'blur'

    if color == 'blur':
        # Open an image
        im = Image.open(filename)
        blurred = im.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        mask = Image.new('L', im.size, 0)
        draw = ImageDraw.Draw(mask)
        for box_xywh in boxes_xywh:
            start_point = (int(box_xywh[0]), int(box_xywh[1]))
            end_point = (int(box_xywh[0] + box_xywh[2]), int(box_xywh[1] + box_xywh[3]))
            draw.rectangle([start_point, end_point], fill=255)
        im.paste(blurred, mask=mask)
        im.save(out_filename if out_filename is not None else filename)
    else:
        img = cv2.imread(filename)
        if color == 'average':
            color = img.mean(axis=0).mean(axis=0)
            assert is_iterable(color) and len(color) == 3

        for box_xywh in boxes_xywh:
            start_point = (int(box_xywh[0]), int(box_xywh[1]))
            end_point = (int(box_xywh[0] + box_xywh[2]), int(box_xywh[1] + box_xywh[3]))
            cv2.rectangle(
                img, start_point, end_point, color=tuple(color), thickness=-1
            )
        cv2.imwrite(out_filename if out_filename is not None else filename, img)


def bboxes_for_category(dataset, cat_id, as_dict=True):
    bboxes_per_image = defaultdict(list)
    for ann in dataset['annotations']:
        if ann['category_id'] == cat_id:
            bboxes_per_image[ann['image_id']].append(ann['bbox'])

    if as_dict:
        return bboxes_per_image
    else:
        return [bbox for img in bboxes_per_image for bbox in bboxes_per_image[img]]


def split_groups_into_train_test(dataset, test_groups, group_splitter, group_splitter_n_parts=1):
    groups = get_filename_groups(dataset['images'], split_char=group_splitter, n_parts=group_splitter_n_parts)
    test_images = np.isin(groups, test_groups)

    splits = {}
    for split, mask in [
        ('train', ~test_images),
        ('test', test_images),
    ]:
        splits[split] = copy.deepcopy(dataset)
        splits[split]['images'] = np.array(splits[split]['images'])[mask].tolist()
        splits[split]['annotations'] = filter_annotations_with_images(splits[split])
    return splits['train'], splits['test']


def remap_categories(dataset, new_categories):
    assert set([c['name'] for c in dataset['categories']]) == set(
        new_categories.values()), '"new_categories" do not match existing categories'
    cat_name_to_id = get_category_names(dataset, as_dict=True, strict=False)
    old_to_new = {cat_name_to_id[new_name]: new_id for new_id, new_name in new_categories.items()}
    for annot in dataset['annotations']:
        annot['category_id'] = old_to_new[annot['category_id']]
    for cat in dataset['categories']:
        cat['id'] = old_to_new[cat['id']]
    dataset['categories'] = sorted(dataset['categories'], key=lambda x: x['id'])


def rename_category(dataset, rename_dict):
    for cat in dataset['categories']:
        if cat['name'] in rename_dict:
            cat['name'] = rename_dict[cat['name']]


def merge_categories(dataset, new_name, new_id=1, new_supercategory=""):
    dataset['categories'] = [dict(id=new_id, name=new_name, supercategory=new_supercategory)]
    for ann in dataset['annotations']:
        ann['category_id'] = new_id


def select_category(dataset, category_id, new_category_id=None):
    assert not is_iterable(category_id) or new_category_id is None

    category_ids = [category_id] if not is_iterable(category_id) else category_id
    dataset['annotations'] = [ann for ann in dataset['annotations'] if ann['category_id'] in category_ids]

    if new_category_id is not None:
        dataset['categories'] = [cat for cat in dataset['categories'] if cat['id'] == category_id]
        assert len(dataset['categories']) == 1
        dataset['categories'][0]['id'] = new_category_id
        for ann in dataset['annotations']:
            ann['category_id'] = new_category_id


def merge_categories(dataset):
    dataset['categories'] = [{'id': 1, 'name': 'Jellyfish', 'supercategory': ''}]
    for ann in dataset['annotations']:
        ann['category_id'] = 1


def split_categories(dataset, out_fn, img_dir, output_dir):
    for cat in dataset['categories']:
        D = copy.deepcopy(dataset)
        cat_id = cat['id']
        cat_name = cat['name']
        print(f'\n{cat_name}')
        D['categories'] = [{'id': 1, 'name': cat_name, 'supercategory': ''},
                           {'id': 2, 'name': 'delete', 'supercategory': ''}]
        for ann in D['annotations']:
            if ann['category_id'] == cat_id:
                ann['category_id'] = 1
            else:
                ann['category_id'] = 2

        save_json(D, out_fn.format(cat_name=cat_name))
        os.makedirs(output_dir.format(cat_name=cat_name), exist_ok=True)
        mask_images(output_dir=output_dir.format(cat_name=cat_name),
                    annot_fn=out_fn.format(cat_name=cat_name),
                    img_dir=img_dir, delete_label='delete',
                    color='blur', overwrite=True)
        D = read_json(out_fn.format(cat_name=cat_name))
        D['images'] = filter_images_with_annotations(D)
        save_json(D, out_fn.format(cat_name=cat_name))

        annot_imgs = [img['file_name'] for img in D['images']]
        dir_imgs = read_files_from_dir(dir_name=output_dir.format(cat_name=cat_name),
                                       filter_func=is_image, basename_only=True)
        for fn in [fn for fn in dir_imgs if fn not in annot_imgs]:
            os.remove(os.path.join(output_dir.format(cat_name=cat_name), fn))
        assert len(read_files_from_dir(dir_name=output_dir.format(cat_name=cat_name),
                                       filter_func=is_image, basename_only=True)) == len(D['images'])


def remove_category(dataset, category_id, verbose=False):
    new_anns = []
    for ann in dataset['annotations']:
        if ann['category_id'] == category_id:
            if verbose:
                print(f'{ann["category_id"]} REMOVED')
            pass
        else:
            if ann['category_id'] > category_id:
                if verbose:
                    print(f'{ann["category_id"]} changed to', end=' ')
                ann['category_id'] -= 1
                if verbose:
                    print(f'{ann["category_id"]}')
            else:
                if verbose:
                    print(f'{ann["category_id"]} kept')
            new_anns.append(ann)

    new_cats = []
    for cat in dataset['categories']:
        if cat['id'] == category_id:
            if verbose:
                print(f'{cat} REMOVED')
            pass
        else:
            if cat['id'] > category_id:
                if verbose:
                    print(f'{cat} changed to', end=' ')
                cat['id'] -= 1
                if verbose:
                    print(f'{cat}')
            else:
                if verbose:
                    print(f'{cat} kept')
            new_cats.append(cat)

    dataset['categories'] = new_cats
    dataset['annotations'] = new_anns


def get_video_fn(name, partner, frames=False, annot=False, version=None):
    assert sum([frames, annot]) < 2
    from config import ANNOT_ROOT
    modality = 'frames' if frames else 'annotations' if annot else 'videos'
    if frames or annot:
        name = name[:-4] if is_video(name) else name
    if version is not None:
        name = f'{name}.{version}'
    return os.path.join(ANNOT_ROOT, modality, partner, f"{name}.json" if annot else name)


def video_list_df(sheet_names):
    from config import ANNOT_ROOT
    fn = os.path.join(ANNOT_ROOT, 'BE_CRC_jelly_videos.xlsx')
    df = []
    for partner in sheet_names:
        df.append(pd.read_excel(fn, sheet_name=partner))
        df[-1]['Partner'] = partner
    df = pd.concat(df).reset_index(drop=True)
    return df


def mask_images(annot_fn, img_dir, delete_label, color=None, case_sensitive=False, output_dir=None, overwrite=False):
    dataset = read_json(annot_fn)
    cat_dict = get_category_names(dataset, lower_case=not case_sensitive, as_dict=True)
    delete_label = delete_label.lower() if not case_sensitive else delete_label
    assert delete_label in cat_dict, f'"{delete_label}" not a category: {list(cat_dict.keys())}'
    delete_id = cat_dict[delete_label]
    bboxes_per_image = bboxes_for_category(dataset, cat_id=delete_id, as_dict=True)
    print(f'Masking {len(bboxes_per_image)} images...')
    for img in dataset['images']:
        filename = os.path.join(img_dir, img['file_name'])
        out_filename = filename if output_dir is None else \
            os.path.join(output_dir, os.path.basename(img['file_name']))
        # if output_dir was specified copy the original image
        if output_dir is not None:
            shutil.copyfile(src=filename, dst=out_filename)
        if img['id'] in bboxes_per_image:
            mask_image(filename=filename, boxes_xywh=bboxes_per_image[img['id']],
                       color=color, out_filename=out_filename)

    print(f'Removing {delete_label} category...')
    remove_category(dataset=dataset, category_id=delete_id, verbose=False)

    if overwrite:
        out_filename = annot_fn
    else:
        out_filename = f'{annot_fn[:-5] if annot_fn.endswith(".json") else annot_fn}.without_{delete_label}.json'
    save_json(dataset=dataset, fn=out_filename)


def get_size(obj, seen=None):
    """
    CREDIT: https://stackoverflow.com/questions/45393694/size-of-a-dictionary-in-bytes
    Recursively finds size of objects
    """
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


def predictions_to_df(dataset=None, pred_fn=None, categories=None, input_fn=None, split_words=['.frame_', '.'],
                      frame_id_type=pd.Int64Dtype(), fish_id_type=pd.Int64Dtype(), box_id_type=pd.Int64Dtype(),
                      all_tracks_start_at_one=True, fix_fps_mod=None):
    if dataset is None:
        if input_fn is not None:
            dataset = read_json(input_fn)
            dataset['annotations'] = read_json(pred_fn, only_preds=True)['annotations']
        else:
            dataset = read_json(pred_fn, assert_correct=False, fix_to_basenames=True, fix_image_ids=True,
                                fill_in_images=True, verbose=False)

        if categories is not None:
            if isinstance(categories, str) and os.path.exists(categories):
                dataset['categories'] = read_json(categories)['categories']
            else:
                assert is_iterable(categories)
                if all([isinstance(c, str) for c in categories]):
                    dataset['categories'] = [{'id': i + 1, 'name': c} for i, c in enumerate(categories)]
                else:
                    assert all([isinstance(c, dict) for c in categories])
                    dataset['categories'] = categories
        else:
            assert isinstance(dataset['categories'], dict)
    else:
        pass

    records = []
    cats = get_category_ids(dataset, as_dict=True)
    anns = get_annotations_dict(dataset)
    for img in dataset['images']:
        if img['id'] in anns:
            for ann in anns[img['id']]:
                records.append([
                    os.path.basename(img['file_name'].split(split_words[0])[0]),
                    ann['track_id'] if 'track_id' in ann else None,
                    img['file_name'].split(split_words[0])[1].split(split_words[1])[0],
                    ann['bbox'][0],
                    ann['bbox'][1],
                    ann['bbox'][2],
                    ann['bbox'][3],
                    ann['bbox'][2] * ann['bbox'][3],
                    cats[ann['category_id']],
                    ann['score'] if 'score' in ann else None,
                    # ann['id'],
                    img['file_name']
                ])

    df = pd.DataFrame.from_records(
        records,
        columns=['video_id', 'fish_id', 'frame_id', 'x1', 'y1', 'width', 'height', 'area', 'label', 'score', 'filename']
    )  # 'box_id'
    df['video_id'] = df['video_id'].astype(str)
    df['frame_id'] = df['frame_id'].astype(frame_id_type)
    df['fish_id'] = df['fish_id'].astype(fish_id_type)
    # df['box_id'] = df['box_id'].astype(box_id_type)
    df = df.sort_values(['video_id', 'fish_id', 'frame_id'])

    if all_tracks_start_at_one:
        df['fish_id'] -= (df['fish_id'].min() - 1)

    if fix_fps_mod is not None and fix_fps_mod != 1:
        for video_id in np.unique(df['video_id']):
            video_mask = df['video_id'] == video_id
            idx = np.arange(df.loc[video_mask, 'frame_id'].min(), df.loc[video_mask, 'frame_id'].max() + 1, fix_fps_mod, dtype=int)
            assert df.loc[video_mask, 'frame_id'].values.isin(idx).all()
            df.loc[video_mask, 'frame_id'] = df.loc[video_mask, 'frame_id'].map(
                {i: j for i, j in zip(idx, np.arange(len(idx)))})

    return df


def assign_fish_new_ids(df, max_missing_frames):
    df = df.reset_index().sort_values(['fish_id', 'frame_id'])
    assert df.index.is_unique
    current_fish = None
    next_available_id = df['fish_id'].max() + 1
    new_id = None
    for i, row in df.iterrows():
        if current_fish != row['fish_id']:
            current_fish = row['fish_id']
            new_id = None
        else:
            if row['frame_id'] - previous_frame_id - 1 > max_missing_frames:
                new_id = next_available_id
                next_available_id += 1
        if new_id is not None:
            df.loc[i, 'fish_id'] = new_id
        previous_frame_id = row['frame_id']

    return df.sort_values(['frame_id', 'fish_id']).set_index(['frame_id', 'fish_id'])


def get_fish_summary(df, fps=None, min_frames=0, max_missing_frames=10):
    print('max_missing_frames', max_missing_frames)
    if max_missing_frames is not None:
        df = assign_fish_new_ids(df, max_missing_frames=max_missing_frames)

    records = []
    for fish_id in df.index.get_level_values('fish_id').unique():
        fish_df = df.loc[df.index.get_level_values('fish_id') == fish_id]
        if len(fish_df) > min_frames:
            mouth_seq, mouth_counts, missing_counts = [], [], []
            previous_frame_id = None
            for (frame_id, fish_id), frame in fish_df.iterrows():
                if previous_frame_id is None:
                    first_frame_id = frame_id
                    mouth_seq.append(frame['label'])
                    previous_change = frame_id
                if mouth_seq[-1] != frame['label']:
                    mouth_counts.append(frame_id - previous_change)
                    mouth_seq.append(frame['label'])
                    previous_change = frame_id
                if previous_frame_id is not None and frame_id - previous_frame_id - 1 != 0:
                    missing_counts.append(frame_id - previous_frame_id - 1)
                previous_frame_id = frame_id
            if len(mouth_seq) != 0:
                mouth_counts.append(frame_id - previous_change + 1)
            assert len(mouth_seq) == len(mouth_counts)
            records.append([fish_id, frame_id - first_frame_id + 1, (fish_df['label'] == 'open').sum(),
                            (fish_df['label'] == 'closed').sum(), mouth_seq, mouth_counts, missing_counts])
    df = pd.DataFrame.from_records(
        records, columns=['fish_id', 'track_len', 'n_open', 'n_closed', 'mouth_seq', 'mouth_counts', 'missing_counts']
    ).set_index('fish_id')
    if fps is not None:
        df['track_len'] /= fps
        df['n_open'] /= fps
        df['n_closed'] /= fps
        df['mouth_counts'] = df['mouth_counts'].apply(lambda x: [c / fps for c in x])
        df['missing_counts'] = df['missing_counts'].apply(lambda x: [c / fps for c in x])
    return df


def get_resize_short_edge_shape(h: int, w: int, short_size: int, long_size: int) -> Tuple[int, int]:
    """
    Copyright (c) 2020 Meta Research
    Apache-2.0 license
    https://github.com/facebookresearch/detectron2

    Compute the output size given input size and target short edge length.
    """
    size = short_size * 1.0
    scale = size / min(h, w)
    if h < w:
        new_h, new_w = size, scale * w
    else:
        new_h, new_w = scale * h, size
    if long_size is not None and max(new_h, new_w) > long_size:
        scale = long_size * 1.0 / max(new_h, new_w)
        new_h = new_h * scale
        new_w = new_w * scale
    new_w = int(new_w + 0.5)
    new_h = int(new_h + 0.5)
    return new_h, new_w


def get_resize_long_edge_shape(h: int, w: int, long_size: int, short_size: int) -> Tuple[int, int]:
    assert short_size is None, 'Not implemented with short_size'
    size = long_size * 1.0
    scale = size / max(h, w)
    if h > w:
        new_h, new_w = size, math.ceil(scale * w)
    else:
        new_h, new_w = math.ceil(scale * h), size
    return int(new_h), int(new_w)


def resize_image(img, new_h, new_w, interp, return_array=True):
    """
    Copyright (c) 2020 Meta Research
    Apache-2.0 license
    https://github.com/facebookresearch/detectron2
    """
    assert img.dtype == np.uint8
    assert len(img.shape) <= 4
    monochrome = len(img.shape) > 2 and img.shape[2] == 1
    pil_image = Image.fromarray(img) if not monochrome else Image.fromarray(img[:, :, 0], mode="L")
    pil_image = pil_image.resize((new_w, new_h), interp)
    if return_array:
        img = np.asarray(pil_image)
        if monochrome:
            img = np.expand_dims(img, -1)
        return img
    else:
        return pil_image


def resize_bbox(bbox, old_img_h, old_img_w, new_img_h, new_img_w, bbox_format='XYWH'):
    """
    Copyright (c) 2020 Meta Research
    Apache-2.0 license
    https://github.com/facebookresearch/detectron2
    """
    assert bbox_format.lower() in ['xywh', 'xyxy']
    original_type = type(bbox)
    single_box = isinstance(bbox[0], (float, int))
    assert single_box, bbox
    # bbox is 1d (per-instance bounding box)
    if bbox_format.lower() == 'xywh':
        bbox = convert_boxes(bbox, from_format='XYWH', to_format='XYXY')
    # clip transformed bbox to image size
    bbox = apply_box(np.array([bbox]), old_img_h, old_img_w, new_img_h, new_img_w)[0].clip(min=0)
    bbox = np.minimum(bbox, [new_img_w, new_img_h, new_img_w, new_img_h])
    if bbox_format.lower() == 'xywh':
        bbox = convert_boxes(bbox, from_format='XYXY', to_format='XYWH')
    return original_type(bbox.flatten().tolist())


def apply_box(box: np.ndarray, old_img_h, old_img_w, new_img_h, new_img_w) -> np.ndarray:
    """
    Copyright (c) 2020 Meta Research
    Apache-2.0 license
    https://github.com/facebookresearch/detectron2

    Apply the transform on an axis-aligned box. By default will transform
    the corner points and use their minimum/maximum to create a new
    axis-aligned box. Note that this default may change the size of your
    box, e.g. after rotations.

    Args:
        box (ndarray): Nx4 floating point array of XYXY format in absolute
            coordinates.
    Returns:
        ndarray: box after apply the transformation.

    Note:
        The coordinates are not pixel indices. Coordinates inside an image of
        shape (H, W) are in range [0, W] or [0, H].

        This function does not clip boxes to force them inside the image.
        It is up to the application that uses the boxes to decide.
    """

    def apply_coords(coords):
        coords[:, 0] = coords[:, 0] * (new_img_w * 1.0 / old_img_w)
        coords[:, 1] = coords[:, 1] * (new_img_h * 1.0 / old_img_h)
        return coords

    # Indexes of converting (x0, y0, x1, y1) box into 4 coordinates of
    # ([x0, y0], [x1, y0], [x0, y1], [x1, y1]).
    idxs = np.array([(0, 1), (2, 1), (0, 3), (2, 3)]).flatten()
    coords = np.asarray(box).reshape(-1, 4)[:, idxs].reshape(-1, 2)
    coords = apply_coords(coords).reshape((-1, 4, 2))
    minxy = coords.min(axis=1)
    maxxy = coords.max(axis=1)
    trans_boxes = np.concatenate((minxy, maxxy), axis=1)
    return trans_boxes


def resize_dataset(dataset, img_dir, new_img_dir, short_size, long_size, resize_shape_op, do_not_enlarge=False,
                   interp=Image.BILINEAR):
    annotations = get_annotations_dict(dataset)
    for img in dataset['images']:
        image = read_image(os.path.join(img_dir, img['file_name']))
        old_img_h, old_img_w = image.shape[:2]
        new_h, new_w = resize_shape_op(h=image.shape[0], w=image.shape[1], short_size=short_size, long_size=long_size)
        if not do_not_enlarge or (new_h < old_img_h and new_w < old_img_w):
            print('Resize', img['file_name'], (old_img_w, old_img_h), '-->', (new_w, new_h))
            image = resize_image(img=image, new_h=new_h, new_w=new_w, interp=interp)
            new_img_h, new_img_w = image.shape[:2]
            assert short_size is None or min(new_img_h, new_img_w) <= short_size
            assert long_size is None or max(new_img_h, new_img_w) <= long_size
            write_image(image=image, image_fn=os.path.join(new_img_dir, img['file_name']))
            img['height'], img['width'] = image.shape[:2]
            for ann in annotations[img['id']]:
                ann['bbox'] = resize_bbox(
                    ann['bbox'], old_img_h=old_img_h, old_img_w=old_img_w,
                    new_img_h=new_img_h, new_img_w=new_img_w, bbox_format='XYWH'
                )
        else:
            print('No change', img['file_name'], (old_img_w, old_img_h))
    return dataset


def area(box_xyxy):
    return (box_xyxy[:, 2] - box_xyxy[:, 0]) * (box_xyxy[:, 3] - box_xyxy[:, 1])


def pairwise_intersection(boxes1, boxes2):
    import torch
    if not isinstance(boxes1, torch.Tensor):
        boxes1 = torch.tensor(boxes1)
    if not isinstance(boxes2, torch.Tensor):
        boxes2 = torch.tensor(boxes2)
    width_height = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    width_height = width_height.clamp_(min=0)  # [N,M,2]
    intersection = width_height.prod(dim=2)  # [N,M]
    return intersection.numpy()


def pairwise_ioa(boxes1, boxes2):
    """
    Similar to :func:`pariwise_iou` but compute the IoA (intersection over boxes2 area).

    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: IoA, sized [N,M].
    """
    area2 = area(boxes2)  # [M]
    inter = pairwise_intersection(boxes1, boxes2)

    # handle empty boxes
    ioa = np.where(inter > 0, inter / area2, 0)
    return ioa


def pairwise_iou(boxes1, boxes2):
    """
    Given two lists of boxes of size N and M, compute the IoU
    (intersection over union) between **all** N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: IoU, sized [N,M].
    """
    area1 = area(boxes1)  # [N]
    area2 = area(boxes2)  # [M]
    inter = pairwise_intersection(boxes1, boxes2)

    # handle empty boxes
    iou = np.where(inter > 0, inter / (area1[:, None] + area2 - inter), 0)
    return iou
