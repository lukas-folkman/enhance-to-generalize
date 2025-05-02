import argparse
import numpy as np
import os
import time
import cv2
from transforms import IMG_TRANSFORMS
import cv_tools as utils


RESIZE_IMAGES_FOR_SPEED_COMPARISON = False
WARMUP = 10


def transform_img_dir(img_dir, img_transform, output_dir, stop_at=None):
    assert img_transform in utils.IMG_TRANSFORMS
    imgs = utils.read_files_from_dir(img_dir, filter_func=utils.is_image, basename_only=False)
    delta_ts = []
    for i, img_fn in enumerate(imgs):
        if stop_at is not None and i >= stop_at:
            break
        new_fn = os.path.join(output_dir, os.path.basename(img_fn))
        if not os.path.exists(new_fn):
            img = utils.read_image(img_fn, format=utils.BGR_FORMAT)
            if RESIZE_IMAGES_FOR_SPEED_COMPARISON:
                img = cv2.resize(img, (1333, int(np.floor((img.shape[0] / img.shape[1]) * 1333))), interpolation=cv2.INTER_CUBIC)
            s = utils.start_timing()
            img = utils.IMG_TRANSFORMS[img_transform](img)
            delta_ts.append(utils.elapsed_time(s))
            utils.write_image(img, new_fn)
    if len(delta_ts) > WARMUP:
        print(f'{img_transform} average processing time (after WARMUP): {np.mean(delta_ts[WARMUP:]) * 1000:.4f} ms (N = {len(delta_ts) - WARMUP})')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', required=True)
    parser.add_argument('--img_transform', choices=list(IMG_TRANSFORMS.keys()), required=True)
    parser.add_argument('--output_dir')
    parser.add_argument('--stop_at', type=int)
    args = parser.parse_args()

    assert args.img_transform in utils.IMG_TRANSFORMS
    if args.output_dir is None:
        args.output_dir = f'{args.img_dir}_{args.img_transform}'
    os.makedirs(args.output_dir, exist_ok=True)
    transform_img_dir(img_dir=args.img_dir, img_transform=args.img_transform, output_dir=args.output_dir, stop_at=args.stop_at)


if __name__ == '__main__':
    main()
