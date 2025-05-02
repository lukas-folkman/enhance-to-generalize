import os
import numpy as np
from PIL import Image
import skimage
import time
import argparse
import cv2
import torch
import cv_tools as utils


def get_cv2_kernel_size(sigma, dtype=float):
    return round(sigma * (3 if dtype == np.uint8 else 4) * 2 + 1)


NVIDIA = True
MAX_KERNEL = 169
MAX_SIGMA = 21
assert get_cv2_kernel_size(MAX_SIGMA) == MAX_KERNEL


if NVIDIA:
    from nvidia.dali import pipeline_def
    import nvidia.dali.fn as fn
    from nvidia.dali.math import log10, clamp
    from nvidia.dali.types import DALIDataType, FLOAT, UINT8
    from nvidia.dali.plugin.pytorch import feed_ndarray

    @pipeline_def()
    def msrcr_gpu_pipeline(resize_width, resize_height, sigmas, windows, alpha, beta, G, b, restore_color=True, device='gpu'):

        def downsizing_gaussian_blur(images, sigma, window_size):
            if window_size > MAX_KERNEL:
                return fn.resize(
                    fn.gaussian_blur(
                        fn.resize(images, resize_x=resize_width * (MAX_KERNEL / window_size), dtype=FLOAT),
                        sigma=MAX_SIGMA, window_size=MAX_KERNEL
                    ),
                    resize_x=resize_width, resize_y=resize_height, dtype=FLOAT
                )
            else:
                return fn.gaussian_blur(images, sigma=sigma, window_size=window_size)

        assert len(sigmas) in [1, 2, 3]
        assert len(sigmas) == len(windows)

        images = fn.external_source(name="images", device=device, dtype=UINT8, batch=False)
        images = fn.cast(images, dtype=FLOAT)
        # files, idx = fn.readers.file(files=file_names, file_root=file_root)
        # images = fn.decoders.image(files, device=device)
        images = images / 255.0 + 1

        images = fn.resize(images, resize_x=resize_width, resize_y=resize_height, dtype=FLOAT)
        if len(sigmas) == 3:
            out_img = ((log10(images) - log10(downsizing_gaussian_blur(images, sigma=sigmas[0], window_size=windows[0]))) + (
                        log10(images) - log10(downsizing_gaussian_blur(images, sigma=sigmas[1], window_size=windows[1]))) + (
                        log10(images) - log10(downsizing_gaussian_blur(images, sigma=sigmas[2], window_size=windows[2])))) / 3
        elif len(sigmas) == 2:
            out_img = ((log10(images) - log10(downsizing_gaussian_blur(images, sigma=sigmas[0], window_size=windows[0]))) + (
                        log10(images) - log10(downsizing_gaussian_blur(images, sigma=sigmas[1], window_size=windows[1])))) / 2
        else:
            out_img = log10(images) - log10(downsizing_gaussian_blur(images, sigma=sigmas[0], window_size=windows[0]))

        if restore_color:
            color_rest = beta * (log10(alpha * images) - log10(fn.reductions.sum(images, axes=2, keep_dims=True)))
            out_img = G * (out_img * color_rest + b)

        min_per_channel = fn.reductions.min(out_img, axes=[0, 1])
        max_per_channel = fn.reductions.max(out_img, axes=[0, 1])
        out_img = (out_img - min_per_channel) / (max_per_channel - min_per_channel)

        out_img = fn.cast(clamp(out_img * 255, lo=0, hi=255), dtype=DALIDataType.UINT8)
        return out_img  #, idx
else:
    def msrcr_gpu_pipeline(resize_width, resize_height, sigmas, windows, alpha, beta, G, b, restore_color=True,
                           batch_size=1, prefetch_queue_depth=None, num_threads=None, device_id=None, device='cpu'):
        raise ValueError('Not implemented')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', required=True)
    parser.add_argument('--out_dir', required=True)

    parser.add_argument('--ext', default='jpg')
    parser.add_argument('--flavour', choices=['MSRCR', 'MSR', 'SSR'], default='MSRCR')
    parser.add_argument('--framework', choices=['nvidia', 'openCV'], default='nvidia')
    parser.add_argument('--device_id', type=int, default=0)

    parser.add_argument('--resize_width', type=int, required=True)
    parser.add_argument('--resize_height', type=int, required=True)
    parser.add_argument('--sigmas', nargs='+', type=int)
    parser.add_argument('--windows', nargs='+', type=int)
    parser.add_argument('--alpha', default=125)
    parser.add_argument('--beta', default=46)
    parser.add_argument('--G', default=5)
    parser.add_argument('--b', default=25)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    file_names = read_files_from_dir(args.img_dir, filter_func=lambda x: x.endswith(f'.{args.ext}'), basename_only=True)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.sigmas is None:
        # do this based on size
        args.sigmas = np.asarray([15, 80, 250], dtype=float)
        args.sigmas = args.sigmas.tolist()
        print('sigmas:', args.sigmas)

    if args.windows is None:
        args.windows = [get_cv2_kernel_size(sigma=s) for s in args.sigmas]
        print('windows:', args.windows)

    assert len(args.sigmas) == len(args.windows)

    for filename in file_names:
        img = cv2.imread(os.path.join(args.img_dir, filename))
        if img.shape[1] != args.resize_width or img.shape[0] != args.resize_height:
            print(f"This will resize {(img.shape[1], img.shape[0])} to {(args.resize_width, args.resize_height)}.")

    if args.framework == 'nvidia':
        msrcr(file_names=file_names, image_dir=args.img_dir, out_dir=args.out_dir,
              resize_width=args.resize_width, resize_height=args.resize_height,
              flavour=args.flavour, sigmas=args.sigmas, windows=args.windows,
              alpha=args.alpha, beta=args.beta, G=args.G, b=args.b, device_id=args.device_id,
              device='gpu', warmup=args.warmup, verbose=args.verbose)
    else:
        print(args.flavour, 'sigmas:', args.sigmas, 'windows:', args.windows)
        delta_ts = []
        for filename in file_names:
            if not os.path.exists(os.path.join(args.out_dir, filename)):
                img = utils.read_image(os.path.join(args.img_dir, filename), format=utils.RGB_FORMAT)
                img = skimage.img_as_float64(img) + 1

                t0 = start_timing()
                img_out = np.zeros_like(img)
                log10_img = np.log10(img)
                for s, w in zip(args.sigmas, args.windows):
                    if w > MAX_KERNEL:
                        _img = cv2.resize(img, (int(round(args.resize_width * (MAX_KERNEL / w))), int(round(args.resize_height * (MAX_KERNEL / w)))))
                        _blur = cv2.GaussianBlur(_img, (MAX_KERNEL, MAX_KERNEL), MAX_SIGMA)
                        _blur = cv2.resize(_blur, (args.resize_width, args.resize_height))
                    else:
                        _blur = cv2.GaussianBlur(img, (MAX_KERNEL, MAX_KERNEL), MAX_SIGMA)
                    img_out += log10_img - np.log10(_blur)
                img_out = img_out / len(args.sigmas)

                if args.flavour == 'MSRCR':
                    color_rest = args.beta * (np.log10(args.alpha * img) - np.log10(np.sum(img, axis=2, keepdims=True)))
                    img_out = args.G * (img_out * color_rest + args.b)

                for i in range(img_out.shape[2]):
                    img_out[:, :, i] = (img_out[:, :, i] - np.min(img_out[:, :, i])) / (
                                np.max(img_out[:, :, i]) - np.min(img_out[:, :, i])) * 255
                img_out = np.uint8(np.minimum(np.maximum(img_out, 0), 255))
                delta_ts.append(elapsed_time(t0))
                if args.verbose:
                    print(img.shape, delta_ts[-1] * 1000, 'ms')
                print(img_out.shape, elapsed_time(t0) * 1000, 'ms')
                utils.write_image(img_out, os.path.join(args.out_dir, filename), flip_channels=True)

        if len(delta_ts) > args.warmup:
            print(f'Average processing time (after warmup): {np.mean(delta_ts[args.warmup:]) * 1000:.4f} ms (N = {len(delta_ts) - args.warmup})')


def msrcr(file_names, image_dir, out_dir, resize_width, resize_height, sigmas=None, windows=None,
          alpha=125.0, beta=46.0, G=5.0, b=25.0, flavour='MSRCR',
          num_threads=4, device_id=0, device='gpu', warmup=10, verbose=False):

    """
    https://github.com/adiMallya/retinex
    https://doi.org/10.1109/83.597272

    MSRCR (Multi-scale retinex with color restoration)

    Parameters :

    img : input image
    sigmas : list of all standard deviations in the X and Y directions, for Gaussian filter
    alpha : controls the strength of the nonlinearity
    beta : gain constant
    G : final gain
    b : offset
    """

    assert flavour in ['MSRCR', 'MSR', 'SSR']
    assert len(sigmas) in [1, 2, 3]
    assert len(sigmas) == len(windows)
    print(flavour, 'sigmas:', sigmas, 'windows:', windows)
    assert flavour != 'SSR' or len(sigmas) == 1

    BATCH_SIZE = 10
    t0 = start_timing()
    pipe_gpu = msrcr_gpu_pipeline(resize_width=resize_width, resize_height=resize_height,
                                  sigmas=sigmas, windows=windows,
                                  alpha=alpha, beta=beta, G=G, b=b, restore_color=flavour == 'MSRCR',
                                  batch_size=BATCH_SIZE, prefetch_queue_depth=1,
                                  num_threads=num_threads, device_id=device_id, device=device)
    pipe_gpu.build()
    print('Building pipeline:', elapsed_time(t0) * 1000, 'ms')

    AS_PYTORCH = False

    delta_ts = []
    N = len(file_names)
    N = 1000 # 4500

    for i in range(0, N, BATCH_SIZE):
        batch_fns = file_names[i:i+BATCH_SIZE]
        imgs = [utils.read_image(os.path.join(image_dir, filename), format=utils.RGB_FORMAT) for filename in batch_fns]
        t0 = start_timing()
        pipe_gpu.feed_input("images", np.array(imgs, dtype=np.uint8))
        pipeline_out = pipe_gpu.run()[0]

        if AS_PYTORCH:
            # Convert to PyTorch tensor
            pipeline_out = pipeline_out.as_tensor()
            imgs_out = torch.empty(pipeline_out.shape(), dtype=torch.uint8,
                                  device=torch.device(f'cuda:{device_id}'))
            feed_ndarray(pipeline_out, imgs_out)  # COPY !!!
            delta_ts.append(elapsed_time(t0) / BATCH_SIZE)
        else:
            imgs_out = pipeline_out.as_cpu().as_array()
            delta_ts.append(elapsed_time(t0) / BATCH_SIZE)
            for fn, img in zip(batch_fns, imgs_out):
                Image.fromarray(img).save(os.path.join(out_dir, fn))

        if verbose:
            print(imgs_out.shape, delta_ts[-1] * 1000 * BATCH_SIZE, 'ms')

    if len(delta_ts) > warmup:
        print(f'Average processing time (after warmup): {np.mean(delta_ts[warmup:]) * 1000:.4f} ms (N = {(len(delta_ts) - warmup) * BATCH_SIZE})')


def start_timing():
    return time.perf_counter()


def elapsed_time(start):
    return time.perf_counter() - start


def read_files_from_dir(dir_name, filter_func=None, basename_only=False):
    return [
        (fn if basename_only else os.path.join(dir_name, fn)) for fn in sorted(os.listdir(dir_name))
        if os.path.isfile(os.path.join(dir_name, fn)) and
           (filter_func(os.path.join(dir_name, fn)) if filter_func is not None else True)
    ]


if __name__ == '__main__':
    main()
