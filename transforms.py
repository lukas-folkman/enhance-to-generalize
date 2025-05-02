import os
import numpy as np
import math
import cv2
import PIL
from PIL import ImageFilter, ImageEnhance, ImageOps
import skimage

from cv_tools import BGR_FORMAT, RGB_FORMAT


def flip_color_channels(image):
    return image[:, :, ::-1]


def enhance(img, which, factor, format=BGR_FORMAT):
    assert which in ['color', 'contrast', 'brightness', 'sharpness']
    assert format in [BGR_FORMAT, RGB_FORMAT]
    if format == BGR_FORMAT:
        img = flip_color_channels(img)

    if which == 'color':
        _Enhancer = ImageEnhance.Color
    elif which == 'contrast':
        _Enhancer = ImageEnhance.Contrast
    elif which == 'brightness':
        _Enhancer = ImageEnhance.Brightness
    elif which == 'sharpness':
        _Enhancer = ImageEnhance.Sharpness
    img = np.array(_Enhancer(PIL.Image.fromarray(img)).enhance(factor=factor))

    if format == BGR_FORMAT:
        img = flip_color_channels(img)

    return img


def unsharp_mask(img, radius=2.0, amount=1.5, threshold=3, format=BGR_FORMAT):
    assert format in [BGR_FORMAT, RGB_FORMAT]
    if format == BGR_FORMAT:
        img = flip_color_channels(img)

    img = np.array(PIL.Image.fromarray(img).filter(
        ImageFilter.UnsharpMask(radius=radius, percent=int(100 * amount), threshold=threshold)))

    if format == BGR_FORMAT:
        img = flip_color_channels(img)

    return img


def adjust_gamma(img, gamma=1, gain=1, format=BGR_FORMAT):
    assert format in [BGR_FORMAT, RGB_FORMAT]
    if format == BGR_FORMAT:
        img = flip_color_channels(img)

    img = skimage.exposure.adjust_gamma(image=img, gamma=gamma, gain=gain)

    if format == BGR_FORMAT:
        img = flip_color_channels(img)
    return img


def adjust_log(img, gain=1, inv=False, format=BGR_FORMAT):
    assert format in [BGR_FORMAT, RGB_FORMAT]
    if format == BGR_FORMAT:
        img = flip_color_channels(img)

    img = skimage.exposure.adjust_log(image=img, gain=gain, inv=inv)

    if format == BGR_FORMAT:
        img = flip_color_channels(img)
    return img


def adjust_sigmoid(img, cutoff=0.5, gain=10, inv=False, format=BGR_FORMAT):
    assert format in [BGR_FORMAT, RGB_FORMAT]
    if format == BGR_FORMAT:
        img = flip_color_channels(img)

    img = skimage.exposure.adjust_sigmoid(image=img, cutoff=cutoff, gain=gain, inv=inv)

    if format == BGR_FORMAT:
        img = flip_color_channels(img)
    return img


def adaptive_histogram_equalization(img, clip_limit=1, tile_grid_size=(8, 8), format=BGR_FORMAT):
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE).
    https://doi.org/10.3390/rs11111381
    """
    assert format in [BGR_FORMAT, RGB_FORMAT]
    if format == RGB_FORMAT:
        img = flip_color_channels(img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    img[:, :, 0] = clahe.apply(img[:, :, 0])
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

    # img = skimage.exposure.equalize_adapthist(image=img, kernel_size=kernel_size, clip_limit=clip_limit, nbins=nbins)
    # img = (img * 255).clip(0, 255).astype(np.uint8)

    if format == RGB_FORMAT:
        img = flip_color_channels(img)
    return img


def autocontrast(img, cutoff=(0, 0), preserve_tone=False, format=BGR_FORMAT):
    assert format in [BGR_FORMAT, RGB_FORMAT]
    if format == BGR_FORMAT:
        img = flip_color_channels(img)

    img = np.array(ImageOps.autocontrast(PIL.Image.fromarray(img), cutoff=cutoff, preserve_tone=preserve_tone))

    if format == BGR_FORMAT:
        img = flip_color_channels(img)

    return img


def ace(img, ace_slope=10, ace_limit=1000, ace_samples=500, format=BGR_FORMAT):
    from colorcorrect.algorithm import automatic_color_equalization
    assert format in [BGR_FORMAT, RGB_FORMAT]
    # https://doi.org/10.1016/S0167-8655(02)00323-9
    if format == BGR_FORMAT:
        img = flip_color_channels(img)
    img = automatic_color_equalization(img, slope=ace_slope, limit=ace_limit, samples=ace_samples)
    if format == BGR_FORMAT:
        img = flip_color_channels(img)

    return img


def dark_channel_prior_dehazing(img, dark_channels='BGR', percent=0.001, shift_blue_to_white=[False, False], format=BGR_FORMAT):
    """
    https://github.com/He-Zhang/image_dehaze
    https://www.sciencedirect.com/science/article/abs/pii/S1047320314001874
    shift_blue_to_white: https://arxiv.org/abs/1807.04169
    """

    def DarkChannel(im, sz, dark_channels='BGR', shift_blue_to_white=False):
        assert len(dark_channels) in [3, 2]

        if shift_blue_to_white:
            im = np.array(im)
            im[:, :, 1] = 1 - im[:, :, 1]
            im[:, :, 2] = 1 - im[:, :, 2]

        bgr = {l: c for l, c in zip(['B', 'G', 'R'], cv2.split(im))}
        dc = cv2.min(cv2.min(bgr['R'], bgr['G']), bgr['B']) if len(dark_channels) == 3 \
            else cv2.min(bgr[dark_channels[0]], bgr[dark_channels[1]])
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
        dark = cv2.erode(dc, kernel)
        return dark

    def AtmLight(im, dark, percent=0.001, shift_blue_to_white=False):
        if not shift_blue_to_white:
            [h, w] = im.shape[:2]
            imsz = h * w
            numpx = int(max(math.floor(imsz * percent), 1))
            darkvec = dark.reshape(imsz);
            imvec = im.reshape(imsz, 3);

            indices = darkvec.argsort();
            indices = indices[imsz - numpx::]

            atmsum = np.zeros([1, 3])
            for ind in range(1, numpx):
                atmsum = atmsum + imvec[indices[ind]]

            A = atmsum / numpx;

        else:
            x = np.argmin(dark) // dark.shape[1]
            y = np.argmin(dark) % dark.shape[1]
            A = im[x, y, :].reshape((1, -1))

        return A

    def TransmissionEstimate(im, A, sz, dark_channels='BGR', shift_blue_to_white=False):
        omega = 0.95;
        im3 = np.empty(im.shape, im.dtype);

        for ind in range(0, 3):
            im3[:, :, ind] = im[:, :, ind] / A[0, ind]

        transmission = 1 - omega * DarkChannel(im3, sz, dark_channels=dark_channels,
                                               shift_blue_to_white=shift_blue_to_white);
        return transmission

    def Guidedfilter(im, p, r, eps):
        mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r));
        mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r));
        mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r));
        cov_Ip = mean_Ip - mean_I * mean_p;

        mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r));
        var_I = mean_II - mean_I * mean_I;

        a = cov_Ip / (var_I + eps);
        b = mean_p - a * mean_I;

        mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r));
        mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r));

        q = mean_a * im + mean_b;
        return q;

    def TransmissionRefine(im, et):
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY);
        gray = np.float64(gray) / 255;
        r = 60;
        eps = 0.0001;
        t = Guidedfilter(gray, et, r, eps);

        return t;

    def Recover(im, t, A, tx=0.1):
        res = np.empty(im.shape, im.dtype);
        t = cv2.max(t, tx);

        for ind in range(0, 3):
            res[:, :, ind] = (im[:, :, ind] - A[0, ind]) / t + A[0, ind]

        return res

    assert format in [BGR_FORMAT, RGB_FORMAT]
    if format == RGB_FORMAT:
        img = flip_color_channels(img)

    I = img.astype('float64') / 255
    dark = DarkChannel(I, 15, dark_channels=dark_channels, shift_blue_to_white=shift_blue_to_white[0])
    A = AtmLight(I, dark, percent=percent, shift_blue_to_white=shift_blue_to_white[0])
    te = TransmissionEstimate(I, A, 15, dark_channels='BGR', shift_blue_to_white=shift_blue_to_white[1])
    t = TransmissionRefine(img, te)
    img = (Recover(I, t, A, 0.1) * 255).clip(0, 255).astype(np.uint8)

    if format == RGB_FORMAT:
        img = flip_color_channels(img)

    return img


def grey_world_LAB(img, brightness_factor=1.0, format=BGR_FORMAT):
    """
    https://www.youtube.com/watch?v=Z0-iM37wseI
    https://github.com/bnsreenu/python_for_microscopists/tree/master/Tips_Tricks_45_white-balance_using_python
    https://github.com/bnsreenu/python_for_microscopists
    """
    assert format in [BGR_FORMAT, RGB_FORMAT]
    if format == RGB_FORMAT:
        img = flip_color_channels(img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    for c in [1, 2]:
        avg = np.average(img[:, :, c])
        img[:, :, c] = img[:, :, c] - ((avg - 128) * (img[:, :, 0] / 255) * brightness_factor)
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    img = img.clip(0, 255).astype(np.uint8)

    if format == RGB_FORMAT:
        img = flip_color_channels(img)
    return img


def MSRCR(img, sigmas=[15, 80, 250], alpha=125.0, beta=46.0, G=5.0, b=25.0, flavour='MSRCR',
          format=BGR_FORMAT, return_time=False):
    """
    '''
    https://github.com/adiMallya/retinex
    https://doi.org/10.1109/83.597272
    '''

    MSRCR (Multi-scale retinex with color restoration)

    Parameters :

    img : input image
    sigmas : list of all standard deviations in the X and Y directions, for Gaussian filter
    alpha : controls the strength of the nonlinearity
    beta : gain constant
    G : final gain
    b : offset
    """

    if return_time:
        import time
        def start_timing():
            return time.perf_counter()
        def elapsed_time(start):
            return time.perf_counter() - start

    def singleScale(img, sigma):
        """
        Single-scale Retinex

        Parameters :

        img : input image
        sigma : the standard deviation in the X and Y directions, for Gaussian filter
        """
        gb_img = cv2.GaussianBlur(img, (0, 0), sigma)
        ssr = np.log10(img) - np.log10(gb_img)
        return ssr

    def multiScale(img, sigmas: list):
        """
        Multi-scale Retinex

        Parameters :

        img : input image
        sigma : list of all standard deviations in the X and Y directions, for Gaussian filter
        """
        retinex = np.zeros_like(img)
        for s in sigmas:
            retinex += singleScale(img, s)
        msr = retinex / len(sigmas)
        return msr

    def crf(img, alpha, beta):
        """
        CRF (Color restoration function)

        Parameters :

        img : input image
        alpha : controls the strength of the nonlinearity
        beta : gain constant
        """
        img_sum = np.sum(img, axis=2, keepdims=True)

        color_rest = beta * (np.log10(alpha * img) - np.log10(img_sum))
        return color_rest


    if return_time:
        s = start_timing()

    if flavour in ['MSRCR', 'MSR']:
        if format == BGR_FORMAT:
            img_msr = flip_color_channels(img)
        img_msr = skimage.img_as_float64(img_msr) + 1
        img_msr = multiScale(img_msr, sigmas=sigmas)

    if format == BGR_FORMAT:
        img = flip_color_channels(img)
    img = skimage.img_as_float64(img) + 1

    if flavour == 'MSRCR':
        img = G * (img_msr * crf(img, alpha=alpha, beta=beta) + b)
    elif flavour == 'MSR':
        img = img_msr
    elif flavour == 'SSR':
        assert len(sigmas) == 1
        img = singleScale(img, sigma=sigmas[0])
    else:
        raise ValueError

    for i in range(img.shape[2]):
        img[:, :, i] = (img[:, :, i] - np.min(img[:, :, i])) / (np.max(img[:, :, i]) - np.min(img[:, :, i])) * 255

    if return_time:
        t = elapsed_time(s)

    img = np.uint8(np.minimum(np.maximum(img, 0), 255))

    if format == BGR_FORMAT:
        img = flip_color_channels(img)

    return (img, t) if return_time else img


IMG_TRANSFORMS = {

    # FOR ARCR, CAP, MLLE, FunIE_GAN, THE FOLLOWING IMPLEMENTATIONS WERE USED:
    # ARCR: https://github.com/26hzhang/optimizedimageenhance
    # CAP: https://github.com/jiamingmai/color-attenuation-prior-dehazing
    # MLLE: https://github.com/li-chongyi/mmle_code
    # FunIE_GAN: https://github.com/xahidbuffon/funie-gan

    'adjust_gamma_down': lambda x: adjust_gamma(x, gamma=0.7),
    'adjust_gamma_up': lambda x: adjust_gamma(x, gamma=1.3),
    'adjust_log': adjust_log,
    'adjust_sigmoid': adjust_sigmoid,
    'autocontrast': autocontrast,
    'CLAHE': adaptive_histogram_equalization,
    'DCP': dark_channel_prior_dehazing,
    'grey_world': grey_world_LAB,
    'ace': ace,
    'sharpen': lambda x: enhance(img=x, which='sharpness', factor=2),
    'unsharp_mask': unsharp_mask,
    'MSRCR_full_kernel': MSRCR,
    'MSR_full_kernel': lambda x: MSRCR(img=x, flavour='MSR'),
    'SSR_full_kernel': lambda x: MSRCR(img=x, flavour='SSR'),

}
