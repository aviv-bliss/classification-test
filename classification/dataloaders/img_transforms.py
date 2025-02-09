import numbers
import random
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from typing import List, Optional, Union
import PIL
import numpy as np
import cv2
from utils.auxiliary import noise_generator, resize_and_fill
try:
    import accimage
except ImportError:
    accimage = None

import torch


class Compose(object):
    """
        Composes several transforms together.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args


class ToTensor(object):
    """Convert a batch of ``PIL.Images`` or ``numpy.ndarrays`` to tensor.

    Converts a batch of PIL.Images or numpy.ndarrays (batch_size x H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (batch_size x C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, pic):
        """
        Args:
            batch of pics (PIL.Images or numpy.ndarrays): Images of shape (batch_size, H, W(, C))
            to be converted to tensor.
        Returns:
            Tensor: Converted batch of images of shape (batch_size, C, H, W).
        """
        if isinstance(pic, np.ndarray):
            # handle numpy array
            if len(pic.shape) == 3:
                pic = pic.reshape(pic.shape[0], pic.shape[1], pic.shape[2], -1)

            img = torch.from_numpy(pic.transpose((0, 3, 1, 2)))
            img = img.float().div(255)
            img = (img-0.5)/0.5

            return img

        if accimage is not None and isinstance(pic, accimage.Image):
            nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return torch.from_numpy(nppic)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img


class RandFlip(object):
    """
        Flip the given PIL.Image horizontally or vertically randomly
    """
    def __init__(self, hflip, vflip):
        self.hflip = hflip
        self.vflip = vflip

    def __call__(self, img1, img2=None, *args):
        rand_num1 = np.random.rand()
        if self.hflip and rand_num1 > 0.5:
            img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
            # print('horizontally flipped')
            if img2 is not None:
                img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)

        rand_num2 = np.random.rand()
        if self.vflip and rand_num2 > 0.5:
            img1 = img1.transpose(Image.FLIP_TOP_BOTTOM)
            # print('vertically flipped')
            if img2 is not None:
                img2 = img2.transpose(Image.FLIP_TOP_BOTTOM)

        results = [img1]
        if img2 is not None:
            results.append(img2)
        results.extend(args)
        return results


class RandHFlip(object):
    """
        Randomly horizontally flips the given np array image with a probability of 0.5
    """
    def __init__(self, hflip):
        self.hflip = hflip

    def __call__(self, img1, img2=None, *args):
        rand_num = np.random.rand()
        if self.hflip and rand_num > 0.5:
            img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
            # print('horizontally flipped')
            if img2 is not None:
                img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)

        results = [img1]
        if img2 is not None:
            results.append(img2)
        results.extend(args)
        return results


class Rotate(object):
    """
    Rotate the given PIL.Image by a random angle within the range [-max_angle, max_angle].
    """
    def __init__(self, max_angle):
        """
        Initialize the rotation transformation with a maximum angle.
        :param max_angle: The maximum absolute angle for rotation.
        """
        self.max_angle = max_angle

    def __call__(self, img1, img2=None, *args):
        """
        Apply the rotation transformation to the given image.
        :param img1: The PIL.Image to rotate.
        :param img2: An optional second image (not used in this implementation).
        :param *args: Additional arguments to include in the results list.
        :return: A list where the first element is the rotated image, followed by any additional arguments.
        """
        if self.max_angle > 0:
            # Random angle between -max_angle and +max_angle
            ang = np.random.uniform(-self.max_angle, self.max_angle)
            img1 = img1.rotate(ang)
            # print(f'Rotated by {ang:.2f} degrees')
            if img2 is not None:
                img2 = img2.rotate(ang)

        results = [img1]
        if img2 is not None:
            results.append(img2)
        results.extend(args)
        return results


class RandCrop(object):
    """
        Crops the given PIL.Image at a random location to have a region of
    crop_max in the range [crop_min_h, 1] and [crop_min_w, 1]
    (crop_min_h and crop_min_w should be in the range (0,1]).
    crop_max can be a tuple (crop_min_h, crop_min_w) or an integer, in which case
    the target will be in the range [crop_min_h, 1] and [crop_min_h, 1]
    """

    def __init__(self, crop_max):
        if isinstance(crop_max, numbers.Number):
            self.crop_max = (int(crop_max), int(crop_max))
        else:
            self.crop_max = crop_max

    def __call__(self, img1, img2=None, *args):
        crop_min_h, crop_min_w = self.crop_max
        assert crop_min_h > 0 and crop_min_w > 0 and crop_min_h <= 1.0 and crop_min_w <= 1.0
        if crop_min_h == 1.0 and crop_min_w == 1.0:
            return [img1, img2, args]

        w, h = img1.size
        if img2 is not None:
            w2, h2 = img2.size
            if w != w2:
                if w > w2:
                    img1 = img1.resize((w2, h))
                else:
                    img2 = img2.resize((w, h2))
            if h != h2:
                if h > h2:
                    img1 = img1.resize((w, h2))
                else:
                    img2 = img2.resize((w, h))

        rand_w = random.randint(int(crop_min_w*w), w)
        rand_h = random.randint(int(crop_min_h*h), h)
        x1 = random.randint(0, w - rand_w)
        y1 = random.randint(0, h - rand_h)
        results = [img1.crop((x1, y1, x1 + rand_w, y1 + rand_h))]
        # print('image croped: ({}, {} ,{}, {})'.format(x1, y1, x1 + rand_w, y1 + rand_h))
        if img2 is not None:
            results.append(img2.crop((x1, y1, x1 + rand_w, y1 + rand_h)))
        results.extend(args)
        return results


class EnhanceColor(object):
    """
    Enhance color (saturation) in the given PIL.Image.
    Values of color_factor:
        0.0: Completely desaturated image (grayscale).
        Between 0.0 and 1.0: Reduced color intensity relative to the original.
        Greater than 1.0: Increased color saturation, making colors more vivid.
    This transform is callable and returns a list of images and any extra arguments.
    - If img2 is provided, the same color enhancement is applied to both images.
    """
    def __init__(self, color):
        """
        Args:
            color (float): A value in [0, 1] that defines how much we vary the
                           color_factor between [1 - color, 1 + color].
                           If color >= 1, no color enhancement is applied.
        """
        self.color = color

    def __call__(self, img1, img2=None, *args):
        """
        Args:
            img1 (PIL.Image): The first image to enhance.
            img2 (PIL.Image, optional): The second image to enhance, if any.
            *args: Additional items that should be passed through unchanged.
        Returns:
            list: A list of images (after optional enhancement) plus any additional args.
                  If img2 is not None, you'll get two images; otherwise, one image.
        """
        if self.color < 1.0:
            color_factor = np.random.uniform(1.0 - self.color, 1.0 + self.color)
            # print(f'color_factor: {color_factor}')

            enhancer1 = ImageEnhance.Color(img1)
            img1 = enhancer1.enhance(color_factor)
            if img2 is not None:
                enhancer2 = ImageEnhance.Color(img2)
                img2 = enhancer2.enhance(color_factor)

        results = [img1]
        if img2 is not None:
            results.append(img2)
        results.extend(args)
        return results


class EnhanceContrast(object):
    """
    Enhance contrast in the given PIL.Image.
    Values of contrast_factor:
        0.0: Converts the image into a flat, uniform gray image (lowest contrast).
        Between 0.0 and 1.0: Decreases contrast relative to the original.
        Greater than 1.0: Increases contrast, making dark areas darker and bright areas brighter.
    This transform is callable and returns a list of images and any extra arguments.
    - If img2 is provided, the same contrast enhancement is applied to both images.
    """
    def __init__(self, contrast):
        """
        Args:
            contrast (float): A value in [0, 1] that defines how much we vary the
                              contrast_factor between [1 - contrast, 1 + contrast].
                              If contrast >= 1, no contrast enhancement is applied.
        """
        self.contrast = contrast

    def __call__(self, img1, img2=None, *args):
        """
        Args:
            img1 (PIL.Image): The first image to enhance.
            img2 (PIL.Image, optional): The second image to enhance, if any.
            *args: Additional items that should be passed through unchanged.
        Returns:
            list: A list of images (after optional enhancement) plus any additional args.
                  If img2 is not None, you'll get two images; otherwise, one image.
        """
        if self.contrast < 1.0:
            contrast_factor = np.random.uniform(1.0 - self.contrast, 1.0 + self.contrast)
            # print('contrast_factor', contrast_factor)

            enhancer1 = ImageEnhance.Contrast(img1)
            img1 = enhancer1.enhance(contrast_factor)
            if img2 is not None:
                enhancer2 = ImageEnhance.Contrast(img2)
                img2 = enhancer2.enhance(contrast_factor)

        results = [img1]
        if img2 is not None:
            results.append(img2)
        results.extend(args)
        return results


class EnhanceBrightness(object):
    """
    Enhance brightness in the given PIL.Image.
    Values of brightness_factor:
        0.0: Turns the image completely black.
        Between 0.0 and 1.0: Darkens the image relative to the original.
        Greater than 1.0: Brightens the image, making it lighter.
    This transform is callable and returns a list of images and any extra arguments.
    - If img2 is provided, the same brightness enhancement is applied to both images.
    """
    def __init__(self, brightness):
        """
        Args:
            brightness (float): A value in [0, 1] for random brightness_factor in the range [1-brightness, 1+brightness].
                                If brightness >= 1, no enhancement is applied.
        """
        self.brightness = brightness

    def __call__(self, img1, img2=None, *args):
        """
        Args:
            img1 (PIL.Image): The first image to enhance.
            img2 (PIL.Image, optional): The second image to enhance, if any.
            *args: Additional items that should be passed through unchanged.
        Returns:
            list: A list of images (after optional enhancement) plus any additional args.
                  If img2 is not None, you’ll get two images; otherwise, one image.
        """
        if self.brightness < 1.0:
            brightness_factor = np.random.uniform(1 - self.brightness, 1.0 + self.brightness)
            # print(f'brightness_factor: {brightness_factor}')

            enhancer1 = ImageEnhance.Brightness(img1)
            img1 = enhancer1.enhance(brightness_factor)

            if img2 is not None:
                enhancer2 = ImageEnhance.Brightness(img2)
                img2 = enhancer2.enhance(brightness_factor)

        results = [img1]
        if img2 is not None:
            results.append(img2)
        results.extend(args)
        return results


class EnhanceSharpness(object):
    """
    Enhance sharpness in the given PIL.Image.
    Values of sharpness_factor:
        0.0: Produces a heavily blurred image (minimum sharpness).
        Between 0.0 and 1.0: Progressively less sharp than the original (some blurring).
        Greater than 1.0: Increases sharpness, making edges and details more defined.
    This transform is callable and returns a list of images and any extra arguments.
    - If img2 is provided, the same sharpness enhancement is applied to both images.
    """
    def __init__(self, sharpness):
        """
        Args:
            sharpness (float): A value in [0, 1] that defines how much we vary the
                               sharpness_factor between [1 - sharpness, 1 + sharpness].
                               If sharpness >= 1, no sharpness enhancement is applied.
        """
        self.sharpness = sharpness

    def __call__(self, img1, img2=None, *args):
        """
        Args:
            img1 (PIL.Image): The first image to enhance.
            img2 (PIL.Image, optional): The second image to enhance, if any.
            *args: Additional items that should be passed through unchanged.
        Returns:
            list: A list of images (after optional enhancement) plus any additional args.
                  If img2 is not None, you'll get two images; otherwise, one image.
        """
        if self.sharpness < 1.0:
            sharpness_factor = np.random.uniform(1.0 - self.sharpness, 1.0 + self.sharpness)
            # print(f'sharpness_factor: {sharpness_factor}')

            enhancer1 = ImageEnhance.Sharpness(img1)
            img1 = enhancer1.enhance(sharpness_factor)

            if img2 is not None:
                enhancer2 = ImageEnhance.Sharpness(img2)
                img2 = enhancer2.enhance(sharpness_factor)

        results = [img1]
        if img2 is not None:
            results.append(img2)
        results.extend(args)
        return results


class RandNoise:
    """
    Add random noise to PIL Images.
    This class provides functionality to add different types of noise to images:
    - Gaussian noise
    - Salt & Pepper noise
    - Gaussian blur
    Attributes:
        noise (bool): Flag to enable/disable noise addition
        noise_probabilities (dict): Probability ranges for different noise types
    """
    def __init__(self, noise: bool = False,
                 noise_probabilities: Optional[dict] = None,
                 noise_params: Optional[dict] = None):
        """
        Initialize the RandNoise transform.
        Args:
            noise (bool): Whether to enable noise addition
            noise_probabilities (dict, optional): Custom probability ranges for noise types.
                Default probabilities are:
                - Gaussian noise: 0.3
                - Salt & Pepper noise: 0.3
                - Gaussian blur: 0.1
                - No noise: 0.3
            noise_params (dict, optional): Parameters for noise generation
                Default parameters are:
                - Gaussian noise: mean=0, var=0.1
                - Salt & Pepper noise: s_vs_p=0.5, amount=0.002
                - Gaussian blur: sigma=1
        """
        self.noise = noise
        self.noise_probabilities = noise_probabilities or {
            'gaussian': 0.3,
            'salt_pepper': 0.3,
            'blur': 0.1,
            'none': 0.3
        }
        self.noise_params = noise_params or {
            'gaussian': {'mean': 0, 'var': 0.1},
            'salt_pepper': {'s_vs_p': 0.5, 'amount': 0.002},
            'blur': {'sigma': 1}
        }
        total_prob = sum(self.noise_probabilities.values())
        if not np.isclose(total_prob, 1.0, atol=1e-10):
            raise ValueError(f"Noise probabilities must sum to 1.0, got {total_prob}")

    def __call__(self, img1: Image.Image,
                 img2: Optional[Image.Image] = None,
                 *args) -> List[Union[Image.Image]]:
        """
        Apply random noise transformation to the input image(s).
        Args:
            img1 (PIL.Image): Primary input image
            img2 (PIL.Image, optional): Secondary input image
            *args: Additional arguments to be passed through
        Returns:
            list: List containing the transformed image(s) and any additional arguments
        """
        if self.noise:
            rand_num = np.random.rand()

            if rand_num < self.noise_probabilities['gaussian']:
                img1_np = np.array(img1)
                img1_noisy = noise_generator(self.noise_params, 'gauss', img1_np)
                # print('added gauss noise')
                img1 = Image.fromarray(img1_noisy.astype(np.uint8))
                if img2 is not None:
                    img2_np = np.array(img2)
                    img2_noisy =noise_generator(self.noise_params, 'gauss', img2_np)
                    img2 = Image.fromarray(img2_noisy.astype(np.uint8))

            elif rand_num > self.noise_probabilities['gaussian'] and rand_num <= self.noise_probabilities['gaussian'] + self.noise_probabilities['salt_pepper']:
                img1_np = np.array(img1)
                img1_noisy = noise_generator(self.noise_params, 's&p', img1_np)
                # print('added s&p noise')
                img1 = Image.fromarray(img1_noisy.astype(np.uint8))
                if img2 is not None:
                    img2_np = np.array(img2)
                    img2_noisy = noise_generator(self.noise_params, 's&p', img2_np)
                    img2 = Image.fromarray(img2_noisy.astype(np.uint8))

            elif rand_num > self.noise_probabilities['gaussian'] + self.noise_probabilities['salt_pepper'] and rand_num <= self.noise_probabilities['gaussian'] + self.noise_probabilities['salt_pepper'] + self.noise_probabilities['blur']:
                img1 = img1.filter(ImageFilter.GaussianBlur(self.noise_params['blur']['sigma']))
                # print('added gaussian blur')
                if img2 is not None:
                    img2 = img2.filter(ImageFilter.GaussianBlur(self.noise_params['blur']['sigma']))

            # else:
                # print('no noise added')

        results = [img1]
        if img2 is not None:
            results.append(img2)
        results.extend(args)
        return results


class Resize(object):
    """
        Resizes the given PIL.Image to the given size.
    size can be a tuple (target_height, target_width) or an integer,
    in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img1, img2=None, *args):
        assert img2 is None or img1.size == img2.size

        w, h = img1.size
        resize_w, resize_h = self.size
        if w == resize_w and h == resize_h:
            return [img1, img2, args]

        results = [img1.resize(self.size, PIL.Image.ANTIALIAS)]
        if img2 is not None:
            results.append(img2.resize(self.size, PIL.Image.ANTIALIAS))
        results.extend(args)

        return results


class ResizeAndFillBlack(object):
    """
    Resizes the given PIL.Image to the given height.
    Calculates the relative width of the given height and fills the empty space black.
    size can be a tuple (target_height, target_width) or an integer,
    in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img1, img2=None, *args):
        if img1.size != img2.size:
            if img1.size[0] > img2.size[0] or img1.size[1] > img2.size[1]:
                img1 = img1.resize(img2.size, PIL.Image.ANTIALIAS)
            else:
                img2 = img2.resize(img1.size, PIL.Image.ANTIALIAS)
        assert img2 is None or img1.size == img2.size

        w, h = img1.size
        resize_w, resize_h = self.size
        if w == resize_w and h == resize_h:
            return [img1, img2, args]

        results = [resize_and_fill(img1, resize_w, resize_h)]
        # print('resized from ({},{}) to ({})'.format(w, h, self.size))
        if img2 is not None:
            results.append(resize_and_fill(img2, resize_w, resize_h))
        results.extend(args)

        return results


class Normalize(object):
    """
        Normalizes a PIL image or np.array with a given min and std.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        assert std > 0, 'std for image normalization is 0'

    def __call__(self, img1, img2=None, *args):
        img1 = np.array(img1).astype(np.float32)
        # img1 /= np.max(img1)                            # normalization of image from 0 to 1
        img1 = (img1 - self.mean)/self.std
        if img2 is not None:
            img2 = np.array(img2).astype(np.float32)
            img2 = (img2 - self.mean) / self.std
            results = [img1, img2]
        else:
            results = img1
        return results


class Pad(object):
    """
        Pads the given PIL.Image on all sides with the given "pad" value
    """

    def __init__(self, padding, fill=0):
        assert isinstance(padding, numbers.Number)
        assert isinstance(fill, numbers.Number) or isinstance(fill, str) or \
               isinstance(fill, tuple)
        self.padding = padding
        self.fill = fill

    def __call__(self, img1, img2=None, *args):
        if img2 is not None:
            img2 = ImageOps.expand(img2, border=self.padding, fill=255)
        if self.fill == -1:
            img1 = np.asarray(img1)
            img1 = cv2.copyMakeBorder(img1, self.padding, self.padding,
                                       self.padding, self.padding,
                                       cv2.BORDER_REFLECT_101)
            img1 = Image.fromarray(img1)
            return (img1, img2, args)
        else:
            return ImageOps.expand(img1, border=self.padding, fill=self.fill), img2


