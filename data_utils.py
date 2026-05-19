from pathlib import Path

import PIL
import torch
import torchvision.transforms.functional as FT
from torchvision.transforms import Compose, CenterCrop, ToTensor, Normalize, Resize
from torchvision.transforms import InterpolationMode

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def collate_fn(batch):
    '''
    function which discard None images in a batch when using torch DataLoader
    Also handles PIL images by keeping them as lists instead of trying to collate them.
    :param batch: input_batch
    :return: output_batch = input_batch - None_values
    '''
    batch = list(filter(lambda x: x is not None, batch))
    
    # If batch is empty after filtering, return empty dict or list
    if len(batch) == 0:
        return {}
    
    # Check if batch contains dicts (typical case)
    if isinstance(batch[0], dict):
        # Get all keys from first element
        keys = batch[0].keys()
        collated = {}
        
        for key in keys:
            values = [d[key] for d in batch]
            # Check if values are PIL images or specific keys that shouldn't be collated
            if (len(values) > 0 and isinstance(values[0], PIL.Image.Image)) or key == 'semantic_aspects':
                # Keep PIL images and textual list fields as list
                collated[key] = values
            else:
                # Use default collate for other types
                try:
                    collated[key] = torch.utils.data.dataloader.default_collate(values)
                except Exception:
                    # If default_collate fails, keep as list
                    collated[key] = values
        return collated
    else:
        # For non-dict batches (like tuples), use default collate
        return torch.utils.data.dataloader.default_collate(batch)


class TargetPad:
    """
    If an image aspect ratio is above a target ratio, pad the image to match such target ratio.
    For more details see Baldrati et al. 'Effective conditioned and composed image retrieval combining clip-based features.' Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2022).
    """

    def __init__(self, target_ratio: float, size: int):
        """
        :param target_ratio: target ratio
        :param size: preprocessing output dimension
        """
        self.size = size
        self.target_ratio = target_ratio

    def __call__(self, image: PIL.Image.Image) -> PIL.Image.Image:
        w, h = image.size
        actual_ratio = max(w, h) / min(w, h)
        if actual_ratio < self.target_ratio:  # check if the ratio is above or below the target ratio
            return image
        scaled_max_wh = max(w, h) / self.target_ratio  # rescale the pad to match the target ratio
        hp = max(int((scaled_max_wh - w) / 2), 0)
        vp = max(int((scaled_max_wh - h) / 2), 0)
        padding = [hp, vp, hp, vp]
        return FT.pad(image, padding, 0, 'constant')


def targetpad_transform(target_ratio: float, dim: int) -> torch.Tensor:
    """
    CLIP-like preprocessing transform computed after using TargetPad pad
    :param target_ratio: target ratio for TargetPad
    :param dim: image output dimension
    :return: CLIP-like torchvision Compose transform
    """
    return Compose([
        TargetPad(target_ratio, dim),
        Resize(dim, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(dim),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
