# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import csv
import warnings
warnings.simplefilter("ignore", UserWarning)
import PIL
import torch
import random
import math
from enum import Enum
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

from PIL import ImageFilter, ImageOps
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import Tensor
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F
from torch.utils.data import Dataset

from glob import glob
from pathlib import Path

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        cutout_size (int): The length (in pixels) of each square patch.
    """
    def __init__(self, cutout_num, cutout_size):
        self.cutout_num = cutout_num
        self.cutout_size = cutout_size

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with cutout_num of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)
        
        mask = torch.ones(h, w).to(torch.float32)
        
        for n in range(self.cutout_num):
            # y = np.random.randint(h)
            # x = np.random.randint(w)

            y = torch.randint(0, h, size=(1,)).item()
            x = torch.randint(0, w, size=(1,)).item()

            y1 = np.clip(y - self.cutout_size // 2, 0, h)
            y2 = np.clip(y + self.cutout_size // 2, 0, h)
            x1 = np.clip(x - self.cutout_size // 2, 0, w)
            x2 = np.clip(x + self.cutout_size // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = mask.expand_as(img)
        img = img * mask

        return img


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR
    https://arxiv.org/abs/2002.05709.
    """

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"sigma = {self.sigma}, "
        return repr_str


class Solarization(object):
    """Solarization augmentation in BYOL
    https://arxiv.org/abs/2006.07733.
    """

    def __call__(self, x):
        return ImageOps.solarize(x)


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'validate') #'val'
    
    dataset = datasets.ImageFolder(root, transform=transform)
    
    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',   # 'bicubic'
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        # crop_pct = 224 / 256
        crop_pct = 0.95
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        # transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
        transforms.Resize(size, interpolation=InterpolationMode.BICUBIC),
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


def get_params(img, scale=(0.2, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0)):
    height, width = img.size
    area = height * width

    log_ratio = torch.log(torch.tensor(ratio))
    for _ in range(10):
        target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
        aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if 0 < w <= width and 0 < h <= height:
            i = torch.randint(0, height - h + 1, size=(1,)).item()
            j = torch.randint(0, width - w + 1, size=(1,)).item()
            return i, j, h, w

    # Fallback to central crop
    in_ratio = float(width) / float(height)
    if in_ratio < min(ratio):
        w = width
        h = int(round(w / min(ratio)))
    elif in_ratio > max(ratio):
        h = height
        w = int(round(h * max(ratio)))
    else:  # whole image
        w = width
        h = height
    i = (height - h) // 2
    j = (width - w) // 2
    return i, j, h, w


def check_xy_in_region(xy, region):
    x, y = xy
    region_x = region[:, 0].numpy()
    region_y = region[:, 1].numpy()
    x_index = np.where(region_x==x)[0].tolist()
    y_index = np.where(region_y==y)[0].tolist()
    flag = False
    for idx in list(set(x_index) & set(y_index)):
        x_, y_ = region[idx]
        if x == x_ and y == y_:
            flag = True
    return flag


def xyxy_to_xywh(xyxy):
    """Convert [x1 y1 x2 y2] box format to [x1 y1 w h] format."""
    if isinstance(xyxy, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(xyxy) == 4
        x1, y1 = xyxy[0], xyxy[1]
        w = xyxy[2] - x1 + 1
        h = xyxy[3] - y1 + 1
        return (x1, y1, w, h)
    elif isinstance(xyxy, np.ndarray):
        # Multiple boxes given as a 2D ndarray
        return np.hstack((xyxy[:, 0:2], xyxy[:, 2:4] - xyxy[:, 0:2] + 1))
    else:
        raise TypeError('Argument xyxy must be a list, tuple, or numpy array.')


def revise_box(corners, mask_nonzeros, degree):

    revise_corners = []
    region_x = mask_nonzeros[:, 0].numpy()
    region_y = mask_nonzeros[:, 1].numpy()

    # """
    for idx, corner in enumerate(corners):
        x, y = corner
        # top_left
        if idx == 0:
            if not check_xy_in_region(corner, mask_nonzeros):
                if degree > 0:
                    y_index = np.where(region_y==y)[0].tolist()
                    x_r = region_x[y_index].min()
                    revise_corners.append((x_r, y))
                else:
                    x_index = np.where(region_x==x)[0].tolist()
                    y_r = region_y[x_index].min()
                    revise_corners.append((x, y_r))
            else:
                revise_corners.append(corner)
        # top_right
        elif idx == 1:
            x1_r, y1_r = revise_corners[0]
            if not check_xy_in_region(corner, mask_nonzeros):
                if degree > 0:
                    x_index = np.where(region_x==x1_r)[0].tolist()
                    y2_r = region_y[x_index].max()
                    revise_corners.append((x1_r, y2_r))
                else:
                    y_index = np.where(region_y==y)[0].tolist()
                    x_r = region_x[y_index].min()
                    revise_corners.append((x_r, y))
                    revise_corners[0] = (x_r, y1_r)
            else:
                if degree > 0:
                    revise_corners.append((x1_r, y))
                else:
                    revise_corners.append(corner)
        
        # bottom_left
        elif idx == 2:
            x1_r, y1_r = revise_corners[0]
            x1_r, y2_r = revise_corners[1]
            if not check_xy_in_region(corner, mask_nonzeros):
                if degree > 0:
                    x_index = np.where(region_x==x)[0].tolist()
                    y1_r = region_y[x_index].min()
                    revise_corners.append((x, y1_r))
                    revise_corners[0] = (x1_r, y1_r)
                else:
                    y_index = np.where(region_y==y1_r)[0].tolist()
                    x2_r = region_x[y_index].max()
                    revise_corners.append((x2_r, y1_r))
            else:
                if degree > 0:
                    revise_corners.append(corner)
                else:
                    revise_corners.append((x, y1_r))
        # bottom_right
        elif idx == 3:
            x1_r, y1_r = revise_corners[0]
            x1_r, y2_r = revise_corners[1]
            x2_r, y1_r = revise_corners[2]
            br_corner = (x2_r, y2_r)
            if not check_xy_in_region(br_corner, mask_nonzeros):
                if degree > 0:
                    y_index = np.where(region_y==y2_r)[0].tolist()
                    x2_r = region_x[y_index].max()
                    revise_corners.append((x2_r, y2_r))
                    revise_corners[2] = (x2_r, y1_r)
                else:
                    x_index = np.where(region_x==x2_r)[0].tolist()
                    y2_r = region_y[x_index].max()
                    revise_corners.append((x2_r, y2_r))
                    revise_corners[1] = (x1_r, y2_r)
            else:
                revise_corners.append(br_corner)
    return revise_corners

import Image

class DatasetType(Enum):
    """
    Used to differentiate between expected data structures
    """

    CSV = 0
    CLASSIFICATION = 1
    # Uses indicators in the filenames to split
    FILENAME = 2

def pil_image_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

def read_paths_csv(csv_path: str) -> List[str]:
    """
    Given a csv of paths, read them as a comma separated row
    """
    path_list = []

    with open(csv_path, "r") as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            path_list.extend(row)

    print(f"Read {len(path_list)} filenames from {csv_path}")

    return path_list


def get_file_stem(path_list: List[str]):
    """
    Remove the exensions from file name and return the
    extension free file name. This function is required
    because some file names include '.'
    which naively splitting based on '.' on file names
    without extensions will produce non existent filenames
    """
    _path_list = []
    for path in path_list:
        if Path(path).suffix in [".png", ".jpg", ".bmp", ".tif", ".tiff"]:
            _path_list.append(Path(path).stem)
        else:
            _path_list.append(Path(path).name)

    return _path_list
    
class PretrainDataset(Dataset):
    """
    Base dataset for pre-training
    """

    def __init__(self, images_list, transform):
        super().__init__()
        self.images_list = images_list
        self.transform = transform

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        path = self.images_list[index]
        sample = pil_image_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
    
class CorrlationDataset2(Dataset):

    def __init__(
        self,
        images_list,
        search_size=224,
        context_size=176,
        template_size=64,
        template_num=6,
        scale=(0.2, 1.0),
        ratio=(1.0, 1.0),
        degree=45,
        interpolation=InterpolationMode.BICUBIC,
        transform=None,
    ):

        super(CorrlationDataset, self).__init__()

        self.images_list = images_list
        self.search_size = search_size
        self.context_size = context_size
        self.template_size = template_size
        self.template_num = template_num
        self.scale = scale
        self.ratio = ratio
        self.degree = degree
        self.interpolation = interpolation
        self.transform = transform

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        path = self.images_list[index]
        image = pil_image_loader(path)
        # path, _ = self.samples[index]
        # image = self.loader(path)
        
        search_image = self.transform["search"](image)

        masks = []
        template_images = []
        for _ in range(self.template_num):
            degree = torch.randint(-self.degree, self.degree, size=(1,)).item()
            H, W = search_image.size
            i, j, h, w = get_params(search_image, self.scale, self.ratio)
            
            tmp_mask_img = torch.ones(1, H, W).type(torch.uint8)
            tmp_mask_img[:, i:i+h,j:j+w] = 2

            rotate_search_img  = F.rotate(search_image, degree, expand=1)
            rotate_mask_img  = F.rotate(tmp_mask_img, degree, expand=1)

            region_mask = rotate_mask_img == 2
            mask = rotate_mask_img.squeeze(0)
            mask_nonzeros = mask.nonzero()

            region_nonzeros = (mask == 2).nonzero()
            x1, x2 = region_nonzeros[:, 0].min().item(), region_nonzeros[:, 0].max().item()
            y1, y2 = region_nonzeros[:, 1].min().item(), region_nonzeros[:, 1].max().item()

            corners = [(x1, y1), (x1, y2), (x2, y1), (x2, y2)] # top_left, top_right, bottom_left, bottom_right

            revise_corners = revise_box(corners, mask_nonzeros, degree)
            x1_r, y1_r = revise_corners[0]
            x2_r, y2_r = revise_corners[3]
            i_r, j_r, h_r, w_r = xyxy_to_xywh([x1_r, y1_r, x2_r, y2_r])
            template_image = F.resized_crop(rotate_search_img, i_r, j_r, h_r, w_r, (self.template_size, self.template_size), self.interpolation)
            
            rotate_mask_img[:, x1_r:x2_r+1, y1_r:y2_r+1] = 3

            reverse_search_img  = F.rotate(rotate_search_img, -degree, expand=0)
            reverse_mask_img  = F.rotate(rotate_mask_img, -degree, expand=0)

            reverse_mask_img_nonzeros = reverse_mask_img.squeeze(0).nonzero()

            x1_rr, x2_rr = reverse_mask_img_nonzeros[:, 0].min().item(), reverse_mask_img_nonzeros[:, 0].max().item()
            y1_rr, y2_rr = reverse_mask_img_nonzeros[:, 1].min().item(), reverse_mask_img_nonzeros[:, 1].max().item()

            i_rr, j_rr, h_rr, w_rr = xyxy_to_xywh([x1_rr, y1_rr, x2_rr, y2_rr])
            
            crop_search_img = F.resized_crop(reverse_search_img, i_rr, j_rr, h_rr, w_rr, (self.context_size, self.context_size), self.interpolation)
            crop_mask_img = F.resized_crop(reverse_mask_img, i_rr, j_rr, h_rr, w_rr, (self.context_size, self.context_size), self.interpolation)

            region_mask = crop_mask_img == 3
            mask_ = torch.zeros(1, self.context_size, self.context_size).to(torch.int64)
            mask_.masked_fill_(region_mask, 1)
            masks.append(mask_[0])
            template_images.append(template_image)

        context_image = F.resize(search_image, (self.context_size, self.context_size), self.interpolation)
        context_image = self.transform["post_context"](context_image)

        temp_images = []
        for template_image in template_images:
            if self.transform["common"]:
                template_image = self.transform["common"](template_image)
            if self.transform["template"]:
                template_image = self.transform["template"](template_image)
            template_image = self.transform["post_template"](template_image)
            temp_images.append(template_image)

        return context_image, temp_images, masks
    
def get_custom_pretrain_dataset(image_directory_list: List[str], split_name, transform,search_size,context_size,template_size,template_num,scale,ratio,degree):
    # Assumes that the split csv file is in the same directory as the image data

    # for each directory find the split csv file
    # then get the full paths of the images in that csv file
    # the assert that the number of collected and viable paths is the same as the csv List
    # output the number of files used from this directory

    # then extend a larger list of files that will be used to create the dataset

    sample_paths = []

    for img_dir in image_directory_list:
        assert os.path.exists(img_dir), "DNE: {}".format(img_dir)
        csv_path = os.path.join(img_dir, f"{split_name}.csv")
        included_paths = read_paths_csv(csv_path)
        included_paths_stems = get_file_stem(included_paths)
        file_paths = glob(os.path.join(img_dir, "*"))
        _sample_paths = [x for x in file_paths if Path(x).stem in included_paths_stems]
        _sample_stems = [Path(x).stem for x in _sample_paths]
        # validate filtering
        print(
            f"[info] Path differences: {set(_sample_stems).symmetric_difference(set(included_paths_stems))}"
        )
        assert len(included_paths) == len(
            _sample_paths
        ), f"{len(_sample_paths) = }, {len(included_paths) = }"
        print(f"[info] Loading {len(_sample_paths) = } from {img_dir}")
        sample_paths.extend(_sample_paths)

    print(f"[info] Using {len(sample_paths) = } total files")
    return CorrlationDataset2(images_list=sample_paths, transform = transform,search_size = search_size,context_size = context_size,template_size = template_size,template_num=template_num,scale=scale,ratio=ratio,degree=degree)


def get_classification_pretrain_dataset(image_directory_list: List[str], transform,search_size,context_size,template_size,template_num,scale,ratio,degree):
    # validate directory existence and get images
    sample_paths = []
    for img_dir in image_directory_list:
        assert os.path.exists(img_dir), "DNE: {}".format(img_dir)
        files = glob(os.path.join(img_dir, "*"))
        sample_paths.extend(files)

    # sort based on file names
    sample_paths = sorted(sample_paths, key=lambda x: Path(x).stem)
    print(f"Found {len(sample_paths) = } images")

    return CorrlationDataset2(images_list=sample_paths, search_size= search_size,context_size=context_size,template_size=template_size,template_num=template_num,scale = scale,ratio = ratio,degree = degree,transform=transform)


def get_filename_pretrain_dataset(dataset: CorrlationDataset2, split_name):
    assert split_name in ["train", "val", "test"]
    orig_len = len(dataset)
    dataset.images_list = [
        x for x in dataset.images_list if split_name in x and ".csv" not in x
    ]
    print(f"{orig_len = }, {len(dataset) = }")
    return dataset

def get_pretrain_dataset(
    image_directory_list: List[str],
    directory_type: DatasetType,
    transform,
    search_size,
    context_size,
    template_size,
    template_num,
    scale,
    ratio, #(3.0 / 4.0, 4.0 / 3.0),
    degree,
    split_name=None
) -> CorrlationDataset2:
    """
    Returns an initialized PretrainDataset
    """
    # Get the full paths
    image_directory_list = [
        os.path.abspath(os.path.expanduser(x)) for x in image_directory_list
    ]

    if directory_type == DatasetType.CSV:
        return get_custom_pretrain_dataset(image_directory_list, split_name, transform,search_size,context_size,template_size,template_num,scale,ratio,degree)
    elif directory_type == DatasetType.CLASSIFICATION:
        return get_classification_pretrain_dataset(image_directory_list, transform,search_size,context_size,template_size,template_num,scale,ratio,degree)
    elif directory_type == DatasetType.FILENAME:
        dataset = get_classification_pretrain_dataset(image_directory_list, transform,search_size,context_size,template_size,template_num,scale,ratio,degree)
        return get_filename_pretrain_dataset(dataset, split_name)



class CorrlationDataset(datasets.ImageFolder):

    def __init__(
        self,
        data_path,
        search_size=224,
        context_size=176,
        template_size=64,
        template_num=6,
        scale=(0.2, 1.0),
        ratio=(1.0, 1.0),
        degree=45,
        interpolation=InterpolationMode.BICUBIC,
        transform=None,
    ):

        super(CorrlationDataset, self).__init__(data_path)

        self.search_size = search_size
        self.context_size = context_size
        self.template_size = template_size
        self.template_num = template_num
        self.scale = scale
        self.ratio = ratio
        self.degree = degree
        self.interpolation = interpolation
        self.transform = transform

    def __getitem__(self, index):
        path, _ = self.samples[index]
        image = self.loader(path)
        
        search_image = self.transform["search"](image)

        masks = []
        template_images = []
        for _ in range(self.template_num):
            degree = torch.randint(-self.degree, self.degree, size=(1,)).item()
            H, W = search_image.size
            i, j, h, w = get_params(search_image, self.scale, self.ratio)
            
            tmp_mask_img = torch.ones(1, H, W).type(torch.uint8)
            tmp_mask_img[:, i:i+h,j:j+w] = 2

            rotate_search_img  = F.rotate(search_image, degree, expand=1)
            rotate_mask_img  = F.rotate(tmp_mask_img, degree, expand=1)

            region_mask = rotate_mask_img == 2
            mask = rotate_mask_img.squeeze(0)
            mask_nonzeros = mask.nonzero()

            region_nonzeros = (mask == 2).nonzero()
            x1, x2 = region_nonzeros[:, 0].min().item(), region_nonzeros[:, 0].max().item()
            y1, y2 = region_nonzeros[:, 1].min().item(), region_nonzeros[:, 1].max().item()

            corners = [(x1, y1), (x1, y2), (x2, y1), (x2, y2)] # top_left, top_right, bottom_left, bottom_right

            revise_corners = revise_box(corners, mask_nonzeros, degree)
            x1_r, y1_r = revise_corners[0]
            x2_r, y2_r = revise_corners[3]
            i_r, j_r, h_r, w_r = xyxy_to_xywh([x1_r, y1_r, x2_r, y2_r])
            template_image = F.resized_crop(rotate_search_img, i_r, j_r, h_r, w_r, (self.template_size, self.template_size), self.interpolation)
            
            rotate_mask_img[:, x1_r:x2_r+1, y1_r:y2_r+1] = 3

            reverse_search_img  = F.rotate(rotate_search_img, -degree, expand=0)
            reverse_mask_img  = F.rotate(rotate_mask_img, -degree, expand=0)

            reverse_mask_img_nonzeros = reverse_mask_img.squeeze(0).nonzero()

            x1_rr, x2_rr = reverse_mask_img_nonzeros[:, 0].min().item(), reverse_mask_img_nonzeros[:, 0].max().item()
            y1_rr, y2_rr = reverse_mask_img_nonzeros[:, 1].min().item(), reverse_mask_img_nonzeros[:, 1].max().item()

            i_rr, j_rr, h_rr, w_rr = xyxy_to_xywh([x1_rr, y1_rr, x2_rr, y2_rr])
            
            crop_search_img = F.resized_crop(reverse_search_img, i_rr, j_rr, h_rr, w_rr, (self.context_size, self.context_size), self.interpolation)
            crop_mask_img = F.resized_crop(reverse_mask_img, i_rr, j_rr, h_rr, w_rr, (self.context_size, self.context_size), self.interpolation)

            region_mask = crop_mask_img == 3
            mask_ = torch.zeros(1, self.context_size, self.context_size).to(torch.int64)
            mask_.masked_fill_(region_mask, 1)
            masks.append(mask_[0])
            template_images.append(template_image)

        context_image = F.resize(search_image, (self.context_size, self.context_size), self.interpolation)
        context_image = self.transform["post_context"](context_image)

        temp_images = []
        for template_image in template_images:
            if self.transform["common"]:
                template_image = self.transform["common"](template_image)
            if self.transform["template"]:
                template_image = self.transform["template"](template_image)
            template_image = self.transform["post_template"](template_image)
            temp_images.append(template_image)

        return context_image, temp_images, masks