from PIL import Image
import os
import numpy as np
import json
import random
import torch.utils.data as data
from torchvision.transforms import ToPILImage
from torchvision.datasets.utils import download_url, check_integrity
import torchvision.datasets
import torch
import torch.nn.functional as F
from torch.autograd import Variable as V
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from torchvision import transforms
from PIL import Image
import shutil
# from jpeg2dct.numpy import load, loads
from dataset.transforms import upscale, center_crop

HOME_DIR = os.getcwd()

center_crop_size = 160

if center_crop_size == 160:
    imagenette_train_dir = '../data/imagenette2-160/train/'
    imagenette_val_dir = '../data/imagenette2-160/val/'
elif center_crop_size == 320:
    imagenette_train_dir = '../data/imagenette2-320/train/'
    imagenette_val_dir = '../data/imagenette2-320/val/'


def class_to_idx(data_path):
    dataset = torchvision.datasets.ImageFolder(data_path)
    return dataset.class_to_idx


class TinyImagenet():
    def __init__(self, datatype):
        self.datatype = datatype
        self.class_to_label = class_to_idx('/data/common/tiny-imagenet-200/train/')
        self.train_data, self.train_labels, self.val_data, self.val_labels = self.process_data()

    def process_data(self):
        train_data = []
        train_labels = []
        val_data = []
        val_labels = []
        with open(HOME_DIR + '/train_list.json') as t:
            train_dict = json.load(t)
        with open(HOME_DIR + '/val_list.json') as v:
            val_dict = json.load(v)
        for class_name in train_dict:
            class_label = self.class_to_label[class_name]
            for img_file in train_dict[class_name]:
                full_path = '/data/common/tiny-imagenet-200/train/' + class_name + '/' + 'images/' + img_file
                train_data.append(full_path)
                train_labels.append(class_label)
            for val_file in val_dict[class_name]:
                full_path = '/data/common/tiny-imagenet-200/train/' + class_name + '/' + 'images/' + val_file
                val_data.append(full_path)
                val_labels.append(class_label)
        train_labels = np.asarray(train_labels, dtype=np.int)
        val_labels = np.asarray(val_labels, dtype=np.int)
        return train_data, train_labels, val_data, val_labels

    def __getitem__(self, idx):
        if self.datatype == 'val':
            img, target = self.val_data[idx], self.val_labels[idx]
        else:
            img, target = self.train_data[idx], self.train_labels[idx]
        # img = center_crop(img,(56,56))
        # print(img)
        dct_y, dct_cb, dct_cr = load(img)
        y_mean, cb_mean, cr_mean = np.load('/home/michal5/cs445/avgs.npy')
        y_std, cb_std, cr_std = np.load('/home/michal5/cs445/stds.npy')
        dct_cb = upscale(dct_cb)
        dct_cr = upscale(dct_cr)
        dct_y = np.divide(np.subtract(dct_y, y_mean), y_std)
        dct_cb = np.divide(np.subtract(dct_cb, cb_mean), cb_std)
        dct_cr = np.divide(np.subtract(dct_cr, cr_mean), cr_std)
        dct_y_t = torch.from_numpy(dct_y).float()
        dct_cr_t = torch.from_numpy(dct_cr).float()
        dct_cb_t = torch.from_numpy(dct_cb).float()
        val = torch.cat((dct_y_t, dct_cb_t, dct_cr_t), dim=1)
        return val, target

    def __len__(self):
        if self.datatype == 'val':
            return len(self.val_labels)
        else:
            return len(self.train_labels)


class TinyImagenetTest():
    def __init__(self):
        self.class_to_label = class_to_idx('/data/common/tiny-imagenet-200/train/')
        self.test_data, self.test_labels = self.process_data()

    def process_data(self):
        data = []
        labels = []
        val_annotate = open('/data/common/tiny-imagenet-200/val/val_annotations.txt')
        for line in val_annotate:
            splits = line.split()
            data.append('/data/common/tiny-imagenet-200/val/images/' + splits[0])
            labels.append(self.class_to_label[splits[1]])
        labels = np.asarray(labels, dtype=np.int)
        return data, labels

    def __getitem__(self, idx):
        img, target = self.test_data[idx], self.test_labels[idx]
        # img = center_crop(img, (56, 56))
        dct_y, dct_cb, dct_cr = load(img)
        y_mean, cb_mean, cr_mean = np.load('/home/michal5/cs445/avgs.npy')
        y_std, cb_std, cr_std = np.load('/home/michal5/cs445/stds.npy')
        dct_cb = upscale(dct_cb)
        dct_cr = upscale(dct_cr)
        dct_y = np.divide(np.subtract(dct_y, y_mean), y_std)
        dct_cb = np.divide(np.subtract(dct_cb, cb_mean), cb_std)
        dct_cr = np.divide(np.subtract(dct_cr, cr_mean), cr_std)
        dct_y_t = torch.from_numpy(dct_y).float()
        dct_cr_t = torch.from_numpy(dct_cr).float()
        dct_cb_t = torch.from_numpy(dct_cb).float()
        val = torch.cat((dct_y_t, dct_cb_t, dct_cr_t), dim=1)
        return val, target

    def __len__(self):
        return len(self.test_labels)


def get_rgb_transform():
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = torchvision.transforms.Compose([transforms.CenterCrop(64), transforms.ToTensor(), normalize, ])
    return transform


class TinyImagenetRGB():
    def __init__(self, datatype):
        self.datatype = datatype
        self.class_to_label = class_to_idx('/data/common/tiny-imagenet-200/train/')
        self.transform = get_rgb_transform()
        self.train_data, self.train_labels, self.val_data, self.val_labels = self.process_data()

    def process_data(self):
        train_data = []
        train_labels = []
        val_data = []
        val_labels = []
        with open(HOME_DIR + '/train_list.json') as t:
            train_dict = json.load(t)
        with open(HOME_DIR + '/val_list.json') as v:
            val_dict = json.load(v)
        for class_name in train_dict:
            class_label = self.class_to_label[class_name]
            for img_file in train_dict[class_name]:
                full_path = '/data/common/tiny-imagenet-200/train/' + class_name + '/' + 'images/' + img_file
                train_data.append(full_path)
                train_labels.append(class_label)
            for val_file in val_dict[class_name]:
                full_path = '/data/common/tiny-imagenet-200/train/' + class_name + '/' + 'images/' + val_file
                val_data.append(full_path)
                val_labels.append(class_label)

        train_labels = np.asarray(train_labels, dtype=np.int)
        val_labels = np.asarray(val_labels, dtype=np.int)
        return train_data, train_labels, val_data, val_labels

    def __getitem__(self, idx):
        if self.datatype == 'val':
            img, target = self.val_data[idx], self.val_labels[idx]
        else:
            img, target = self.train_data[idx], self.train_labels[idx]
        img = Image.open(img)
        if img.mode == 'L':
            tr = transforms.Grayscale(num_output_channels=3)
            img = tr(img)
        img = self.transform(img)

        return img, target

    def __len__(self):
        if self.datatype == 'val':
            return len(self.val_labels)
        else:
            return len(self.train_labels)


class TinyImagenetTestRGB():
    def __init__(self):
        self.class_to_label = class_to_idx('/data/common/tiny-imagenet-200/train/')
        self.transfrom = get_rgb_transform()
        self.test_data, self.test_labels = self.process_data()

    def process_data(self):
        data = []
        labels = []
        val_annotate = open('/data/common/tiny-imagenet-200/val/val_annotations.txt')
        for line in val_annotate:
            splits = line.split()
            data.append('/data/common/tiny-imagenet-200/val/images/' + splits[0])
            labels.append(self.class_to_label[splits[1]])
        labels = np.asarray(labels, dtype=np.int)
        return data, labels

    def __getitem__(self, idx):
        img, target = self.test_data[idx], self.test_labels[idx]
        img = Image.open(img)
        if img.mode == 'L':
            tr = transforms.Grayscale(num_output_channels=3)
            img = tr(img)
        img = self.transfrom(img)
        return img, target

    def __len__(self):
        return len(self.test_labels)


class ImagenetteTrain():
    def __init__(self, datatype):
        self.datatype = datatype
        self.class_to_label = class_to_idx(imagenette_train_dir)
        self.train_data, self.train_labels, self.val_data, self.val_labels = self.process_data()

    def process_data(self):
        train_data = []
        train_labels = []
        val_data = []
        val_labels = []
        with open(HOME_DIR + '/dataset/imagnette/train_list_imagenette.json') as t:
            train_dict = json.load(t)
        with open(HOME_DIR + '/dataset/imagnette/val_list_imagenette.json') as v:
            val_dict = json.load(v)
        for class_name in train_dict:
            class_label = self.class_to_label[class_name]
            for img_file in train_dict[class_name]:
                full_path = imagenette_train_dir + class_name + '/' + img_file
                train_data.append(full_path)
                train_labels.append(class_label)
            for val_file in val_dict[class_name]:
                full_path = imagenette_train_dir + class_name + '/' + val_file
                val_data.append(full_path)
                val_labels.append(class_label)
        train_labels = np.asarray(train_labels, dtype=np.int)
        val_labels = np.asarray(val_labels, dtype=np.int)
        return train_data, train_labels, val_data, val_labels

    def __getitem__(self, idx):
        if self.datatype == 'val':
            img, target = self.val_data[idx], self.val_labels[idx]
        else:
            img, target = self.train_data[idx], self.train_labels[idx]
        crop_img_val = center_crop(img, (center_crop_size, center_crop_size))
        # print(img)
        # dct_y, dct_cb, dct_cr = load(img)
        # y_mean, cb_mean, cr_mean = np.load(HOME_DIR + '/dataset/imagnette/avgs_imagenette_320.npy')
        # y_std, cb_std, cr_std = np.load(HOME_DIR + '/dataset/imagnette/stds_imagenette_320.npy')
        # dct_cb = upscale(dct_cb)
        # dct_cr = upscale(dct_cr)
        # dct_y = np.divide(np.subtract(dct_y, y_mean), y_std)
        # dct_cb = np.divide(np.subtract(dct_cb, cb_mean), cb_std)
        # dct_cr = np.divide(np.subtract(dct_cr, cr_mean), cr_std)
        # dct_y_t = torch.from_numpy(dct_y).float()
        # dct_cr_t = torch.from_numpy(dct_cr).float()
        # dct_cb_t = torch.from_numpy(dct_cb).float()
        #
        # val = torch.cat((dct_y_t, dct_cb_t, dct_cr_t), dim=2)

        # cb_mean = upscale(cb_mean)
        crop_img_val = torch.from_numpy(crop_img_val).float()
        crop_img_val = crop_img_val.permute(2, 0, 1)
        return crop_img_val, target
        # return val, target

    def __len__(self):
        if self.datatype == 'val':
            return len(self.val_labels)
        else:
            return len(self.train_labels)


class ImagenetteTest():
    def __init__(self):
        self.class_to_label = class_to_idx(imagenette_val_dir)
        self.test_data, self.test_labels = self.process_data()

    def process_data(self):
        data = []
        labels = []
        with open(HOME_DIR + '/dataset/imagnette/val_list_imagenette.json') as v:
            test_dict = json.load(v)
        for class_name in test_dict:
            class_label = self.class_to_label[class_name]
            for img_file in test_dict[class_name]:
                full_path = imagenette_val_dir + class_name + '/' + img_file
                data.append(full_path)
                labels.append(class_label)
        labels = np.asarray(labels, dtype=np.int)
        return data, labels

    def __getitem__(self, idx):
        img, target = self.test_data[idx], self.test_labels[idx]
        crop_img_val = center_crop(img, (center_crop_size, center_crop_size))
        # dct_y, dct_cb, dct_cr = load(img)
        # y_mean, cb_mean, cr_mean = np.load(HOME_DIR + '/dataset/imagnette/avgs_imagenette_320.npy')
        # y_std, cb_std, cr_std = np.load(HOME_DIR + '/dataset/imagnette/stds_imagenette_320.npy')
        #
        # dct_cb = upscale(dct_cb)
        # dct_cr = upscale(dct_cr)
        #
        # dct_y = np.divide(np.subtract(dct_y, y_mean), y_std)
        # dct_cb = np.divide(np.subtract(dct_cb, cb_mean), cb_std)
        # dct_cr = np.divide(np.subtract(dct_cr, cr_mean), cr_std)
        # dct_y_t = torch.from_numpy(dct_y).float()
        #
        # dct_cr_t = torch.from_numpy(dct_cr).float()
        #
        # dct_cb_t = torch.from_numpy(dct_cb).float()
        #
        # val = torch.cat((dct_y_t, dct_cb_t, dct_cr_t), dim=2)
        crop_img_val = torch.from_numpy(crop_img_val).float()
        crop_img_val = crop_img_val.permute(2, 0, 1)
        return crop_img_val, target
        # return val, target

    def __len__(self):
        return len(self.test_labels)
