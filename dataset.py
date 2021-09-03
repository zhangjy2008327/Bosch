from torch.utils.data import Dataset
# from PIL import Image
import torch
import cv2
import numpy as np
from label_scripts import visualize_labels
import config
import imageio as io
# from label_scripts.segmentation_labels import create_binary_segmentation_label

root_path = '/media/lab9102/A270A338DFEE854B/bosch'
label_image_path = '/media/lab9102/A270A338DFEE854B/bosch/labels'
shape = (256, 128)

def create_binary_segmentation_label(json_path):
    """ Creates binary segmentation image from label

    Parameters
    ----------
    json_path: str
               path to label file

    Returns
    -------
    numpy.array
        binary image, 0 for background or 1 for marker, (716, 1276), numpy.uint8
    """
    blank_image = np.zeros((717, 1276), dtype=np.uint8)
    blank_image = visualize_labels.create_segmentation_image(json_path, color=1, image=blank_image)

    return blank_image


def read_labels(label_path):
    # print('2------', label_path)
    labels = open(label_path, 'r')
    lines = labels.readlines()
    # print('1-------', len(lines))
    # exit(0)
    return lines


class RoadSequenceDatasetList(Dataset):

    def __init__(self, file_path, transforms):
        self.img_list = read_labels(file_path)
        self.dataset_size = len(self.img_list)
        self.transforms = transforms
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        # img_path_list = self.img_list[idx]
        # print('img_path_list-------------', self.img_list[idx].split()[5])

        data = []
        for i in range(5):
            new_path = root_path + self.img_list[idx].split()[i]
            img = io.imread(new_path)
            img = cv2.resize(img, shape, interpolation=cv2.INTER_CUBIC)  # shape = (320, 192)
            img = img.astype('float32') / 255.0  # (192, 320, 3)
            img = img.transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            img1 = torch.from_numpy(img)
            data.append(torch.unsqueeze(img1, dim=0))
        data = torch.cat(data, 0)
        t2 = label_image_path+self.img_list[idx].split()[5].replace('/labels_img', '')
        t2 = t2[:-4] + '.json'
        label = create_binary_segmentation_label(t2)
        label = cv2.resize(label, (256, 128), interpolation=cv2.INTER_CUBIC)
        label = torch.from_numpy(np.array(label).astype('float32'))
        sample = {'data': data, 'label': label}
        return sample

def val_read_labels(label_path):
    labels = open(label_path, 'r')
    lines = labels.readlines()
    return lines


class val_RoadSequenceDatasetList(Dataset):

    def __init__(self, file_path, transforms):
        self.img_list = val_read_labels(file_path)
        self.dataset_size = len(self.img_list)
        self.transforms = transforms
    def __len__(self):
        # print('reutn-------7777---')
        return self.dataset_size

    def __getitem__(self, idx):
        data = []
        for i in range(5):
            new_path = root_path + self.img_list[idx].split()[i]
            img = io.imread(new_path)
            img = cv2.resize(img, shape, interpolation=cv2.INTER_CUBIC)  # shape = (320, 192)
            img = img.astype('float32') / 255.0  # (192, 320, 3)
            img = img.transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            img1 = torch.from_numpy(img)
            data.append(torch.unsqueeze(img1, dim=0))
        data = torch.cat(data, 0)
        t2 = label_image_path + self.img_list[idx].split()[5].replace('/labels_img', '')
        t2 = t2[:-4] + '.json'
        label = create_binary_segmentation_label(t2)
        cv2.imwrite(config.save_path + self.img_list[idx].split()[0].split('/')[3] + '__' +
                    self.img_list[idx].split()[0].split('/')[4] + '_b.png', label * 255,
                    [cv2.IMWRITE_PNG_COMPRESSION, 0])
        label = cv2.resize(label, (256, 128), interpolation=cv2.INTER_CUBIC)
        label = torch.from_numpy(np.array(label).astype('float32'))
        tmp_name = self.img_list[idx].split()[5].split('/')[3] + '_' + self.img_list[idx].split()[5].split('/')[4]
        tmp_name = tmp_name[:-4]
        # print('5-------', tmp_name)
        # exit(0)
        sample = {'data': data, 'label': label, 'name_': tmp_name, 'new_path':new_path}
        return sample


