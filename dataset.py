import os
import cv2
import torch
from glob import glob
from PIL import Image
from utils import iou_width_height

class Dataset(torch.utils.data.Dataset):
    """
    Dataset class for object detection which is compatible with PyTorch's Dataset API.
    This class is responsible for loading and transforming object detection datasets.
    
    Attributes:
        path (str): Path to the dataset directory.
        anchors (list): List containing three lists of anchor boxes. Each sub-list represents
                        anchor boxes at a different scale.
        S (list, optional): List of sizes of feature maps. Default is [13, 26, 52].
        C (int, optional): Number of classes in the dataset. Default is 20.
        transform (callable, optional): A function/transform that takes in a sample and returns
                                        a transformed version. E.g, `transforms.RandomCrop` for images.
        ignore_iou_thresh (float, optional): Threshold for ignoring anchor that also has a IOU 
                                             greater than this threshold. Default is 0.5.
        
    Parameters:
        path (str): Path to the directory where the dataset is located.
        anchors (list): List of anchors grouped by each detection scale. Each element in the list
                        is a list of tuples representing the anchor boxes for that scale.
        mode (str, optional): Specifies the dataset mode (e.g., 'train', 'test'). Default is 'train'.
        S (list, optional): Sizes of feature maps for different scales. Default is [13, 26, 52].
        C (int, optional): Number of classes. Default is 20.
        transform (callable, optional): A function/transform that takes an image as input and returns
                                        a transformed version. Default is None.
        ignore_iou_thresh (float, optional): IOU threshold to ignore anchors when the IOU is above
                                             this threshold. Default is 0.5.
    """
    def __init__(
        self, path,
        anchors,
        mode='train',
        S=[13, 26, 52],
        C=20,
        transoform=None,
        ignore_iou_thresh=0.5
    ):
        self.path = path
        self.mode = mode
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors = self.anchors.shape[0]
        self.S = S
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_thresh = ignore_iou_thresh

        self.images = sorted(os.listdir(self.path + "/" + self.mode + "/images"))
        self.labels = sorted(os.listdir(self.path + "/" + self.mode + "/labels"))

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        label = os.path.join(self.path, self.mode, "labels", self.labels[index])
        boxes = np.roll(self.check_txtfile(label), 4, axis=1).tolist()

        image = Image.open(os.path.join(self.path, self.mode, 'images', self.images[index]))
        boxes = torch.tensor(boxes)

        if self.transform:
            image, boxes = self.transform(image, boxes)

        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]

        for box in boxes:
            iou_anchors = iou_width_height(box[..., 2:4], self.anchors)
            anchor_indices = torch.argsort(iou_anchors, descending=True)
            x, y, w, h, class_id = box
            has_anchors = [False] * 3

            for anchor_idx in anchor_indices:
                #  e.g. anchor_idx = 8 && num_anchors_per_scale = 3
                scale_idx = anchor_idx // self.num_anchors_per_scale  # e.g. 8 // 3 -> 2 -> 2nd scale index
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale # e.g. 8 % 3 -> 1 -> first anchor on second scale
                S = self.S[scale_idx]
                i, j = int(y * S), int(x * S) # e.g. x = 0.5 && S = 13 -> j = (13 * 0.5) = 6
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]

                if not anchor_taken and has_anchors[scale]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = x * S - j, y * S - i
                    w_cell, h_cell = (
                        w * S,
                        h * S
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = torch.tensor([
                        x_cell, y_cell, w_cell, h_cell
                    ])
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_id)
                    has_anchors[scale_idx] = True

                if not anchor_taken and  iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1
        
        return image, tuple(targets)



    def check_txtfile(self, txtfile):
        # check if txtfile exist
        boxes = []
        if not os.path.exists(txtfile):
            return boxes

        with open(txtfile, 'r') as f:
            for label in f.readlines():
                class_id, x, y, w, h = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]

                boxes.append([class_id, x, y, w, h])
        
        return boxes