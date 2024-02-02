import cv2
import os
import torch
import numpy as np
from model import Yolov3
from metrics import nms
from utils import convert_predboxes
from utils import draw_bounding_boxes

import matplotlib.pyplot as plt
from dataset import Dataset
from config import ANCHORS
from config import test_transforms

def predict(
    model_path,
    anchors,
    src="/home/appuser/data/test/images/",
    iou_threshold=0.5,
    conf=0.5,
    device='cpu',
):
    

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(model_path)
    model = Yolov3(in_channels=3, num_classes=20).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    

    orig_image = cv2.imread(src)
    image = cv2.resize(orig_image, (416, 416))

    image = (image / 255.).astype(np.float32)

    image = torch.tensor(image).unsqueeze(0)
    image = image.permute(0, 3, 1, 2)
    image = image.to(device)

    out = model(image)
    
    bboxes = []
    for i, pred_per_scale in enumerate(out):
        # prediction -> (N, 3, S, S, 5 + C)
        S = pred_per_scale.shape[2]
        anchor = torch.tensor([*anchors[i]]).to(device) * S
        pred_boxes = convert_predboxes(
            predictions=pred_per_scale,
            anchors=anchor,
            S=S,
        )

        for idx, (box) in enumerate(pred_boxes):
            bboxes += box
    
    boxes_after_nms = nms(
        bboxes, iou_threshold=iou_threshold, conf=conf, xywh=True
    )

    print(len(boxes_after_nms))


    annotated_image = draw_bounding_boxes(
        image=orig_image,
        boxes=boxes_after_nms,
    )


    plt.imshow(annotated_image)
    plt.show()
    


    print("Succeed")


if __name__ == "__main__":

    test_data = Dataset(
        path='/home/appuser/data',
        mode='train',
        anchors=ANCHORS,
        transform=test_transforms
    )


    dataloader = torch.utils.data.DataLoader(test_data, batch_size=16)

    predict(
        model_path='./yolov3_last.pth', 
        src="/home/appuser/data/test/images/000002.jpg",
        anchors=ANCHORS,
        iou_threshold=0.45,
        conf=0.5,
        device='cpu'
        )
