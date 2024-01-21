import torch
import torch.nn as nn
from utils import iou




class Loss(nn.Module):
    def __init__(self,):
        super(Loss, self).__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        self.lambda_noobj = 1
        self.lambda_obj = 1
        self.lambda_class = 1
        self.lambda_box = 1

    def forward(self, predictions, target, anchors):
        # predictions -> (N, 3, S, S, C + 5)
        # target -> (N, 3, S, S, 6)
        # anchors -> (3, 2) 3 anchors, each anchor has width and height
        """
        forward pass for the loss function of each scale seperately

        Args:
            predictions (Tensor): Tensor with a shape (N, 3, S, S, C + 5). output of each scale of the yolov3 model.
            target (Tensor): Tensor with a shape (N, 3, S, S, 6). ground truth
            anchors (Tensor): anchors bounding boxes of the scale

        Return:
            loss (Tensor)
        """
        obj_exists = target[..., 0] == 1
        noobj_exists = target[..., 0] == 0

        ##  no object loss
        no_object_loss = self.bce(
            self.predictions[..., 0:1][noobj_exists],
            self.target[..., :1][noobj_exists]
        )

        ## object loss
        anchors = anchors.reshape((1, 3, 1, 1, 2)) # (3, 2) >>> (1, 3, 1, 1, 2)
        pred_boxes = torch.cat((
            self.sigmoid(predictions[..., 1:3]),
            torch.exp(predictions[..., 3:5]) * anchors
        ), dim=-1)
        ious = iou(
            pred_boxes[obj_exists],
            target[..., 1:5][obj_exists]
        ).detach()

        object_loss = self.bce(
            predictions[..., 0:1][obj_exists], (target[..., 0:1][obj_exists] * ious) 
        )

        ## coordinate loss
        predictions[..., 1:3] = torch.sigmoid(predictions[..., 1:3]) # from the paper bx = sigmoid(tx)
        target[..., 3:5] = torch.log(
            target[..., 3:5] / anchors  # instead of doing exp(tw) for better gradient flow
        )

        box_loss = self.mse(
            predictions[..., 1:5][obj_exists], target[..., 1:5][obj_exists]
        )

        # class loss
        class_loss = self.entropy(
            (predictions[..., 5:][obj_exists]), (target[..., 5][obj_exists]).long()
        )

        return (
            self.lambda_noobj * no_object_loss
            + self.lambda_obj * object_loss
            + self.lambda_box * box_loss
            + self.lambda_class + class_loss
        )

