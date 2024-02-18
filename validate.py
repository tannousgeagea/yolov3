import torch
from tqdm import tqdm
from utils import convert_trueboxes, convert_predboxes
from metrics import nms, mean_average_precision, iou


def val(model, validationloader, anchors, iou_threshold=0.45, conf=0.25, device='cuda'):

    model.eval()
    pbar = tqdm(validationloader, ncols=125)
    predictions = []
    target = []
    train_idx = 0

    for batch_idx, (x, y) in enumerate(pbar):
        x = x.to(device)
        y1, y2, y3 = (
            y[0].to(device),
            y[1].to(device),
            y[2].to(device),
        )

        with torch.no_grad():
            out = model(x)

        batch_size = x.shape[0]
        bboxes = [[] for _ in range(batch_size)]
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
                bboxes[idx] += box

        true_boxes = convert_trueboxes(
            y[2], S=S
        )
        

        for idx in range(batch_size):
            boxes_after_nms = nms(
                bboxes[idx], iou_threshold=iou_threshold, conf=conf, xywh=True
            )

            for box in boxes_after_nms:
                predictions.append([train_idx] + box)

            for box in true_boxes[idx]:
                if box[1] > 0.5:
                    target.append([train_idx] + box)

            train_idx += 1

    print(len(predictions))
    if len(predictions):
        print(predictions[0])
    print(target[0])
    mAP, _ = mean_average_precision(
        predictions=predictions,
        target=target,
        num_classes=20
    )
    
    model.train()
    return mAP



if __name__ == "__main__":

    from train import transform
    from model import Yolov3
    from dataset import Dataset
    import config
    from utils import save_model, load_model 
    IMAGE_SIZE = 416
    S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]
    ANCHORS = [
        [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
        [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
        [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
    ]

    val_dataset = Dataset(
        path='/home/appuser/data',
        mode='test',
        anchors=ANCHORS,
        transform=config.test_transforms
    )

    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model =  Yolov3(in_channels=3, num_classes=20).to(device)
    
    load_model(model, checkpoint_file='./yolov3_last.pth')
    mAP = val(model, valloader, device=device, anchors=ANCHORS, conf=0.45)

    print("mAP: %s" % mAP)
