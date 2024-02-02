import cv2
import os
import sys
import torch
import logging
import torch.optim as optim
from tqdm import tqdm
from loss import Loss
from dataset import Dataset
from dataset import Compose
from model import Yolov3
from torchvision import transforms
import validate
import config
from metrics import mean_average_precision
from utils import save_model, load_model
import albumentations as A

IMAGE_SIZE = 416
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]

ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]
LR = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 1000


transform = Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])


torch.backends.cudnn.benchmark = True
def train_step(model, trainloader, loss_fn, optimizer, scaler, scheduler, scaled_anchors, epoch=0, device='cuda'):
    pbar = tqdm(trainloader, ncols=150)
    losses = []

    for batch_idx, (x, y) in enumerate(pbar):
        x = x.to(device)
        y1, y2, y3 = (
            y[0].to(device),
            y[1].to(device),
            y[2].to(device)
        )

        with torch.cuda.amp.autocast():
            out = model(x)

            loss = (
                loss_fn(out[0], y1, scaled_anchors[0]) + 
                loss_fn(out[1], y2, scaled_anchors[1]) +
                loss_fn(out[2], y3, scaled_anchors[2])
            )

        
        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        mean_loss = sum(losses) / len(losses)

        current_memory = torch.cuda.memory_allocated() / (1024 ** 3)
        max_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)

        lr = scheduler.get_last_lr()[0]
        pbar.set_postfix(epoch=f"{epoch + 1} / {EPOCHS}", loss=f"{mean_loss:.2f}", gpu_usage=f"{current_memory:.2f} Gb", max_gpu_usage=f"{max_memory:.2f} Gb", lr=f"{lr}")


def main():

    from test_model import YOLOv3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = YOLOv3(num_classes=20).to(device) 
    model =  Yolov3(in_channels=3, num_classes=20).to(device)
    
    load_model(model, checkpoint_file='./yolov3_best.pth')
    loss = Loss()
    # loss = YoloLoss()

    optimizer = optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    scaler = torch.cuda.amp.GradScaler()
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 500], gamma=0.1)

    dataset = Dataset(
        path='/home/appuser/data',
        mode='train',
        anchors=ANCHORS,
        transform=config.train_transforms
    )

    trainloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    scaled_anchors = (
        torch.tensor(ANCHORS)
        * torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(device)

    val_dataset = Dataset(
        path='/home/appuser/data',
        mode='test',
        anchors=ANCHORS,
        transform=config.test_transforms
    )

    valloader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=True)

    best_score = -1
    mAP = 0.
    for epoch in range(EPOCHS):
        train_step(
            model=model,
            trainloader=trainloader,
            optimizer=optimizer,
            loss_fn=loss,
            scaler=scaler,
            scheduler=scheduler,
            scaled_anchors=scaled_anchors,
            device=device
        )


        scheduler.step()

        if epoch > 0 and not epoch % 5:
            mAP = validate.val(model, valloader, device=device, anchors=ANCHORS, conf=0.45)
            tqdm.write(f'Mean Average Precision: {mAP}')

        
        checkpoint = {
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'map': mAP
        }

        save_model(model, save_path='./yolov3_last.pth', checkpoint=checkpoint)
        
        if mAP > best_score:
            save_model(model, save_path='./yolov3_best.pth', checkpoint=checkpoint)
            best_score = mAP


if __name__ == "__main__":
    main()