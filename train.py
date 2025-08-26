#coding:utf-8
import os
import json
import time
import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn.functional as F
from dataset import Clsdata
from torchsummary import summary
import argparse
import random
from torch.optim.lr_scheduler import StepLR
from loss import FocalLoss
import timm
from models import URFNet

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_all(seed: int = 123):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(model, train_root, test_root, batch_size, lr, epochs):
    # step1: data
    with open(train_root, 'r') as fr:
        train_imgs = json.load(fr)
    with open(test_root, 'r') as fr:
        test_imgs = json.load(fr)

    train_data = Clsdata(train_imgs, train=True, shuffle=True)
    test_data  = Clsdata(test_imgs,  train=False, shuffle=False, test=True)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False, num_workers=4)

    train_length = len(train_data)
    test_length  = len(test_data)
    
    # step2: criterion and optimizer
    criterion_cls = torch.nn.CrossEntropyLoss()
    #criterion_cls = FocalLoss(alpha=0.25, gamma=2.0, num_classes=13)  # 使用 Focal Loss
    #optimizer = torch.optim.SGD(filter(lambda param: param.requires_grad, model.parameters()),
                                #lr=lr, momentum=0.9, weight_decay=5e-4)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.5)

    trainLoss_cls = 0.0
    best_accuracy = 0.0
    best_acc_cls  = 0.0

    # step3: training
    for epoch in range(epochs):
        epochStart = time.time()
        print("Epoch: {}/{}".format(epoch+1, epochs))
        model.train()
        for i, (img1, img2, tgt_cls) in enumerate(train_dataloader):
            img1 = img1.to(device)
            img2 = img2.to(device)
            tgt_cls = tgt_cls.to(device)

            out_cls = model(img1, img2)
            loss_cls = criterion_cls(out_cls, tgt_cls)
            optimizer.zero_grad()
            loss_cls.backward()
            trainLoss_cls += loss_cls.item()
            optimizer.step()
        scheduler.step()

        epochEnd = time.time()
        # step4: test
        acc_cls = val(model, test_dataloader, test_length)
        print("Epoch : {:03d}, Accuracy_cls: {:.4f}%, Time: {:.4f}s".format(epoch, acc_cls*100, epochEnd-epochStart))

        #model_name = args["model"]
        best_path = f'./URFNet.pth'

        if acc_cls > best_accuracy:
            best_accuracy = acc_cls
            best_acc_cls = acc_cls
            bestEpoch = epoch
            torch.save(model, best_path)
            print(f"  -> New best model saved: {best_path} (Epoch {bestEpoch})")
        print("* best Accuracy_cls {:.4f}%".format(best_acc_cls*100))


@torch.no_grad()
def val(model, dataloader, test_len):
    model.eval()
    test_acc_cls = 0.0
    for j, (img1, img2, tgt_cls) in enumerate(dataloader):
        img1 = img1.to(device)
        img2 = img2.to(device)
        tgt_cls = tgt_cls.to(device)
        pred_cls = model(img1,img2)
        _, predictions = torch.max(pred_cls.data, 1)
        correct_counts = predictions.eq(tgt_cls.data.view_as(predictions))
        acc_cls = torch.mean(correct_counts.type(torch.FloatTensor))
        test_acc_cls += acc_cls.item() * img1.size(0)
    acc_cls = test_acc_cls / test_len
    return acc_cls


if __name__=='__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-trd", "--train_root", default='/root/autodl-tmp/train.json', help="train json path")
    ap.add_argument("-ted", "--test_root",  default='/root/autodl-tmp/test.json', help="test json path")
    ap.add_argument("-b", "--batch_size", type=int, default=32, help="training batch size")
    ap.add_argument('-l', '--learning_rate', type=float, default=1e-3, help='Learning rate')
    ap.add_argument("-e", "--epochs", type=int, default=200, help="no. of epochs")
    ap.add_argument("-im","--img-size", type=int, default=224, help="imagesize")
    ap.add_argument("-s","--seed", type=int, default=123, help="seed")
    args = vars(ap.parse_args())

    seed_all(args['seed'])
    train_root = args["train_root"]
    test_root  = args["test_root"]

    # 模型（支持 torchvision & timm）
    model = URFNet(num_classes=3)
    model = model.to(device)
    print(model)

    # 结构摘要
    #summary(model, input_size=(3, 224, 224), batch_size=args["batch_size"])
    # 训练
    train(model, train_root=train_root, test_root=test_root,
          batch_size=args["batch_size"], lr=args["learning_rate"], epochs=args["epochs"])

