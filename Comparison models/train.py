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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_all(seed: int = 123):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



TIMM_FAMILIES = ("efficientnet", "vit", "deit", "efficientformer")

def build_model(name: str, num_classes: int, pretrained: bool = True):

    name = name.lower()
    try:
        if name == "alexnet":
            m = models.alexnet(weights=models.AlexNet_Weights.DEFAULT if pretrained else None)
            in_f = m.classifier[6].in_features
            m.classifier[6] = nn.Linear(in_f, num_classes)
        elif name in ["vgg16", "vgg19"]:
            if name == "vgg16":
                m = models.vgg16(weights=models.VGG16_Weights.DEFAULT if pretrained else None)
            else:
                m = models.vgg19(weights=models.VGG19_Weights.DEFAULT if pretrained else None)
            in_f = m.classifier[6].in_features
            m.classifier[6] = nn.Linear(in_f, num_classes)
        elif name in ["resnet18","resnet34","resnet50"]:
            ctor = {"resnet18": models.resnet18,
                    "resnet34": models.resnet34,
                    "resnet50": models.resnet50}[name]
            weights = {
                "resnet18": models.ResNet18_Weights.DEFAULT,
                "resnet34": models.ResNet34_Weights.DEFAULT,
                "resnet50": models.ResNet50_Weights.DEFAULT,
            }[name] if pretrained else None
            m = ctor(weights=weights)
            in_f = m.fc.in_features
            m.fc = nn.Linear(in_f, num_classes)
        elif name in ["densenet121","densenet169"]:
            if name == "densenet121":
                m = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if pretrained else None)
            else:
                m = models.densenet169(weights=models.DenseNet121_Weights.DEFAULT if pretrained else None)
            in_f = m.classifier.in_features
            m.classifier = nn.Linear(in_f, num_classes)
        elif name in ["mobilenet_v2","mobilenet_v3_small"]:
            if name == "mobilenet_v2":
                m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None)
            elif name == "mobilenet_v3_small":
                m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
            else:
                m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
            last = m.classifier[-1]
            m.classifier[-1] = nn.Linear(last.in_features, num_classes)

        elif name.startswith(TIMM_FAMILIES):
            m = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)

        else:
            raise ValueError(f"Unsupported model: {name}")

    except Exception:
        ctor = {
            "alexnet": models.alexnet,
            "vgg16": models.vgg16, "vgg19": models.vgg19,
            "resnet18": models.resnet18, "resnet34": models.resnet34, "resnet50": models.resnet50,
            "densenet121": models.densenet121,
            "densenet169": models.densenet169,
            "mobilenet_v2": models.mobilenet_v2,
            "mobilenet_v3_small": models.mobilenet_v3_small,
        }.get(name, None)
        if ctor is None:
            if name.startswith(TIMM_FAMILIES):
                m = timm.create_model(name, pretrained=True, num_classes=num_classes)
                return m
            raise
        m = ctor(pretrained=True)
        return build_model(name, num_classes, pretrained=False)
    return m


def freeze_backbone(model, name: str):

    name = name.lower()
    if name in ["alexnet","vgg16","vgg19"]:
        for p in model.features.parameters():
            p.requires_grad = False

    elif name.startswith("resnet") or name.startswith("densenet") or name.startswith("mobilenet"):
        for p in model.parameters():
            p.requires_grad = False
        head = (["fc"] if name.startswith("resnet") else
                ["classifier"] if name.startswith("mobilenet") or name=="densenet121"
                else [])
        for n, m in model.named_modules():
            if any(h == n for h in head):
                for p in m.parameters():
                    p.requires_grad = True
        if name in ["densenet121", "densenet169"]:
            for p in model.classifier.parameters():
                p.requires_grad = True
        if name.startswith("mobilenet"):
            for p in model.classifier.parameters():
                p.requires_grad = True
        if name.startswith("resnet"):
            for p in model.fc.parameters():
                p.requires_grad = True

    elif name.startswith(TIMM_FAMILIES):
        head_keys = ("head", "fc", "classifier", "heads", "last_layer")
        for p in model.parameters():
            p.requires_grad = False
        for n, p in model.named_parameters():
            if any(k in n for k in head_keys):
                p.requires_grad = True

    return model
    

def train(model, train_root, test_root, batch_size, lr, epochs):
    # step1: data
    with open(train_root, 'r') as fr:
        train_imgs = json.load(fr)
    with open(test_root, 'r') as fr:
        test_imgs = json.load(fr)
        

    train_data = Clsdata(train_imgs, train=True, shuffle=True)
    test_data  = Clsdata(test_imgs, train=False, shuffle=False, test=True)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader  = DataLoader(test_data,  batch_size=batch_size, shuffle=False, num_workers=4)

    train_length = len(train_data)
    test_length  = len(test_data)
    
    # step2: criterion and optimizer
    criterion_cls = torch.nn.CrossEntropyLoss()
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
        for i, (img, tgt_cls) in enumerate(train_dataloader):
            img = img.to(device)
            tgt_cls = tgt_cls.to(device)

            out_cls = model(img)
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

        model_name = args["model"]
        best_path = f'./{model_name}.pth'

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
    for j, (img, tgt_cls) in enumerate(dataloader):
        img = img.to(device)
        tgt_cls = tgt_cls.to(device)
        pred_cls = model(img)
        _, predictions = torch.max(pred_cls.data, 1)
        correct_counts = predictions.eq(tgt_cls.data.view_as(predictions))
        acc_cls = torch.mean(correct_counts.type(torch.FloatTensor))
        test_acc_cls += acc_cls.item() * img.size(0)
    acc_cls = test_acc_cls / test_len
    return acc_cls


if __name__=='__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-trd", "--train_root", default='/root/autodl-tmp/mu/train.json', help="train json path")
    ap.add_argument("-ted", "--test_root",  default='/root/autodl-tmp/mu/test.json', help="test json path")
    ap.add_argument("-b", "--batch_size", type=int, default=32, help="training batch size")
    ap.add_argument('-l', '--learning_rate', type=float, default=1e-3, help='Learning rate')
    ap.add_argument("-e", "--epochs", type=int, default=200, help="no. of epochs")
    ap.add_argument("-m", "--model", type=str, default="efficientnet_b0", help="model name (torchvision or timm)")
    ap.add_argument("-s","--seed", type=int, default=123, help="seed")
    args = vars(ap.parse_args())

    seed_all(args['seed'])
    train_root = args["train_root"]
    test_root  = args["test_root"]

    model = build_model(args['model'], num_classes=3, pretrained=True).to(device)
    summary(model, input_size=(3, 224, 224), batch_size=args["batch_size"])
    train(model, train_root=train_root, test_root=test_root,
          batch_size=args["batch_size"], lr=args["learning_rate"], epochs=args["epochs"])

