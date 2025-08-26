import torch 
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import os
import argparse
import json
from dataset import Clsdata
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score, auc, roc_curve
from sklearn.preprocessing import label_binarize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch
import random


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(model, test_root,  batch_size):
    
    with open(test_root, 'r') as fr:
         test_imgs = json.load(fr)
    test_data  = Clsdata(test_imgs, train=False, shuffle=False, test=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4)
    test_length = len(test_data)
    # Set to evaluation mode
    model.eval()
    test_acc_cls=0.0
    y_true=[]
    y_pred=[]

    # Validation loop
    for i, (img, tgt_cls) in enumerate(test_dataloader):
        img = img.to(device)
        tgt_cls = tgt_cls.to(device)
        y_true.extend(tgt_cls.cpu().numpy().tolist())

        # Forward pass - compute outputs on input data using the model
        pred_cls = model(img)

        pred, predictions = torch.max(pred_cls.data, 1)
        y_pred.extend(predictions.cpu().numpy().tolist())
        correct_counts = predictions.eq(tgt_cls.data.view_as(predictions))
        # Convert correct_counts to float and then compute the mean
        acc_cls = torch.mean(correct_counts.type(torch.FloatTensor))
        # Compute total accuracy in the whole batch and add to valid_acc
        test_acc_cls += acc_cls.item() * img.size(0)


    Acc = test_acc_cls / test_length
    #accuracy = accuracy_score(y_true, y_pred)
    Pre = precision_score(y_true, y_pred, average='macro')
    Rec = recall_score(y_true, y_pred, average='macro')
    #F1 = f1_score(y_true, y_pred, average='macro')
    F1 = (2*Pre*Rec)/(Pre+Rec)



    return Acc, Pre, Rec, F1


if __name__=='__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-ted", "--test_root", default = '/root/autodl-tmp/mu/test.json',
		help = "test image path")
    ap.add_argument("-b", "--batch_size", type=int, default=32,
		help="training batch size")
    args = vars(ap.parse_args())

    random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    test_root = args["test_root"]


    model_dir = "/root/autodl-tmp/mu"  
    model_path = os.path.join(model_dir, "alexnet.pth")
    model1 = torch.load(model_path, weights_only=False)
    model_path2 = os.path.join(model_dir, "vgg16.pth")
    model2 = torch.load(model_path2, weights_only=False)
    model_path3 = os.path.join(model_dir, "vgg19.pth")
    model3 = torch.load(model_path3, weights_only=False)
    model_path4 = os.path.join(model_dir, "resnet18.pth")
    model4= torch.load(model_path4, weights_only=False)
    model_path5 = os.path.join(model_dir, "resnet34.pth")
    model5 = torch.load(model_path5, weights_only=False)
    model_path6 = os.path.join(model_dir, "resnet50.pth")
    model6 = torch.load(model_path6, weights_only=False)
    model_path7 = os.path.join(model_dir, "densenet121.pth")
    model7 = torch.load(model_path7, weights_only=False)
    model_path8 = os.path.join(model_dir, "densenet169.pth")
    model8= torch.load(model_path8, weights_only=False)
    model_path9 = os.path.join(model_dir, "mobilenet_v2.pth")
    model9 = torch.load(model_path9, weights_only=False)
    model_path10 = os.path.join(model_dir, "mobilenet_v3_small.pth")
    model10 = torch.load(model_path10, weights_only=False)
    model_path11 = os.path.join(model_dir, "efficientnet_b0.pth")
    model11= torch.load(model_path11, weights_only=False)
    model_path12 = os.path.join(model_dir, "efficientnet_b1.pth")
    model12 = torch.load(model_path12, weights_only=False)
    model_path13 = os.path.join(model_dir, "vit_tiny_patch16_224.pth")
    model13 = torch.load(model_path13, weights_only=False)
    model_path14 = os.path.join(model_dir, "vit_small_patch16_224.pth")
    model14= torch.load(model_path14, weights_only=False)
    model_path15 = os.path.join(model_dir, "vit_base_patch16_224.pth")
    model15 = torch.load(model_path15, weights_only=False)
    model_path16 = os.path.join(model_dir, "vit_large_patch16_224.pth")
    model16 = torch.load(model_path16, weights_only=False)
    model_path17 = os.path.join(model_dir, "deit_tiny_patch16_224.pth")
    model17 = torch.load(model_path17, weights_only=False)
    model_path18 = os.path.join(model_dir, "deit_small_patch16_224.pth")
    model18= torch.load(model_path18, weights_only=False)
    model_path19 = os.path.join(model_dir, "deit_base_patch16_224.pth")
    model19 = torch.load(model_path19, weights_only=False)
    model_path20 = os.path.join(model_dir, "efficientformer_l1.pth")
    model20 = torch.load(model_path20, weights_only=False)
    model_path21 = os.path.join(model_dir, "efficientformer_l3.pth")
    model21= torch.load(model_path21, weights_only=False)

    
    # Test the model
    models_and_names = [
        ("alexnet",                model1),
        ("vgg16",                  model2),
        ("vgg19",                  model3),
        ("resnet18",               model4),
        ("resnet34",               model5),
        ("resnet50",               model6),
        ("densenet121",            model7),
        ("densenet169",            model8),
        ("mobilenet_v2",           model9),
        ("mobilenet_v3_small",     model10),
        ("efficientnet_b0",        model11),
        ("efficientnet_b1",        model12),
        ("vit_tiny_patch16_224",   model13),
        ("vit_small_patch16_224",  model14),
        ("vit_base_patch16_224",   model15),
        ("vit_large_patch16_224", model16),  
        ("deit_tiny_patch16_224",  model17),
        ("deit_small_patch16_224", model18),
        ("deit_base_patch16_224",  model19),
        ("efficientformer_l1",     model20),
        ("efficientformer_l3",     model21),
    ]

    # 逐个评测并打印
    for name, mdl in models_and_names:
        mdl = mdl.to(device)
        Acc, Pre, Rec, F1 = test(mdl, test_root=test_root, batch_size=args["batch_size"])
        print(f"{name:>24s}: Acc {Acc*100:6.2f}%, Pre {Pre*100:6.2f}%, Rec {Rec*100:6.2f}%, F1 {F1*100:6.2f}%")





