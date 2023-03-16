import argparse
import cv2
import glob
import numpy as np
import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T

import torchvision


from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt

from models import MyModel
from utils import seed_everything, MyDataset


epochs = 200
b_size = 64
lr = 0.0001
dev = torch.device("cuda:0" if torch.cuda.is_available()
                   else "cpu")  # GPUが使えたらGPU、使えなかったらCPU


# データセットのありか
datapath = '/auto/proj/proj/okuda/workspace/educ/capsule'
# 学習用正常データの読み出し
good_list = glob.glob(os.path.join(datapath, 'train/good', '*'))

# 評価用正常データの読み出し
good_test_list = glob.glob(os.path.join(datapath, "test/good/", '*'))

# 評価用異常データの読み出し
bad_test_list = glob.glob(os.path.join(datapath, "test/crack", '*')) + glob.glob(os.path.join(datapath, "test/faulty_imprint", '*')) + glob.glob(
    os.path.join(datapath, "test/poke", '*')) + glob.glob(os.path.join(datapath, "test/scratch", '*')) + glob.glob(os.path.join(datapath, "test/squeeze", '*'))

train_list = good_list
test_list = []
test_list.extend(good_test_list)
test_list.extend(bad_test_list)

train_dataset = MyDataset(train_list)
test_dataset = MyDataset(test_list)
train_loader = DataLoader(train_dataset, batch_size=b_size)
test_loader = DataLoader(test_dataset, batch_size=1)

model = MyModel().to(dev)
criterion = nn.MSELoss()


def train():
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for i, train_data in enumerate(train_loader):
            train_data = train_data.to(dev)
            optimizer.zero_grad()
            train_output = model(train_data)
            loss = criterion(train_output, train_data)
            loss.backward()
            optimizer.step()
            print("[Epoch %d/%d] [Batch %d/%d] [loss: %f]"
                  % (epoch, epochs, i, len(train_loader), loss.item())
                  )
    torch.save(model.state_dict(), '../result/model.pth')


def test():
    margin_w = 10
    prepocess = T.Compose([T.Resize((128, 128)),
                           T.ToTensor(),
                           ])
    model.eval()
    loss_list = []

    labels = [0]*len(good_test_list) + [1]*len(bad_test_list)
    for idx, path in enumerate(test_list):
        img = Image.open(path)
        _img = img
        img = prepocess(img).unsqueeze(0).to(dev)
        with torch.no_grad():
            output = model(img)[0]
        print(output)
        _output = torchvision.transforms.functional.to_pil_image(output.cpu())
        _output.save('result.jpg')
        image = output.squeeze(0)
        image = T.ToPILImage(image)
        output = output.cpu().numpy().transpose(1, 2, 0)
        output = np.uint8(np.maximum(np.minimum(output*255, 255), 0))
        origin = np.uint8(img[0].cpu().numpy().transpose(1, 2, 0)*255)

        diff = np.uint8(np.abs(output.astype(
            np.float32) - origin.astype(np.float32)))
        loss_list.append(np.sum(diff))
        heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
        margin = np.ones((diff.shape[0], margin_w, 3))*255

        result = np.concatenate(
            [origin[:, :, ::-1], margin, output[:, :, ::-1], margin, heatmap], axis=1)
        # result = np.concatenate(
        #     [_img, margin, image, margin, heatmap], axis=1)
        label = 'good' if idx < len(good_test_list) else 'bad'
        cv2.imwrite(f"../result/{idx}_{label}.jpg", result)
    fpr, tpr, thresholds = roc_curve(labels, loss_list)

    plt.plot(fpr, tpr, marker='o')

    plt.xlabel('FPR: False positive rate')
    plt.ylabel('TPR: True positive rate')
    plt.grid()
    plt.savefig('./sklearn_roc_curve.png')
    print(roc_auc_score(labels, loss_list))


if __name__ == "__main__":
    seed_everything()
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    x = input("enter to use exiting model: ")
    if x == "":
        if os.path.isfile('../result/model.pth'):
            model.load_state_dict(torch.load('../result/model.pth'))
        else:
            train()
    else:
        train()
    test()
