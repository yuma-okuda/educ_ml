import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn

import wandb

from models import Net


def main():
    wandb.init(project="educ_ml")
    # 訓練データを取得
    mnist_train = MNIST("./mydata",
                        train=True, download=True,
                        transform=transforms.ToTensor())
    # テストデータの取得
    mnist_test = MNIST("./mydata",  # 保存先フォルダの指定
                       train=False, download=True,
                       transform=transforms.ToTensor())
    print("訓練データの数:", len(mnist_train), "テストデータの数:", len(mnist_test))

    # DataLoaderの設定
    batch_size = 256
    train_loader = DataLoader(mnist_train,
                              batch_size=batch_size,
                              shuffle=True)
    test_loader = DataLoader(mnist_test,
                             batch_size=len(mnist_test),
                             shuffle=False)

    dev = torch.device("cuda:0" if torch.cuda.is_available()
                       else "cpu")  # GPUが使えたらGPU、使えなかったらCPU
    net = Net()
    net = net.to(dev)  # GPU対応

    # 交差エントロピー誤差関数
    loss_fnc = nn.CrossEntropyLoss()

    lr = 0.01
    epochs = 10
    # SGD
    optimizer = optim.SGD(net.parameters(), lr=lr)
    wandb.config.lr = lr
    # 損失のログ
    record_loss_train = []
    record_loss_test = []

    for i in range(epochs):  # 10エポック学習
        net.train()  # 訓練モードへ
        loss_train = 0
        for j, (x, t) in enumerate(train_loader):  # ミニバッチ(x,t)を取り出す
            x, t = x.to(dev), t.to(dev)  # GPUのメモリに配置する
            y = net(x)
            loss = loss_fnc(y, t)
            loss_train += loss.item()  # ミニバッチなので、誤差を蓄積させていく
            optimizer.zero_grad()  # RNNではためることもあるが普通は初めに勾配を0初期化必須
            loss.backward()  # 逆伝播してパラメタを計算
            optimizer.step()  # 計算した値でパラメタを更新
        loss_train /= j+1  # ループから抜けたらロスの平均を計算
        record_loss_train.append(loss_train)
        print("[Epoch %d/%d] [Batch %d/%d] [loss: %f]"
              % (i, epochs, j+1, len(train_loader), loss.item())
              )
        wandb.log({'train_loss': loss})

    correct = 0
    total = 0
    net.eval()  # 評価モードへ
    for i, (x, t) in enumerate(test_loader):
        x, t = x.to(dev), t.to(dev)   # GPU対応
        x = x.view(-1, 784)
    with torch.no_grad():
        y = net(x)
        test_loss = loss_fnc(y, t)
    correct += (y.argmax(1) == t).sum().item()
    total += len(x)
    print("正解率:", str(correct/total*100) + "%")
    wandb.log({'test_loss': test_loss})
    wandb.save('models.py')


if __name__ == '__main__':
    main()
