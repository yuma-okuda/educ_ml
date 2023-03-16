# import torch.nn as nn

# class MyModel(nn.Module):
#     def __init__(self, height, width):
#         super(MyModel, self).__init__()
#         self.layer1 = nn.Sequential(
#             nn.Linear(height * width, 128),
#             nn.ReLU(),
#         )
#         self.layer2 = nn.Sequential(
#             nn.Linear(128, 64),
#             nn.ReLU(),
#         )
#         self.layer3 = nn.Sequential(
#             nn.Linear(64, 32),
#             nn.ReLU(),
#         )
#         self.output = nn.Linear(32, 5)

#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.output(x)
#         return x
import torch.nn as nn
import torchvision

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel,self).__init__() 
        # Encoderの構築。
        # nn.Sequential内にはEncoder内で行う一連の処理を記載する。
        # create_convblockは複数回行う畳み込み処理をまとめた関数。
        # 畳み込み→畳み込み→プーリング→畳み込み・・・・のような動作
        self.Encoder = nn.Sequential(self.create_convblock(3,16),     #256
                                     nn.MaxPool2d((2,2)),
                                     self.create_convblock(16,32),    #128
                                     nn.MaxPool2d((2,2)),
                                     self.create_convblock(32,64),    #64
                                     nn.MaxPool2d((2,2)),
                                     self.create_convblock(64,128),   #32
                                     nn.MaxPool2d((2,2)),
                                     self.create_convblock(128,256),  #16
                                     nn.MaxPool2d((2,2)),
                                     self.create_convblock(256,512),  #8
                                    )
        # Decoderの構築。
        # nn.Sequential内にはDecoder内で行う一連の処理を記載する。
        # create_convblockは複数回行う畳み込み処理をまとめた関数。
        # deconvblockは逆畳み込みの一連の処理をまとめた関数
        # 逆畳み込み→畳み込み→畳み込み→逆畳み込み→畳み込み・・・・のような動作
        self.Decoder = nn.Sequential(self.create_deconvblock(512,256), #16
                                     self.create_convblock(256,256),
                                     self.create_deconvblock(256,128), #32
                                     self.create_convblock(128,128),
                                     self.create_deconvblock(128,64),  #64
                                     self.create_convblock(64,64),
                                     self.create_deconvblock(64,32),   #128
                                     self.create_convblock(32,32),
                                     self.create_deconvblock(32,16),   #256
                                     self.create_convblock(16,16),
                                    )
        # 出力前の調整用
        self.last_layer = nn.Conv2d(16,3,1,1)
                                        
    # 畳み込みブロックの中身                            
    def create_convblock(self,i_fn,o_fn):
        conv_block = nn.Sequential(nn.Conv2d(i_fn,o_fn,3,1,1),
                                   nn.BatchNorm2d(o_fn),
                                   nn.ReLU(),
                                   nn.Conv2d(o_fn,o_fn,3,1,1),
                                   nn.BatchNorm2d(o_fn),
                                   nn.ReLU()
                                  )
        return conv_block
    # 逆畳み込みブロックの中身
    def create_deconvblock(self,i_fn , o_fn):
        deconv_block = nn.Sequential(nn.ConvTranspose2d(i_fn, o_fn, kernel_size=2, stride=2),
                                      nn.BatchNorm2d(o_fn),
                                      nn.ReLU(),
                                     )
        return deconv_block

    # データの流れを定義     
    def forward(self,x):
        x = self.Encoder(x)
        x = self.Decoder(x)
        x = self.last_layer(x)           
        return x