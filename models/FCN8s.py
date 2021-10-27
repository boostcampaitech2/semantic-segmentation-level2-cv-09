import torch
import torch.nn as nn

class FCN8s(nn.Module):
    def __init__(self, backbone, num_classes=11):
        super(FCN8s, self).__init__()

        # conv1
        self.conv1_1 = self.CBR(3, 64, 3, 1, 1)
        self.conv1_2 = self.CBR(64, 64, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, stride=2 , ceil_mode=True) # size 1/2

        # conv2
        self.conv2_1 = self.CBR(64, 128, 3, 1, 1)
        self.conv2_2 = self.CBR(128, 128, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True) # size 1/4

        # conv3
        self.conv3_1 = self.CBR(128, 256, 3, 1, 1)
        self.conv3_2 = self.CBR(256, 256, 3, 1, 1)
        self.conv3_3 = self.CBR(256, 256, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # Score pool3
        self.score_pool3_fr = nn.Conv2d(256,
                                        num_classes, 
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
        
        # conv4
        self.conv4_1 = self.CBR(256, 512, 3, 1, 1)
        self.conv4_2 = self.CBR(512, 512, 3, 1, 1)
        self.conv4_3 = self.CBR(512, 512, 3, 1, 1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # Score pool4
        self.score_pool4_fr = nn.Conv2d(512,
                                        num_classes, 
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

        # conv5
        self.conv5_1 = self.CBR(512, 512, 3, 1, 1)
        self.conv5_2 = self.CBR(512, 512, 3, 1, 1)
        self.conv5_3 = self.CBR(512, 512, 3, 1, 1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 1)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        # Score
        self.score_fr = nn.Conv2d(4096, num_classes, kernel_size = 1)

        # UpScore2
        self.upscore2 = nn.ConvTranspose2d(num_classes,
                                           num_classes,
                                           kernel_size=4,
                                           stride=2,
                                           padding=1)

        # UpScore2_pool4
        self.upscore2_pool4 = nn.ConvTranspose2d(num_classes, 
                                                 num_classes, 
                                                 kernel_size=4,
                                                 stride=2,
                                                 padding=1)

        # UpScore8
        self.upscore8 = nn.ConvTranspose2d(num_classes, 
                                           num_classes,
                                           kernel_size=16,
                                           stride=8,
                                           padding=4)

    def CBR(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        return nn.Sequential(nn.Conv2d(in_channels=in_channels, 
                                        out_channels=out_channels, 
                                        kernel_size=kernel_size, 
                                        stride=stride,
                                        padding=padding),
                            nn.ReLU(inplace=True)
                            )
    
    def forward(self, x):
        h = self.conv1_1(x)
        h = self.conv1_2(h)
        h = self.pool1(h)

        h = self.conv2_1(h)
        h = self.conv2_2(h)
        h = self.pool2(h)

        h = self.conv3_1(h)
        h = self.conv3_2(h)
        h = self.conv3_3(h)
        h = self.pool3(h)
        pool3 = h

        score_pool3c = self.score_pool3_fr(pool3)

        h = self.conv4_1(h)
        h = self.conv4_2(h)
        h = self.conv4_3(h)
        h = self.pool4(h)
        pool4 = h

        score_pool4c = self.score_pool4_fr(pool4)

        h = self.conv5_1(h)
        h = self.conv5_2(h)
        h = self.conv5_3(h)
        h = self.pool5(h)

        h = self.fc6(h)
        h = self.drop6(h)

        h = self.fc7(h)
        h = self.drop7(h)

        h = self.score_fr(h)

        upscore2 = self.upscore2(h)

        h = upscore2 + score_pool4c

        upscore2_pool4 = self.upscore2_pool4(h)

        h = upscore2_pool4 + score_pool3c

        upscore8 = self.upscore8(h)

        return upscore8