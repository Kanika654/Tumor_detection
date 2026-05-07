import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Generator(nn.Module):
    def __init__(self, z_dim=100, num_classes=2, img_channels=4, img_size=128):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.img_channels = img_channels
        self.init_size = img_size // 4
        
        # Embed condition
        self.label_emb = nn.Embedding(num_classes, num_classes)
        
        self.l1 = nn.Sequential(nn.Linear(z_dim + num_classes, 128 * self.init_size ** 2))
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, img_channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        gen_input = torch.cat((self.label_emb(labels.long()), noise), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self, num_classes=2, img_channels=4, img_size=128):
        super(Discriminator, self).__init__()
        
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25)
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters))
            return block

        self.model = nn.Sequential(
            *discriminator_block(img_channels + num_classes, 32, bn=False),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
        )

        downsized_size = img_size // (2 ** 4)
        self.adv_layer = nn.Sequential(
            nn.Linear(256 * downsized_size ** 2, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # Broadcast label embedding to image spatial dimensions
        c = self.label_embedding(labels.long())
        c = c.view(c.shape[0], c.shape[1], 1, 1).expand(-1, -1, img.shape[2], img.shape[3])
        d_in = torch.cat((img, c), 1)
        out = self.model(d_in)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity

class CNNClassifier(nn.Module):
    def __init__(self, img_channels=4, img_size=128):
        super(CNNClassifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(img_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 64x64
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 32x32
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 16x16
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 8x8
        )
        downsized = img_size // (2 ** 4)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * downsized * downsized, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

class ResNetClassifier(nn.Module):
    def __init__(self, img_channels=4, img_size=128):
        super(ResNetClassifier, self).__init__()
        # Load a pretrained resnet18
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Modify the first conv layer to accept 'img_channels' instead of 3
        old_conv1 = self.model.conv1
        self.model.conv1 = nn.Conv2d(img_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.model.conv1.weight[:, :3, :, :] = old_conv1.weight
            if img_channels > 3:
                self.model.conv1.weight[:, 3:, :, :] = old_conv1.weight.mean(dim=1, keepdim=True)
        # Modify the fully connected layer for binary classification
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class ViTClassifier(nn.Module):
    def __init__(self, img_channels=4, img_size=128):
        super(ViTClassifier, self).__init__()
        # Load a pretrained vit_b_16
        self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        
        # Modify the patch embedding layer (conv_proj) to accept 'img_channels'
        old_conv = self.model.conv_proj
        self.model.conv_proj = nn.Conv2d(img_channels, old_conv.out_channels, 
                                         kernel_size=old_conv.kernel_size, 
                                         stride=old_conv.stride, 
                                         padding=old_conv.padding)
        with torch.no_grad():
            self.model.conv_proj.weight[:, :3, :, :] = old_conv.weight
            if img_channels > 3:
                self.model.conv_proj.weight[:, 3:, :, :] = old_conv.weight.mean(dim=1, keepdim=True)
                
        # Modify the fully connected head for binary classification
        num_ftrs = self.model.heads.head.in_features
        self.model.heads.head = nn.Sequential(
            nn.Linear(num_ftrs, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # vit_b_16 expects 224x224 inputs for its positional embeddings to match
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.model(x)
