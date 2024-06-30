import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.vision.transforms as transforms
from paddle.io import DataLoader
import numpy as np
import random

# Data Augmentation
class AutoTransforms(transforms.BaseTransform):
    def __init__(self, transforms=None, keys=None):
        super(AutoTransforms, self).__init__(keys)
        self.transforms = transforms

    def _apply_image(self, image):
        if self.transforms is None:
            return image
        choose = np.random.randint(0, len(self.transforms))
        return self.transforms[choose](image)

mean = [0.5071, 0.4867, 0.4408]
std = [0.2675, 0.2565, 0.2761]
transforms_list = [
    transforms.BrightnessTransform(0.5),
    transforms.SaturationTransform(0.5),
    transforms.ContrastTransform(0.5),
    transforms.HueTransform(0.5),
    transforms.RandomRotation(15, expand=True, fill=128),
    transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
    transforms.Grayscale(3),
]

train_tx = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    AutoTransforms(transforms_list),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomVerticalFlip(),
    transforms.Transpose(),
    transforms.Normalize(0.0, 255.0),
    transforms.Normalize(mean, std),
])

val_tx = transforms.Compose([
    transforms.Transpose(),
    transforms.Normalize(0.0, 255.0),
    transforms.Normalize(mean, std),
])

trainset = paddle.vision.datasets.Cifar100(mode='train', transform=train_tx)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = paddle.vision.datasets.Cifar100(mode='test', transform=val_tx)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Define Vision Transformer Model
class PatchEmbedding(nn.Layer):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=64):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2D(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, embed_dim, H/P, W/P]
        x = x.flatten(2)  # [B, embed_dim, N]
        x = x.transpose([0, 2, 1])  # [B, N, embed_dim]
        return x

class ViT(nn.Layer):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=100, embed_dim=64, depth=4, num_heads=4, mlp_ratio=4., dropout=0.1, attn_dropout=0.1):
        super(ViT, self).__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = self.create_parameter(shape=[1, 1, embed_dim], default_initializer=nn.initializer.Constant(0.))
        self.pos_embed = self.create_parameter(shape=[1, (img_size // patch_size) ** 2 + 1, embed_dim], default_initializer=nn.initializer.TruncatedNormal(std=0.02))
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.LayerList([
            nn.TransformerEncoderLayer(embed_dim, num_heads, int(embed_dim * mlp_ratio), dropout, attn_dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embedding(x)  # [B, N, embed_dim]
        cls_tokens = self.cls_token.expand([B, -1, -1])  # [B, 1, embed_dim]
        x = paddle.concat((cls_tokens, x), axis=1)  # [B, N+1, embed_dim]
        x = x + self.pos_embed
        x = self.dropout(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        cls_token_final = x[:, 0]  # [B, embed_dim]
        x = self.head(cls_token_final)  # [B, num_classes]
        return x

# Training and Evaluation Functions
def train(model, trainloader, criterion, optimizer, epoch):
    model.train()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(inputs)}/{len(trainloader.dataset)}]\tLoss: {loss.numpy()[0]:.6f}')

def test(model, testloader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with paddle.no_grad():
        for inputs, targets in testloader:
            outputs = model(inputs)
            test_loss += criterion(outputs, targets).numpy()[0]
            pred = outputs.argmax(axis=1)
            correct += (pred == targets).astype(paddle.int32).sum().numpy()[0]
    test_loss /= len(testloader.dataset)
    accuracy = 100. * correct / len(testloader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(testloader.dataset)} ({accuracy:.0f}%)\n')

# Main Function
def main():
    paddle.seed(42)
    model = ViT()
    criterion = nn.CrossEntropyLoss()
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=0.001)
    epochs = 10

    for epoch in range(1, epochs + 1):
        train(model, trainloader, criterion, optimizer, epoch)
        test(model, testloader, criterion)

if __name__ == '__main__':
    main()
