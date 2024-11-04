import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  # For progress tracking

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define a basic transform for your images
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load images from a folder
def load_images_from_folder(folder, transform, limit=None):
    images = []
    for i, filename in enumerate(os.listdir(folder)):
        if limit and i >= limit:  # Limit the number of images loaded
            break
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path)
        if transform:
            img = transform(img)
        images.append(img)
    return torch.stack(images)

# Folder containing real images
image_folder = 'C:/Users/Balaji/Documents/Project/Sample'
real_images = load_images_from_folder(image_folder, transform)

# Create a DataLoader
dataloader = torch.utils.data.DataLoader(real_images, batch_size=32, shuffle=True)

# Hyperparameters
latent_dim = 100
lr = 0.0002
beta1 = 0.5
beta2 = 0.999
num_epochs = 2000
num_generated_images = 1000
save_folder = 'C:/Users/Balaji/Documents/Project/Gan/acitinic keratosis'

# Define the generator
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4 * 4),  # Project and reshape latent dim into a feature map
            nn.ReLU(),
            nn.Unflatten(1, (256, 4, 4)),  # Start with 256 channels of size 4x4
            nn.Upsample(scale_factor=2),  # 4x4 -> 8x8
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # 8x8 -> 16x16
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # 16x16 -> 32x32
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # 32x32 -> 64x64
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # 64x64 -> 128x128
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Tanh()  # Output 3-channel RGB image (values between -1 and 1)
        )

    def forward(self, z):
        img = self.model(z)
        return img


# Define the discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 128x128 -> 64x64
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 64x64 -> 32x32
            nn.BatchNorm2d(64, momentum=0.82),
            nn.LeakyReLU(0.25),
            nn.Dropout(0.25),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 32x32 -> 16x16
            nn.BatchNorm2d(128, momentum=0.82),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 16x16 -> 8x8
            nn.BatchNorm2d(256, momentum=0.8),
            nn.LeakyReLU(0.25),
            nn.Dropout(0.25),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 8x8 -> 4x4
            nn.BatchNorm2d(512, momentum=0.8),
            nn.LeakyReLU(0.25),
            nn.Dropout(0.25),
        )
        self.fc = nn.Linear(512 * 4 * 4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.size(0), -1)  # Flatten the tensor
        validity = self.fc(out)
        return self.sigmoid(validity)


# Define the generator and discriminator
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

# Loss function
adversarial_loss = nn.BCELoss()

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))

# Ensure the save folder exists
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Training loop
for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
    for i, real_images in enumerate(dataloader):
        real_images = real_images.to(device)

        # Adversarial ground truths
        valid = torch.ones(real_images.size(0), 1, device=device)
        fake = torch.zeros(real_images.size(0), 1, device=device)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = torch.randn(real_images.size(0), latent_dim, device=device)
        fake_images = generator(z)

        # Discriminator losses
        real_loss = adversarial_loss(discriminator(real_images), valid)
        fake_loss = adversarial_loss(discriminator(fake_images.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()

        # Generator loss
        g_loss = adversarial_loss(discriminator(fake_images), valid)
        g_loss.backward()
        optimizer_G.step()

    # Print the losses every 100 epochs for monitoring
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}] - Generator Loss: {g_loss.item():.4f}, Discriminator Loss: {d_loss.item():.4f}")

# Generate and save images after 1000 epochs
with torch.no_grad():
    for i in range(num_generated_images // 16):
        z = torch.randn(16, latent_dim, device=device)
        generated_images = generator(z).detach().cpu()

        for j, image in enumerate(generated_images):
            img_name = os.path.join(save_folder, f"generated_{i*16 + j + 1}.png")
            torchvision.utils.save_image(image, img_name, normalize=True)

print(f"All {num_generated_images} images saved to {save_folder}.")