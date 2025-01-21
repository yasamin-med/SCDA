import os
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
from math import floor
import argparse



parser = argparse.ArgumentParser(description='Train Dataset on DCGAN',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', type=str,
                    default='./data/', help='Path of dataset for train ')
parser.add_argument('--dest_path', type=str,
                    default='./data/', help='Path of dataset for test')

args = parser.parse_args()
print(args)


def zero():
    return np.random.uniform(0.0, 0.01, size = [1])

def one():
    return np.random.uniform(0.99, 1.0, size = [1])

def noise(n):
    return np.random.uniform(-1.0, 1.0, size = [n, 4096])

os.makedirs(args.dest_path, exist_ok=True)
os.makedirs(os.path.join(args.dest_path , "Models"), exist_ok=True)
os.makedirs(os.path.join(args.dest_path , "Results"), exist_ok=True)
print("Importing Images...")

Images = []
images_path = args.data_path #"/home/yasamin/Documents/Data_breast/Data_breast_splits/split_1/train/benign"
#os.getcwd() + '\images'
files = os.listdir(images_path)
for name in os.listdir(images_path):
    path_img = os.path.join(images_path , name)
    temp1 = Image.open(path_img)
    temp1 = temp1.resize((256 , 256))           
    temp = np.array(temp1.convert('L'), dtype='float32')
    
    Images.append(temp / 255)
    
    Images.append(np.flip(Images[-1], 1))



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import os
from torchvision import transforms

# Helper functions for noise and labels
def zero():
    return torch.rand(1) * 0.01

def one():
    return torch.rand(1) * 0.01 + 0.99

def noise(n):
    return torch.randn(n, 4096)

# GAN class implementation in PyTorch
class GAN(nn.Module):
    def __init__(self):
        super(GAN, self).__init__()
        
        # Config
        self.LR = 0.0001
        self.steps = 1

        # Define the models
        self.D = self.discriminator()
        self.G = self.generator()

        # Define optimizers
        self.optimizer_D = optim.Adam(self.D.parameters(), lr=self.LR)
        self.optimizer_G = optim.Adam(self.G.parameters(), lr=self.LR)

    def discriminator(self):
        model = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding = 'same'),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(8, 16, kernel_size=3, padding = 'same'),
            nn.BatchNorm2d(16, momentum=0.7),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding = 'same'),
            nn.BatchNorm2d(32, momentum=0.7),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding = 'same'),
            nn.BatchNorm2d(64, momentum=0.7),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding = 'same'),
            nn.BatchNorm2d(128, momentum=0.7),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding = 'same'),
            nn.BatchNorm2d(256, momentum=0.7),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            nn.AvgPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(4 * 4 * 256, 128),  # Update input features here to match output of conv layers
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        return model





    def generator(self):
        
        
        model = nn.Sequential(
            # 1x1x4096
            nn.ConvTranspose2d(4096, 256, kernel_size=4),
            nn.ReLU(),

            # 4x4x256
            nn.Conv2d(256, 256, kernel_size=4, padding='same'),
            nn.BatchNorm2d(256, momentum=0.7),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            # 8x8x256
            nn.Conv2d(256, 128, kernel_size=4, padding='same'),
            nn.BatchNorm2d(128, momentum=0.7),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            # 16x16x128
            nn.Conv2d(128, 64, kernel_size=3, padding='same'),
            nn.BatchNorm2d(64, momentum=0.7),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            # 32x32x64
            nn.Conv2d(64, 32, kernel_size=3, padding='same'),
            nn.BatchNorm2d(32, momentum=0.7),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            # 64x64x32
            nn.Conv2d(32, 16, kernel_size=3, padding='same'),
            nn.BatchNorm2d(16, momentum=0.7),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            # 128x128x16
            nn.Conv2d(16, 8, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            # 256x256x8
            nn.Conv2d(8, 1, kernel_size=3, padding='same'),
            nn.Sigmoid()
        )
        return model
    
    def forward_G(self, x):
        x = x.view(x.size(0), 4096, 1, 1)  # Reshape to [batch_size, 4096, 1, 1]
        return self.G(x)


    def forward_D(self, x):
        return self.D(x)


    def train_discriminator(self, real_data, fake_data):
        self.optimizer_D.zero_grad()

        # Train on real data
        real_pred = self.forward_D(real_data)
        real_loss = F.binary_cross_entropy(real_pred, torch.zeros_like(real_pred))
        real_loss.backward()

        # Train on fake data
        fake_pred = self.forward_D(fake_data)
        fake_loss = F.binary_cross_entropy(fake_pred, torch.ones_like(fake_pred))
        fake_loss.backward()

        self.optimizer_D.step()
        return real_loss.item(), fake_loss.item()

    def train_generator(self, batch_size):
        self.optimizer_G.zero_grad()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        z = noise(batch_size).to(device)
        fake_data = self.forward_G(z)

        fake_pred = self.forward_D(fake_data)
        loss = F.binary_cross_entropy(fake_pred, torch.zeros_like(fake_pred))
        loss.backward()

        self.optimizer_G.step()
        return loss.item()

# Model GAN implementation in PyTorch
class Model_GAN:
    def __init__(self):
        self.GAN = GAN()

    def train(self, images, batch_size=16, device='cpu'):
        self.GAN.D.to(device)  # Ensure the Discriminator is on the correct device
        self.GAN.G.to(device)  # Ensure the Generator is on the correct device
        
        #for i in range(500000):  # Adjust the range as needed
        im_no = random.randint(0, len(images) - batch_size - 1)
        
        # Convert the selected images to tensors, ensuring any negative strides are removed
        real_data = torch.stack([transforms.ToTensor()(np.copy(img)) for img in images[im_no:im_no + int(batch_size / 2)]])
        real_data = real_data.to(device)  # Move data to the device
        
        z = noise(int(batch_size / 2)).to(device)  # Move noise to the device
        fake_data = self.GAN.forward_G(z)

        d_real, d_fake = self.GAN.train_discriminator(real_data, fake_data)
        g_loss = self.GAN.train_generator(batch_size)
        
        print(f"D Real: {d_real}, D Fake: {d_fake}, G All: {g_loss}")

        if self.GAN.steps % 500 == 0:
            self.save(floor(self.GAN.steps / 1000))
            self.evaluate(device)
        self.GAN.steps += 1


    def evaluate(self, device='cpu'):
        self.GAN.G.to(device)  # Ensure the Generator is on the correct device
        with torch.no_grad():
            z = noise(48).to(device)  # Move noise to the device
            generated_images = self.GAN.forward_G(z).cpu()
            # Visualization logic goes here using matplotlib or any other tool
            # e.g., plt.imshow(generated_images[0].permute(1, 2, 0)) to show the first image
    def save(self, num):
        # Save the generator and discriminator state dictionaries
        torch.save(self.GAN.G.state_dict(), os.path.join(args.dest_path , "Models/gen.pth"))
        torch.save(self.GAN.D.state_dict(),  os.path.join(args.dest_path ,"Models/dis.pth"))
        print(f"Model number {str(num)} Saved!")
    
    def load(self, num):
        steps1 = self.GAN.steps
        
        # Reinitialize the GAN model
        self.GAN = GAN()

        # Load the state dictionaries into the models
        self.GAN.G.load_state_dict(torch.load(os.path.join(args.dest_path , "Models/gen.pth")))
        self.GAN.D.load_state_dict(torch.load(os.path.join(args.dest_path ,"Models/dis.pth")))

        # Ensure models are set to evaluation mode (if needed)
        self.GAN.G.eval()
        self.GAN.D.eval()

        # Reinitialize the necessary components
        self.generator = self.GAN.generator()
        self.DisModel = self.GAN.DisModel()
        self.AdModel = self.GAN.AdModel()
        
        # Restore the step count
        self.GAN.steps = steps1

    
    def eval2(self, num=0, device='cpu'):
        self.GAN.G.to(device)  # Ensure the Generator is on the correct device
        with torch.no_grad():
            z = noise(48).to(device)  # Move noise to the device
            generated_images = self.GAN.forward_G(z).cpu()  # Move the generated images to the CPU for saving
            generated_images = np.squeeze(generated_images)
            print(generated_images.shape)
            generated_images = generated_images.permute(0, 1, 2).numpy()
            print(generated_images.shape)
            r1 = np.concatenate(generated_images[:8], axis=1)
            r2 = np.concatenate(generated_images[8:16], axis=1)
            r3 = np.concatenate(generated_images[16:24], axis=1)
            r4 = np.concatenate(generated_images[24:32], axis=1)
            r5 = np.concatenate(generated_images[32:40], axis=1)
            r6 = np.concatenate(generated_images[40:48], axis=1)
            c1 = np.concatenate([r1, r2, r3, r4, r5, r6], axis=0)
            image = (c1 * 255).astype(np.uint8)
            print(image.shape)
            print(type(image))
            Image.fromarray(image , mode = 'L').save(os.path.join( args.dest_path , "Results/i.png"))




# Check if CUDA is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# If training a new model:
model = Model_GAN()
model.GAN.D.to(device)
model.GAN.G.to(device)
# Print model summaries
print("Discriminator Summary:")
print(model.GAN.D)
print("\nGenerator Summary:")
print(model.GAN.G)

print("We're off! See you in a while!")

# Set the initial step count if needed
# model.GAN.steps = 165001

# Assuming 'Images' is a preloaded dataset of images for training
while model.GAN.steps < 500000:
    
    # Train the model
    model.train(Images, device=device)  # Pass the device to the train method
    
    if model.GAN.steps % 500 == 0:
        print("\nRound: " + str(model.GAN.steps))
    
    if model.GAN.steps % 1000 == 0:
        print("\n\n\n\nRound " + str(model.GAN.steps) + ":")
        model.eval2(int(model.GAN.steps / 1000), device=device)  # Pass the device to eval2
