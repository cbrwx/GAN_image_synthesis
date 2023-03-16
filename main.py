import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Set up the save checkpoint function
def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, f'{os.path.dirname(filename)}/best_model.pth')

# Set values
checkpoint_dir = 'your/checkpoint/'
model_dir = 'your/model/dir'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
image_size = 1024
batch_size = 32 # Adjust to fit your GPU(s) mem
num_epochs = 200
start_epoch = 0 # Checkpoint if

# Load checkpoint
if os.path.isfile(os.path.join(checkpoint_dir, 'checkpoint.pth')):
    print("=> Loading checkpoint")
    checkpoint = torch.load(os.path.join(checkpoint_dir, 'checkpoint.pth'))
    start_epoch = checkpoint['epoch']
    D.load_state_dict(checkpoint['D_state_dict'])
    G.load_state_dict(checkpoint['G_state_dict'])
    optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
    optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
    print(f"=> Loaded checkpoint (epoch {checkpoint['epoch']})")
else:
    print("=> No checkpoint found, starting from scratch")

# Set up the training data
data_transforms = transforms.Compose([
    transforms.Resize((image_size + 64, image_size + 64)),
    transforms.CenterCrop((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

dataset = datasets.ImageFolder(
    root='your/image/folder',
    transform=data_transforms
)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define Battle status bars
def print_battle_status(G_loss, D_loss, G_loss_history, D_loss_history, max_bar_length=30):
    G_loss_history.append(G_loss)
    D_loss_history.append(D_loss)

    G_bar_length = round((G_loss / (G_loss + D_loss)) * max_bar_length)  # Use round() function
    D_bar_length = max_bar_length - G_bar_length

    G_bar = "\033[1;42m" + " " * G_bar_length + "\033[0m"  # Green background for generator
    D_bar = "\033[1;41m" + " " * D_bar_length + "\033[0m"  # Red background for discriminator

    # Bold yellow text for the newest losses
    G_loss_text = f"\033[1;33m{G_loss:.4f}\033[0m"
    D_loss_text = f"\033[1;33m{D_loss:.4f}\033[0m"

    # Print the performance indicator with colored background and bold yellow text for the newest value
    print("")
    print("[Performance strength indicator]                  [Historical steps]")
    print(f"    Generator: [{G_bar:<{max_bar_length}}] {G_loss_text}", end="")
    print("".join([f"\033[38;5;{240}m {G_loss_history[-(i + 2):][0]:.4f}\033[0m" for i in range(min(5, len(G_loss_history) - 1))]))

    print(f"Discriminator: [{D_bar:<{max_bar_length}}] {D_loss_text}", end="")
    print("".join([f"\033[38;5;{240}m {D_loss_history[-(i + 2):][0]:.4f}\033[0m" for i in range(min(5, len(D_loss_history) - 1))]))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Set up the generator and discriminator models
class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, ndf, nc):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ndf * 16, 1, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# Set up the models and training parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set up the models and training parameters
nz = 100 # Size of latent vector
ngf = 64 # Size of feature maps in generator
ndf = 64 # Size of feature maps in discriminator
nc = 3 # Number of channels in the input images
lr = 0.0002 # Learning rate
beta1 = 0.5 # Beta1 for Adam optimizer

D = Discriminator(ndf, nc)
D.apply(weights_init)
if torch.cuda.device_count() > 1:
    D = nn.DataParallel(D)
D.to(device)

criterion = nn.BCELoss()
optimizerD = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))

G = Generator(nz, ngf, nc)
G.apply(weights_init)
if torch.cuda.device_count() > 1:
    G = nn.DataParallel(G)
G.to(device)

optimizerG = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))

# Set up the training data
transforms = transforms.Compose([transforms.Resize(image_size),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                                     
dataset = datasets.ImageFolder(root='your/image/folder', transform=transforms)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Train the GAN
fixed_noise = torch.randn(batch_size, nz, 1, 1, device=device)

accumulation_steps = 4  # Number of gradient accumulation steps
generator_accumulation_steps = 8

# Initialize lists to store the losses
G_losses = []
D_losses = []
G_loss_history = [] # Add this line to initialize G_loss_history
D_loss_history = [] # Add this line to initialize D_loss_history

for epoch in range(start_epoch, num_epochs):
    for i, (real_images, _) in enumerate(dataloader):
        real_images = real_images.to(device)

        # Train discriminator with real images
        D.zero_grad()
        real_labels = torch.full((real_images.size(0),), 1, device=device, dtype=torch.float)
        output = D(real_images).view(-1)
        errD_real = criterion(output, real_labels)
        errD_real.backward()
        D_x = output.mean().item()

        # Train discriminator with fake images
        noise = torch.randn(real_images.size(0), nz, 1, 1, device=device)
        fake_images = G(noise)
        fake_labels = torch.full((real_images.size(0),), 0, device=device, dtype=torch.float)
        output = D(fake_images.detach()).view(-1)
        errD_fake = criterion(output, fake_labels)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake

        if (i + 1) % accumulation_steps == 0:
            optimizerD.step()

        # Train generator
        G.zero_grad()
        fake_labels = torch.full((real_images.size(0),), 1, device=device, dtype=torch.float)
        output = D(fake_images).view(-1)
        errG = criterion(output, fake_labels)
        errG.backward()
        D_G_z2 = output.mean().item()

        if (i + 1) % generator_accumulation_steps  == 0:
            optimizerG.step()

        # Append the losses to their lists
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Print loss and save images
        if i % 1 == 0: # Decrease for more frequent updates to the print output
            clear_output(wait=True)  # Add this line to clear the output cell
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch+1, num_epochs, i+1, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            print_battle_status(errG.item(), errD.item(), G_loss_history, D_loss_history) # Add the missing arguments
            vutils.save_image(real_images, '%s/real_samples.png' % 'your/generated/real/image/dir', normalize=True)
            fake_images = G(fixed_noise)
            vutils.save_image(fake_images.detach(), '%s/fake_samples_epoch_%03d.png' % ('your/fake/generated/image/dir', epoch+1), normalize=True)

    # Save a checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        save_checkpoint({
            'epoch': epoch + 1,
            'D_state_dict': D.state_dict(),
            'G_state_dict': G.state_dict(),
            'optimizerD_state_dict': optimizerD.state_dict(),
            'optimizerG_state_dict': optimizerG.state_dict(),
        }, False, os.path.join(checkpoint_dir, 'checkpoint.pth'))

# Plot the losses
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Losses")
plt.plot(G_losses, label="Generator Loss")
plt.plot(D_losses, label="Discriminator Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
