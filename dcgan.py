import urllib.request
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import mindspore as ms
import mindspore.dataset as ds
import numpy as np
from mindspore import nn
from mindspore import ops
from mindspore import save_checkpoint, Tensor
from mindspore.common import initializer as init
from mindspore.dataset import vision
from mindspore.nn import Adam
from mindspore.nn import BCELoss
from tqdm import tqdm

URL = "https://download.mindspore.cn/dataset/Faces/faces.zip"
DATA_DIR = Path("./datasets/faces")
ZIP_PATH = DATA_DIR.with_suffix('.zip')


# Functions
def download_dataset(url, path):
    if not path.exists():
        print("Downloading faces dataset...")
        urllib.request.urlretrieve(url, path)
        print("Download complete.")
    else:
        print("Dataset zip file already exists.")


def extract_dataset(zip_path, extract_to):
    if not extract_to.exists():
        print("Extracting files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to.parent)
        print("Extraction complete.")
    else:
        print("Dataset already extracted.")


def check_data_directory(data_dir):
    if data_dir.exists():
        files = list(data_dir.iterdir())
        print(f"Files in data directory: {files}")
        if not files:
            print("Warning: The data directory is empty!")
    else:
        print("Error: The data directory does not exist!")


# Constants
DATA_ROOT = "./datasets/faces"
BATCH_SIZE = 128
IMAGE_SIZE = 64
NC = 3  # Color channel
NZ = 100  # Length of the latent vector
NGF = 64  # Feature map size in the generator
NDF = 64  # Feature map size in the discriminator
NUM_EPOCHS = 10
LR = 0.0002
BETA1 = 0.5

ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")  # Set the context for the MindSpore session


def create_dataset_imagenet(dataset_path, image_size, nz, batch_size):
    """Load and preprocess the dataset."""
    # Load the dataset with basic configurations
    data_set = ds.ImageFolderDataset(
        dataset_path, num_parallel_workers=4, shuffle=True, decode=True)

    # Transformation pipeline
    transform_img = [
        vision.Resize(image_size),
        vision.CenterCrop(image_size),
        vision.HWC2CHW(),
        lambda x: ((x / 255).astype("float32"), np.random.normal(size=(nz, 1, 1)).astype("float32"))
    ]

    # Apply transformations, project columns, and batch the dataset in one flow
    return data_set.map(
        input_columns="image", operations=transform_img, num_parallel_workers=4, output_columns=["image", "latent_code"]
    ).project(["image", "latent_code"]).batch(batch_size)


# Create the dataset with the specified processing
data = create_dataset_imagenet(DATA_ROOT, IMAGE_SIZE, NZ, BATCH_SIZE)

# Get the size of the dataset
size = data.get_dataset_size()


def _conv_transpose_block(in_channels, out_channels, kernel_size, stride, padding):
    """A block of the generator's architecture."""
    weight_init = init.Normal(mean=0, sigma=0.02)
    gamma_init = init.Normal(mean=1, sigma=0.02)
    return nn.SequentialCell([
        nn.Conv2dTranspose(in_channels=in_channels, out_channels=out_channels,
                           kernel_size=kernel_size, stride=stride, padding=padding,
                           weight_init=weight_init, has_bias=False, pad_mode="pad"),
        nn.BatchNorm2d(num_features=out_channels, gamma_init=gamma_init),
        nn.ReLU()
    ])


class Generator(nn.Cell):
    """Generator part of the DCGAN network."""

    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        self.generator = nn.SequentialCell([
            _conv_transpose_block(nz, ngf * 8, kernel_size=4, stride=1, padding=0),
            _conv_transpose_block(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1),
            _conv_transpose_block(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1),
            _conv_transpose_block(ngf * 2, ngf, kernel_size=4, stride=2, padding=1),
            nn.Conv2dTranspose(ngf, nc, kernel_size=4, stride=2, padding=1,
                               weight_init=init.Normal(0, 0.02), has_bias=False, pad_mode="pad"),
            nn.Tanh()
        ])

    def construct(self, x):
        return self.generator(x)


# Instantiate the generator
netG = Generator(NZ, NGF, NC)


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0, pad_mode="pad", use_bias=False):
    """Define a convolutional layer with normal weight initialization."""
    weight_init = init.Normal(mean=0, sigma=0.02)
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     weight_init=weight_init, has_bias=use_bias, pad_mode=pad_mode)


def bn(num_features):
    """Define a BatchNorm2d layer with normal weight initialization."""
    gamma_init = init.Normal(mean=1, sigma=0.02)
    return nn.BatchNorm2d(num_features=num_features, gamma_init=gamma_init)


class Discriminator(nn.Cell):
    """Discriminator part of the DCGAN network."""

    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        layers = [
            conv(nc, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2),
            conv(ndf, ndf * 2, 4, 2, 1),
            bn(ndf * 2),
            nn.LeakyReLU(0.2),
            conv(ndf * 2, ndf * 4, 4, 2, 1),
            bn(ndf * 4),
            nn.LeakyReLU(0.2),
            conv(ndf * 4, ndf * 8, 4, 2, 1),
            bn(ndf * 8),
            nn.LeakyReLU(0.2),
            conv(ndf * 8, 1, 4, 1),
            nn.Sigmoid()
        ]
        self.discriminator = nn.SequentialCell(layers)

    def construct(self, x):
        return self.discriminator(x)


class WithLossCellG(nn.Cell):
    """Combine Generator with loss function."""

    def __init__(self, netD, netG, loss_fn):
        super(WithLossCellG, self).__init__()
        self.netD = netD
        self.netG = netG
        self.loss_fn = loss_fn

    def construct(self, latent_code):
        fake_data = self.netG(latent_code)
        out = self.netD(fake_data)
        loss = self.loss_fn(out, ops.OnesLike()(out))
        return loss


def _compute_loss(data, label_fn, net, loss_fn, stop_grad=False):
    if stop_grad:
        data = ops.stop_gradient(data)
    out = net(data)
    label = label_fn()(out)
    return loss_fn(out, label)


class WithLossCellD(nn.Cell):
    """Combine Discriminator with loss function."""

    def __init__(self, netD, netG, loss_fn):
        super(WithLossCellD, self).__init__()
        self.netD = netD
        self.netG = netG
        self.loss_fn = loss_fn

    def construct(self, real_data, latent_code):
        loss_real = _compute_loss(real_data, ops.OnesLike, self.netD, self.loss_fn)
        loss_fake = _compute_loss(self.netG(latent_code), ops.ZerosLike, self.netD, self.loss_fn, stop_grad=True)
        return loss_real + loss_fake


# Instantiate Discriminator
netD = Discriminator(NC, NDF)

# Instantiate the loss function
loss = BCELoss(reduction='mean')

# Create a batch of latent vectors to visualize the progression of the generator
np.random.seed(1)  # For reproducibility
fixed_noise = Tensor(np.random.randn(64, NZ, 1, 1), dtype=ms.float32)

# Set up optimizers for the generator and discriminator
optimizerD = Adam(netD.trainable_params(), learning_rate=LR, beta1=BETA1)
optimizerG = Adam(netG.trainable_params(), learning_rate=LR, beta1=BETA1)


class DCGAN(nn.Cell):
    """DCGAN Network Definition."""

    def __init__(self, train_step_discriminator, train_step_generator):
        super(DCGAN, self).__init__()
        self.train_step_discriminator = train_step_discriminator
        self.train_step_generator = train_step_generator

    def construct(self, real_data, latent_code):
        # Train discriminator and compute its loss
        discriminator_output = self.train_step_discriminator(real_data, latent_code).view(-1)
        discriminator_loss = discriminator_output.mean()

        # Train generator and compute its loss
        generator_output = self.train_step_generator(latent_code).view(-1)
        generator_loss = generator_output.mean()

        return discriminator_loss, generator_loss


# Setup training components for DCGAN
net_d_with_loss = WithLossCellD(netD, netG, loss)
net_g_with_loss = WithLossCellG(netD, netG, loss)
train_step_discriminator = nn.TrainOneStepCell(net_d_with_loss, optimizerD)
train_step_generator = nn.TrainOneStepCell(net_g_with_loss, optimizerG)

# Initialize the DCGAN network and training data
dcgan = DCGAN(train_step_discriminator, train_step_generator).set_train()
data_loader = data.create_dict_iterator(output_numpy=True, num_epochs=NUM_EPOCHS)

# Initialize lists for tracking losses and generated images
g_losses, d_losses, image_list = [], [], []

# Script execution
if __name__ == "__main__":
    # Step 1: Prepare the dataset
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    download_dataset(URL, ZIP_PATH)
    extract_dataset(ZIP_PATH, DATA_DIR)
    check_data_directory(DATA_DIR)

    # Step 2: Set the MindSpore context
    ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")

    # Step 3: Create the dataset for training
    data = create_dataset_imagenet(DATA_ROOT, IMAGE_SIZE, NZ, BATCH_SIZE)
    size = data.get_dataset_size()

    # Step 4: Initialize DCGAN network components
    netG = Generator(NZ, NGF, NC)
    netD = Discriminator(NC, NDF)
    optimizerD = Adam(netD.trainable_params(), learning_rate=LR, beta1=BETA1)
    optimizerG = Adam(netG.trainable_params(), learning_rate=LR, beta1=BETA1)

    net_d_with_loss = WithLossCellD(netD, netG, loss)
    net_g_with_loss = WithLossCellG(netD, netG, loss)
    train_step_discriminator = nn.TrainOneStepCell(net_d_with_loss, optimizerD)
    train_step_generator = nn.TrainOneStepCell(net_g_with_loss, optimizerG)
    dcgan = DCGAN(train_step_discriminator, train_step_generator)
    dcgan.set_train()

    # Step 5: Run the training loop
    g_losses = []
    d_losses = []
    image_list = []
    print("Starting Training Loop...")
    for epoch in range(NUM_EPOCHS):
        # Wrap data_loader with tqdm for a progress bar
        data_loader_with_progress = tqdm(enumerate(data_loader), total=size, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        for i, data_batch in data_loader_with_progress:
            real_data = Tensor(data_batch['image'])
            latent_code = Tensor(data_batch["latent_code"])
            net_d_loss, net_g_loss = dcgan(real_data, latent_code)

            # Log training progress
            if i % 50 == 0 or i == size - 1:
                # Update the progress bar description
                data_loader_with_progress.set_description(
                    f'[{epoch + 1}/{NUM_EPOCHS}][{i + 1}/{size}] '
                    f'Loss_D: {net_d_loss.asnumpy():.4f} '
                    f'Loss_G: {net_g_loss.asnumpy():.4f}')

            d_losses.append(net_d_loss.asnumpy())
            g_losses.append(net_g_loss.asnumpy())
            # Generate images after each epoch
            generated_images = netG(fixed_noise)
            image_list.append(generated_images.transpose(0, 2, 3, 1).asnumpy())

            # Save model parameters
            save_checkpoint(netG, "Generator.ckpt")
            save_checkpoint(netD, "Discriminator.ckpt")

    # Step 6: Display or save results
    # Plot the training losses
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_losses, label="G", color='blue')
    plt.plot(d_losses, label="D", color='orange')
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
