import argparse
import random
import time

import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from models import GAN, Generator, Discriminator
from models64 import *
from train import Trainer
from util import *

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, help='path to inputs folder', default='organic',required=False)
parser.add_argument('--sample_rate', type=int, help='epochs between samples', default=10,required=False)
parser.add_argument('--checkpoint_rate', type=int, help='epochs between checkpoints', default=25,required=False)
parser.add_argument('--start', type=int, help='epoch to start from', default=0,required=False)
parser.add_argument('--name', type=str, help='save dir name',default='',required=False)
parser.add_argument('--epochs', type=int, help='num of epochs to train',default=301,required=False)
parser.add_argument('--lr', type=float, help='learning rate', default=0.0002,required=False)
parser.add_argument('--bc', type=int, help='batch size', default=64,required=False)
parser.add_argument('--noise_std', type=float, help='standard deviation of noise', default=0.0,required=False)
parser.add_argument('--noise_fade', type=float, help='share of epochs with noise', default=1 / 3,required=False)
parser.add_argument('--small', type=bool, help='use small training set', default=True,required=False)
parser.add_argument('--small_size', type=int, help='small training set size', default=5000,required=False)
parser.add_argument('--config', type=str, help='config name', default=None,required=False)
parser.add_argument('--gpu_pool', type=int, help="num of GPU's to draw from", default=1,required=False)

args = parser.parse_args()

stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)


def fit(epochs, lr, fixed_latent, generator, discriminator, start_idx=0, name="model", std=0.1,
        fade_noise=(True, 1 / 2)):
    torch.cuda.empty_cache()
    gl_time = time.time()
    # Losses & scores
    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []
    fake_images = []
    sample_epochs = []

    # Create optimizers
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    if start_idx != 0:  # load model
        checkpoint = torch.load(os.path.join(args.name, '{}_{0:0=4d}.pth'.format(name, start_idx)))
        start_idx = checkpoint["epoch"]
        generator.load_state_dict(checkpoint["gen_sd"])
        opt_g.load_state_dict(checkpoint["opt_g_sd"])
        losses_g = checkpoint["loss_g"]
        discriminator.load_state_dict(checkpoint["dis_sd"])
        opt_d.load_state_dict(checkpoint["opt_d_sd"])
        losses_d = checkpoint["loss_d"]
        fixed_latent = checkpoint["fixed_latent"]

    trainer = Trainer(discriminator, generator, args.bc, device, latent_size)

    train_std = std
    for epoch in range(start_idx, epochs):
        tim = time.time()
        if fade_noise[0]:
            train_std = std * (1 - min(0.95, fade_noise[1] * epoch / epochs))
            # linearly fade to a fraction over first half of the training
        for real_images, _ in train_dl:
            # Train discriminator
            if train_std > 0:
                real_images = addGaussianNoise(real_images, device, std=train_std)
            loss_d, real_score, fake_score = trainer.train_discriminator(real_images, opt_d)
            # Train generator
            loss_g = trainer.train_generator(opt_g, std=train_std)

        # Record losses & scores
        losses_g.append(loss_g)
        losses_d.append(loss_d)
        real_scores.append(real_score)
        fake_scores.append(fake_score)

        # Log losses & scores (last batch)
        print(
            "Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}, epoch_time:{:.4f}, time:{:.4f}".format(
                epoch + 1, epochs, loss_g, loss_d, real_score, fake_score, time.time() - tim, time.time() - gl_time))

        # Save generated images
        if (epoch != start_idx and epoch % args.sample_rate == 0) or epoch == epochs - 1:
            fake_images.append(gen_save_samples(generator, args.name, epoch + 1, fixed_latent, stats, show=False))
            sample_epochs.append(epoch)
        if not args.small:
            if (epoch != start_idx and epoch % args.checkpoint_rate == 0) or epoch == epochs - 1:
                torch.save({
                    "epoch": epoch + 1, "gen_sd": generator.state_dict(), "opt_g_sd": opt_g.state_dict(),
                    "loss_g": losses_g,
                    "dis_sd": discriminator.state_dict(), "opt_d_sd": opt_d.state_dict(), "loss_d": losses_d,
                    "fixed_latent": fixed_latent
                }, os.path.join(args.name, '{}_{:0=4d}.pth'.format(name, epoch + 1)))
                print("saved checkpoint {}_{:0=4d}.pth".format(name, epoch + 1))

    train_summary(fake_images,losses_g,losses_d,fake_scores,real_scores,sample_epochs,epochs,
                  '{}_{}_{}_{}_lr{}_noise{}_{}'.format(args.name,args.image_size,
                                                         generator.__class__.__name__,
                                                         discriminator.__class__.__name__,
                                                         str(round(args.lr,6))[2:],str(round(args.noise_std,6))[2:],
                                                         str(round(args.noise_fade,4))[2:]),
                  prefix=args.name+'/')
    return losses_g, losses_d, real_scores, fake_scores


# 8k images, 32*32, ls=64, eps=10 , bs=256 => 1782 s, results= okeish
if __name__ == '__main__':

    torch.manual_seed(42)
    device = get_default_device(random.randint(0, args.gpu_pool-1))
    if args.config is None or args.config == 'mixed64':
        latent_size = 64
        image_size = 64
        model = GAN(GeneratorSkip64(latent_size, device).to(device), DiscriminatorResidual64().to(device))
    elif args.config == 'simple32':
        latent_size = 32
        image_size = 32
        model = GAN(Generator(latent_size).to(device), Discriminator().to(device))
    elif args.config == 'simple64':
        latent_size = 64
        image_size = 64
        model = GAN(Generator64(latent_size).to(device), Discriminator64().to(device))
    elif args.config == 'skip64':
        latent_size = 64
        image_size = 64
        model = GAN(GeneratorSkip64(latent_size,device).to(device), DiscriminatorSkip64(device).to(device))
    elif args.config == 'residual64':
        latent_size = 64
        image_size = 64
        model = GAN(GeneratorResidual64(latent_size,device).to(device), DiscriminatorResidual64().to(device))
    elif args.config == 'mixed64':
        latent_size = 64
        image_size = 64
        model = GAN(GeneratorSkip64(latent_size,device).to(device), DiscriminatorResidual64().to(device))
    elif args.config == 'FFMixed64':
        latent_size = 32
        image_size = 64
        model = GAN(GeneratorIntermidiate64(latent_size,device).to(device), DiscriminatorResidual64().to(device))
    else:
        latent_size = 64
        image_size = 64
        model = GAN(GeneratorSkip64(latent_size, device).to(device), DiscriminatorResidual64().to(device))

    args.image_size = image_size
    if args.name == '':
        args.name ='{}_{}_{}_{}_lr{}_noise{}_{}'.format(args.name, args.image_size,
                                         model.generator.__class__.__name__,
                                         model.discriminator.__class__.__name__,
                                         str(round(args.lr, 6))[2:], str(round(args.noise_std, 6))[2:],
                                         str(round(args.noise_fade, 4))[2:])

    train_ds = ImageFolder(args.source, transform=T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(*stats),
    ]))

    if args.small:
        train_dl = DataLoader(train_ds, args.bc, shuffle=False, num_workers=3, pin_memory=True,
                              sampler=range(0, args.small_size))
    else:
        train_dl = DataLoader(train_ds, args.bc, shuffle=True, num_workers=3, pin_memory=True)


    train_dl = DeviceDataLoader(train_dl, device)


    os.makedirs(args.name, exist_ok=True)

    fixed_latent = torch.randn(64, latent_size, 1, 1, device=device)
    gen_save_samples(model.generator, args.name, 0, fixed_latent, stats)
    history = fit(args.epochs, args.lr, fixed_latent, model.generator, model.discriminator, start_idx=args.start,
                  std=args.noise_std, fade_noise=((args.noise_fade * args.noise_std > 0), args.noise_fade))
    print('done')

