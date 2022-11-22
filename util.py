import os
from torchvision.utils import save_image
from torchvision.utils import make_grid
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
plt.rcParams['animation.ffmpeg_path'] = '/bin/ffmpeg'


def denorm(img_tensors, stats):
        return img_tensors * stats[1][0] + stats[0][0]

def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid(denorm(images.detach()[:nmax]), nrow=8).permute(1, 2, 0))

def show_batch(dl, nmax=64):
    for images, _ in dl:
        show_images(images, nmax)
        break

def get_default_device(x=0):
    if torch.cuda.is_available():
        return torch.device(f'cuda:{x}')
    else:
        return torch.device('cpu')

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def gen_save_samples(generator, sample_dir, index, latent_tensors, stats, show=True,prefix=""):
        fake_images = generator(latent_tensors)
        fake_fname = prefix+'generated-images-{0:0=4d}.png'.format(index)
        save_image(denorm(fake_images, stats), os.path.join(sample_dir, fake_fname), nrow=8)
        print('Saving', fake_fname, "to", sample_dir)
        if show:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_xticks([]);
            ax.set_yticks([])
            ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0),vmin=-1,vmax=1)
        return fake_images
def train_summary(fakes,loss_g,loss_d,sample_epochs,name,fid_score = -1,prefix=''):
    print(name,sample_epochs,len(fakes),fakes[0].shape)
    fig = plt.figure(figsize=(18, 12), layout="constrained")
    spec = fig.add_gridspec(4, 8,left=0.05, right=0.95,hspace=1,wspace=1)
    axd = []
    axd.append(fig.add_subplot(spec[0:4, 0:4]))
    axd.append(fig.add_subplot(spec[0:4, 4:8]))
    images = axd[0].imshow(make_grid((fakes[0].cpu().detach()+1)/2, nrow=8).permute(1, 2, 0),vmax=1,vmin=0,interpolation_stage='rgba')

    axd[1].set_title('Loss')
    loss_line_g, =axd[1].plot([],[],label='gen_loss')
    loss_line_d, =axd[1].plot([],[],label='disc_loss')

    plt.legend()
    writer = FFMpegWriter(fps=0.5)
    with writer.saving(fig,prefix+name+'.mp4',dpi=100):
        for idx,ep in enumerate(sample_epochs):
            images.set_data(make_grid((fakes[idx].cpu().detach()+1)/2, nrow=8).permute(1, 2, 0))
            axd[1].set_xlim(left=-2,right=sample_epochs[-1]+8)
            axd[1].set_ylim(bottom=min([100]+loss_g[:ep]+loss_d[:ep])*0.9, top=max([0]+loss_g[:ep]+loss_d[:ep])*1.1)
            loss_line_g.set_data(range(min(ep,len(loss_g))),loss_g[:ep])
            loss_line_d.set_data(range(min(ep,len(loss_d))),loss_d[:ep])
            ep+=1
            print("frame for epoch {}".format(ep))
            if idx == len(sample_epochs)-1: # last slide
                axd[0].set_title('epoch: {}, FID={}'.format(ep + 1,fid_score))
            else:
                axd[0].set_title('epoch: {}'.format(ep+1))
            writer.grab_frame()

        writer.grab_frame()
        writer.grab_frame()
        writer.grab_frame()


class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)
    def __len__(self):
        return len(self.dl)

def addGaussianNoise(tensor,device,mean=0,std=1):
    return tensor + (torch.randn(tensor.size())*std +mean).to(device)