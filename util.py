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
def train_summary(fakes,loss_g,loss_d,score_fake,score_reall,sample_epochs,name,prefix=''):
    print(name)
    fig = plt.figure(figsize=(18, 12), layout="constrained")
    spec = fig.add_gridspec(4, 8,left=0.05, right=0.95,hspace=1,wspace=1)
    axd = []
    axd.append(fig.add_subplot(spec[0:4, 0:4]))
    axd.append(fig.add_subplot(spec[0:2, 4:8]))
    axd.append(fig.add_subplot(spec[2:4, 4:8]))
    images = axd[0].imshow(make_grid((fakes[0].cpu().detach()+1)/2, nrow=8).permute(1, 2, 0),vmax=1,vmin=0,interpolation_stage='rgba')

    axd[1].set_title('Loss')
    loss_line_g, =axd[1].plot([],[],label='gen_loss')
    loss_line_d, =axd[1].plot([],[],label='disc_loss')

    axd[2].set_title('Accuracy')
    score_line_f, =axd[2].plot([],[],label='fake_ac')
    score_line_r, =axd[2].plot([],[],label='real_ac')
    axd[2].set_ylim(bottom=0,top=1)
    plt.legend()
    print(len(loss_g),len(loss_d),len(score_fake),len(score_reall))
    writer = FFMpegWriter(fps=0.5)
    with writer.saving(fig,prefix+name+'.mp4',dpi=100):
        for idx,ep in enumerate(sample_epochs):
            ep+=1
            axd[0].set_title('epoch: {}'.format(ep+1))
            images.set_data(make_grid((fakes[idx].cpu().detach()+1)/2, nrow=8).permute(1, 2, 0))
            loss_line_g.set_data(range(ep),loss_g[:ep])
            axd[1].set_xlim(left=-2,right=ep+2)
            axd[1].set_ylim(bottom=min(loss_g[:ep]+loss_d[:ep]), top=max(loss_g[:ep]+loss_d[:ep]))
            loss_line_d.set_data(range(ep),loss_d[:ep])
            score_line_f.set_data(range(ep),score_reall[:ep])
            score_line_r.set_data(range(ep),score_fake[:ep])
            axd[2].set_xlim(left=-2,right=ep+2)
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