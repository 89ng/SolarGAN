import os    
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt

import numpy as np
import random
from tqdm import tqdm
from math import sqrt

import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from torch.utils.tensorboard import SummaryWriter

from gan_training.dvae.dvae import BetaVAE_H, normal_init
from gan_training.dvae.dataset import return_data
from gan_training.dvae.loss import reconstruction_loss, kl_divergence


__COLOR_DATASETS__ = ['chairs', 'cars', 'celeba']
__SOLARGAN_DATASETS__ = ['grayfisheye', 'graycube']


class Solver(object):
    def __init__(self, args):
        use_cuda = args.cuda and torch.cuda.is_available()
        self.device = 'cuda' if use_cuda else 'cpu'

        self.global_iter = 0
        self.max_iter = args.max_iter
        self.ckpt_save_iter = args.ckpt_save_iter
        self.log_line_iter = args.log_line_iter
        self.log_img_iter = args.log_img_iter

        self.name = args.name
        self.output_dir = os.path.join(args.output_dir, args.name)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(self.output_dir, 'chkpts')
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=True)
        self.log_dir = os.path.join(self.output_dir, 'tensorboard')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)

        self.data_loader = return_data(args)
        self.writer = SummaryWriter(self.log_dir)

        self.c_dim = args.c_dim
        self.beta = args.beta
        self.nc = 5 if args.dataset.lower() in __SOLARGAN_DATASETS__ else 3 if args.dataset.lower() in __COLOR_DATASETS__ else 1
        self.dec_dist = args.dec_dist
        self.net = BetaVAE_H(self.c_dim, self.nc).to(self.device)
        self.net.apply(normal_init)
        self.optim = optim.Adam(self.net.parameters(), lr=args.lr,
                                betas=(args.beta1, args.beta2))
        if args.load_ckpt != 0:
            if args.load_ckpt == -1:
                filename = 'last'
            else:
                filename = str(args.load_ckpt)
            self.load_checkpoint(filename)

    def train(self):
        pbar = tqdm(total=self.max_iter)
        pbar.update(self.global_iter)
        out = False if self.global_iter < self.max_iter else True
        while not out:
            for x in self.data_loader:
                self.net.train()
                self.global_iter += 1
                pbar.update(1)

                x = x.to(self.device)
                x_recon, c, mu, logvar = self.net(x)

                recon_loss = reconstruction_loss(x, x_recon, self.dec_dist)
                kld = kl_divergence(mu, logvar)
                beta_vae_loss = recon_loss + self.beta*kld

                self.optim.zero_grad()
                beta_vae_loss.backward()
                self.optim.step()

                pbar.set_description('[{}] recon_loss:{:.3f} kld:{:.3f}'.format(
                    self.global_iter, recon_loss.item(), kld.item()))

                if self.global_iter % self.log_line_iter == 0:
                    self.writer.add_scalar('recon_loss', recon_loss, self.global_iter)
                    self.writer.add_scalar('kld', kld, self.global_iter)

                if self.global_iter % self.log_img_iter == 0:
                        
                    if self.nc ==5:
                        x_im = self.one_hot_tensor_to_mask_recover_tensor(x)
                        x_im_recon = x_recon.softmax(dim=1).argmax(dim=1).unsqueeze(1).cpu()/(self.nc-1)
                        
                        
                        x_im = make_grid(x_im, nrow=int(sqrt(x_im.size(0))), padding=2, pad_value=1)
                        x_im_recon = make_grid(x_im_recon, nrow=int(sqrt(x_im_recon.size(0))), padding=2, pad_value=1)
                        x_im_vis = make_grid(torch.stack([x_im, x_im_recon]), nrow=2, padding=2, pad_value=0)
                        self.writer.add_image('reconstruction', x_im_vis, self.global_iter)
                        
                        #plt.figure(dpi=288) #figsize adjustment

                        #self.show(x_im_vis)
                        
                        
                        
                        # visualize traverse (convert to imgs first)
                        self.traverse_solargan(c_post=mu[:1], c_prior=torch.randn_like(mu[:1]))
                        
                    else: 
                    # visualize reconstruction 
                        x = make_grid(x, nrow=int(sqrt(x.size(0))), padding=2, pad_value=1)
                        x_recon = make_grid(x_recon.sigmoid(), nrow=int(sqrt(x_recon.size(0))), padding=2, pad_value=1)
                        x_vis = make_grid(torch.stack([x, x_recon]), nrow=2, padding=2, pad_value=0)
                        self.writer.add_image('reconstruction', x_vis, self.global_iter)
    
                        # visualize traverse
                        self.traverse(c_post=mu[:1], c_prior=torch.randn_like(mu[:1]))
                        
                        


                if self.global_iter % self.ckpt_save_iter == 0:
                    self.save_checkpoint()
                    pbar.write('Saved checkpoint (iter:{})'.format(self.global_iter))

                if self.global_iter >= self.max_iter:
                    self.save_checkpoint()
                    pbar.write('Saved checkpoint (iter:{})'.format(self.global_iter))
                    out = True
                    break

        pbar.write("[Training Finished]")
        pbar.close()
        
    def one_hot_tensor_to_mask_recover_tensor(self,one_hot_tensor):
    
        batch_size = one_hot_tensor.size(0)
        image_size = one_hot_tensor.size(2)
        
        #platte = [64,128,192,256]
        
        #gt_concat = np.concatenate((black_edge,one_hot_tensor.cpu().numpy()),axis=1)
        gt_concat = one_hot_tensor.cpu().numpy()
        #print(gt_concat.shape)
        
        platte2trainid ={0:0, 64:1, 128:2, 192:3, 256:4}
        #platte = [64,128,192,256]
        
        #black_edge=gt_one_hot[:,:,0].numpy()
        #black_edge = black_edge.reshape(1,image_size,image_size)
        #black_edge = np.zeros((batch_size,1,image_size,image_size))
    
        def onehot2mask(one_hot):
            """
            Converts a mask (K, C, H, W) to (K,1,H,W)
            """
            _mask = np.argmax(one_hot, axis=1)
            _mask = _mask.reshape((batch_size,1,image_size,image_size))
            return _mask
        
        mask_with_trainid = onehot2mask(gt_concat)
        
        mask_recover = mask_with_trainid # shape = (H, W)
        #mask_df = pd.DataFrame(mask_recover)
        #mask_df.to_csv('mask.csv')
        for k, v in platte2trainid.items():
            mask_recover[mask_with_trainid == v] = k
            
        mask_recover = torch.from_numpy(mask_recover/256)
            
        return mask_recover

    def traverse(self, c_post, c_prior, limit=3, npoints=7, pos=-1):
        assert isinstance(pos, (int, list, tuple))

        self.net.eval()
        c_dict = {'c_posterior':c_post, 'c_prior':c_prior}
        interpolation = torch.linspace(-limit, limit, npoints)

        for c_key in c_dict:
            c_ori = c_dict[c_key]
            samples = []
            for row in range(self.c_dim):
                if pos != -1 and row not in pos:
                    continue

                c = c_ori.clone()
                for val in interpolation:
                    c[:, row] = val
                    sample = self.net(c=c, decode_only=True).sigmoid().data
                    samples.append(sample)

            samples = torch.cat(samples, dim=0).cpu()
            samples = make_grid(samples, nrow=npoints, padding=2, pad_value=1)
            tag = 'latent_traversal_{}'.format(c_key)
            self.writer.add_image(tag, samples, self.global_iter)

        self.net.train()
        
    def traverse_solargan(self, c_post, c_prior, limit=3, npoints=7, pos=-1):
        assert isinstance(pos, (int, list, tuple))

        self.net.eval()
        c_dict = {'c_posterior':c_post, 'c_prior':c_prior}
        interpolation = torch.linspace(-limit, limit, npoints)

        for c_key in c_dict:
            c_ori = c_dict[c_key]
            samples = []
            for row in range(self.c_dim):
                if pos != -1 and row not in pos:
                    continue

                c = c_ori.clone()
                for val in interpolation:
                    c[:, row] = val
                    sample = self.net(c=c, decode_only=True).softmax(dim=1).argmax(dim=1).unsqueeze(1).data
                    sample = sample/(self.nc-1)
                    samples.append(sample)

            samples = torch.cat(samples, dim=0).cpu()
            samples = make_grid(samples, nrow=npoints, padding=2, pad_value=1)
            tag = 'latent_traversal_{}'.format(c_key)
            self.writer.add_image(tag, samples, self.global_iter)

        self.net.train()

    def save_checkpoint(self):
        model_states = {'net':self.net.state_dict(),
                        'c_dim':self.c_dim,
                        'nc':self.nc}
        optim_states = {'optim':self.optim.state_dict(),}
        states = {'iter':self.global_iter,
                  'model_states':model_states,
                  'optim_states':optim_states}

        file_path = os.path.join(self.ckpt_dir, str(self.global_iter))
        with open(file_path, mode='wb+') as f:
            torch.save(states, f)

        file_path = os.path.join(self.ckpt_dir, 'last')
        with open(file_path, mode='wb+') as f:
            torch.save(states, f)

    def load_checkpoint(self, filename):
        file_path = os.path.join(self.ckpt_dir, filename)
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path)
            self.global_iter = checkpoint['iter']
            self.net.load_state_dict(checkpoint['model_states']['net'])
            self.optim.load_state_dict(checkpoint['optim_states']['optim'])
            tqdm.write("=> loaded checkpoint '{} (iter {})'".format(file_path, self.global_iter))
        else:
            tqdm.write("=> no checkpoint found at '{}'".format(file_path))
            
    def show(self,img_tensor):
        npimg = img_tensor.numpy()
        plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
