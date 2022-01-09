import torch
import torch.nn.functional as F

import numpy as np


def reconstruction_loss(x, x_recon, distribution):
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, reduction='sum').div(batch_size)
    elif distribution == 'gaussian':
        x_recon = torch.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, reduction='sum').div(batch_size)
    elif distribution == 'laplacian':
        x_recon = torch.sigmoid(x_recon)
        recon_loss = F.l1_loss(x_recon, x, reduction='sum').div(batch_size)
        
    elif distribution == 'categorical':
        
        def one_hot_tensor_to_order_label_tensor(one_hot_tensor):

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
                Converts a mask (K, C, H, W) to (K,H,W)
                """
                _mask = np.argmax(one_hot, axis=1)
                _mask = _mask.reshape((batch_size,image_size,image_size))
                return _mask
        
            order_with_trainid = onehot2mask(gt_concat)
        
            order_recover = order_with_trainid # shape = (K,H, W)
        
            order_recover = torch.from_numpy(order_recover)
        
            return order_recover
        
        x_order = torch.LongTensor(one_hot_tensor_to_order_label_tensor(x)).cuda()
        
        recon_loss = F.cross_entropy(x_recon, x_order, reduction='sum').div(batch_size)
        
        
    else:
        raise NotImplementedError

    return recon_loss


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0

    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    kld = (-0.5*(1 + logvar - mu.pow(2) - logvar.exp())).sum(1).mean(0, True)
    return kld
