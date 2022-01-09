
import torch
import torch.utils.data
import torch.utils.data.distributed
import torchvision


def one_hot_tensor_to_mask_recover_tensor(one_hot_tensor):

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


def save_images(imgs, outfile, nrow=8):
    imgs = imgs / 2 + 0.5     # unnormalize
    torchvision.utils.save_image(imgs, outfile, nrow=nrow)


def get_nsamples(data_loader, N):
    x = []
    y = []
    n = 0
    while n < N:
        #x_next, y_next = iter(data_loader).next()
        x_next = iter(data_loader).next()
        x.append(x_next)
        #y.append(y_next)
        n += x_next.size(0)
    x = torch.cat(x, dim=0)[:N]
    #y = torch.cat(y, dim=0)[:N]
    #return x, y
    return x


def update_average(model_tgt, model_src, beta):
    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert(p_src is not p_tgt)
        p_tgt.copy_(beta*p_tgt + (1. - beta)*p_src)
