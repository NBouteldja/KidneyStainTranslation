# This code was copied from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix and adapted for our purpose accordingly.
from scipy import misc
import os, cv2, torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

def load_test_data(image_path, size=256):
    img = misc.imread(image_path, mode='RGB')
    img = misc.imresize(img, [size, size])
    img = np.expand_dims(img, axis=0)
    img = preprocessing(img)

    return img

def norm(t_list):
    for i,t in enumerate(t_list):
        t_list[i] = (t-t.mean())/(t.std()+1e-15)*2 + .5
    return t_list


def concat_zeros(tensor_list):
    for i in range(len(tensor_list)):
        tensor_list[i]= torch.cat((tensor_list[i],torch.zeros_like(tensor_list[i])),dim = 1)
    return tuple(tensor_list) if len(tensor_list) >= 2 else tensor_list[0]

def split_6_ch_to_3(tensor_list):
    for i in range(len(tensor_list)):
        tensor_list[i]= tensor_list[i].split(3,dim = 1)[0]
    return  tuple(tensor_list) if len(tensor_list) >= 2 else tensor_list[0]

def split_6_ch_to_2x3(tensor_list):
    for i in range(len(tensor_list)):
        a,b = tensor_list[i].split(3,dim = 1)
        tensor_list[i] = torch.cat((a,b),dim = 3)
    return  tuple(tensor_list)


def preprocessing(x):
    x = x/127.5 - 1 # -1 ~ 1
    return x


def get_puzzle_params(ps = 256, n_patches = 9, imgsize= 640):
    n = int(np.sqrt(n_patches))
    assert(n>=2),"n_patches should be at least 4"
    to =  n * ps - imgsize
    oo = int(to/(n-1)+.5)
    
    l = []
    for i in range(n):
        for j in range(n):
            if j == 0:
                x = 0
            elif j == n-1:
                x = imgsize - ps 
            else: 
                x = j * (ps - oo)
            if i == 0:
                y = 0
            elif i == n-1:
                y = imgsize  - ps 
            else: 
                y = i * (ps - oo)
            l.append((x ,y ,x+ps ,y+ps ))
    return l

def get_imgs(img, params):
    cropped_np_imgs = []
    for param in params:
        img.crop(param)
        cropped_np_imgs.append(np.array(img.crop(param)))
    return cropped_np_imgs

def get_overlap_weights(params, imgsize):
    arr = np.zeros((imgsize,imgsize)) 
    for param in params:
        for x in range(param[1],param[3]):
            for y in range(param[0],param[2]):
                arr[x][y] = arr[x][y] + 1
    return 1/arr

def puzzle(np_imgs, out_size):
    n_patches = len(np_imgs)
    ps = np_imgs[0].shape[0]
    assert(np.sqrt(n_patches) * ps >= out_size), "There are not enough patches to fill the desired outputsize"
    params = get_puzzle_params(ps, n_patches, out_size)
    arr = np.zeros((out_size,out_size,3))
    for i,param in enumerate(params):
        arr[param[1]:param[3], param[0]:param[2]] += np_imgs[i]
    weights = get_overlap_weights(params, out_size)
    weights = np.stack((weights,weights,weights), axis = -1)
    arr = arr*weights
    return arr.astype('uint8')    

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def inverse_transform(images):
    return (images+1.) / 2

def imsave(images, size, path):
    return misc.imsave(path, merge(images, size))

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[h*j:h*(j+1), w*i:w*(i+1), :] = image

    return img

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def str2bool(x):
    return x.lower() in ('true')

def cam(x, size = 256):
    x = x - np.min(x)
    cam_img = x / np.max(x)
    cam_img = np.uint8(255 * cam_img)
    cam_img = cv2.resize(cam_img, (size, size))
    cam_img = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
    return cam_img / 255.0

def imagenet_norm(x):
    mean = [0.485, 0.456, 0.406]
    std = [0.299, 0.224, 0.225]
    mean = torch.FloatTensor(mean).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(x.device)
    std = torch.FloatTensor(std).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(x.device)
    return (x - mean) / std

def denorm(x):
    return x * 0.5 + 0.5

def tensor2numpy(x):
    return x.detach().cpu().numpy().transpose(1,2,0)

def RGB2BGR(x):
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)

def BGR2RGB(x):
    return cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

def one_hot(num_labels,n):
    c_org = torch.FloatTensor(torch.eye(num_labels)[n])
    c_trg = torch.FloatTensor(torch.eye(num_labels)[np.random.choice([i for i in range(0,num_labels) if i not in [n]])])
    return c_org, c_trg

def get_trg_label(num_labels,n):
    if n == 0:
        return n, np.random.choice([i for i in range(0,num_labels)]) 
    elif np.random.randint(0,2): return n, 0                         
    return n, np.random.choice([i for i in range(0,num_labels)])


def get_trg_label_2(num_labels,n):
    if n == 0:
        return 0, np.random.choice([i for i in range(0,num_labels)])
    else:
        return n, 0                          # translate non Pas into Pas
    

### gradient_penalty from the Stargan paper
def gradient_penalty(device , y, x):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    weight = torch.ones(y.size()).to(device)
    dydx = torch.autograd.grad(outputs=y,
                                inputs=x,
                                grad_outputs=weight,
                                retain_graph=True,
                                create_graph=True,
                                only_inputs=True)[0]

    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
    return torch.mean((dydx_l2norm-1)**2)

def load_dataset(loader):
    batches=[]
    for batch in loader:
        batches.append(batch)
    return batches
def set_name(args):
    if args.stain_root == 'stains[0]': 
        stain_root = ''
        args.stains.sort()
        for stain in args.stains:
            stain_root = stain_root + stain + '_'    

        if args.star: 
            if args.gen_org:
                stain_root = 'org_' + stain_root
            elif args.gen_trg:
                stain_root = 'trg_' + stain_root
            if args.no_cross:
                stain_root = 'no_cross_' + stain_root
            stain_root = 'stargan_' + stain_root 
        elif args.guided:
            stain_root = 'guided_' + stain_root     
        elif args.hydra:
            stain_root = 'hydra_' + stain_root     
        else:
            stain_root = 'baseline_' + stain_root    
             
        args.stain_root = stain_root + 'sixpack' if args.six_pack else stain_root[:-1]


def init_model(self, args):
    self.light = args.light
    
    if self.light :
        self.model_name = 'UGATIT_light'
    else :
        self.model_name = 'UGATIT'
    
    if args.star: self.model_name = 'Stargan_' + self.model_name 
    elif args.guided:self.model_name = 'Guided_' + self.model_name 
    
    self.result_dir = args.result_dir
    self.dataset = args.dataset
    self.stains = args.stains
    self.stain_root = args.stain_root
    self.mult_stains = len(self.stains) >= 2
    self.num_stains = len(self.stains)
    self.stainB = args.stainB


    self.iteration = args.iteration
    self.decay_flag = args.decay_flag

    self.batch_size = args.batch_size
    self.print_freq = args.print_freq
    self.save_freq = args.save_freq

    self.lr = args.lr
    self.weight_decay = args.weight_decay
    self.ch = args.ch

    self.puzzle = args.puzzle
    self.preload = args.preload
    if args.phase == 'train':
        self.writer = SummaryWriter(log_dir = os.path.join(args.result_dir, args.stain_root, 'logs'))

    """ Weight """
    self.adv_weight = args.adv_weight
    self.cycle_weight = args.cycle_weight
    self.identity_weight = args.identity_weight
    self.cam_weight = args.cam_weight
    self.cls_weight = args.cls_weight
    self.gp_weight = args.gp_weight


    """ Generator """
    self.n_res = args.n_res
    self.six_pack = args.six_pack
    
    """ Discriminator """
    self.n_dis = args.n_dis

    self.img_size = args.img_size
    self.img_ch = args.img_ch
    self.crop_size = args.crop_size

    self.device = args.device
    self.benchmark_flag = args.benchmark_flag
    self.resume = args.resume

    self.phase = args.phase

    if torch.backends.cudnn.enabled and self.benchmark_flag:
        print('set benchmark !')
        torch.backends.cudnn.benchmark = True

    print()

    print("##### Information #####")
    print("# Name : ", self.stain_root)
    print("# light : ", self.light)
    print("# dataset : ", self.dataset)
    print("# stains : ", self.stains)
    print("# batch_size : ", self.batch_size)
    print("# iteration per epoch : ", self.iteration)
    if self.six_pack: print("# this net went to the gym! ")
    print()
    print("#Image size: ", self.crop_size)
    
    print("##### Generator #####")
    print("# residual blocks : ", self.n_res)

    print()

    print("##### Discriminator #####")
    print("# discriminator layer : ", self.n_dis)

    print()

    print("##### Weight #####")
    print("# adv_weight : ", self.adv_weight)
    print("# cycle_weight : ", self.cycle_weight)
    print("# identity_weight : ", self.identity_weight)
    print("# cam_weight : ", self.cam_weight)
    if args.star:
        print("# class loss weight : ", self.cls_weight) 
        print("# gradient penalty loss weight : ", self.gp_weight) 

