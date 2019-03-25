import numpy as np
import cmath
import math
import shutil

def cartesianToPolar(input):
    magnitude = np.vectorize(np.linalg.norm)
    phase = np.vectorize(np.angle)

    return np.dstack([magnitude, phase])

def polarToCartesian(input):
    output = np.zeros((input.shape[0], input.shape[1]), dtype=np.complex64)

    it = np.nditer(output, flags=['multi_index'])
    while not it.finished:
        output[it.multi_index] = cmath.rect(input[it.multi_index][0], input[it.multi_index][1])
        temp = it.iternext()

    return output

import torch
import torch.cuda as cuda
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.nn import functional as F
from skimage.measure import compare_ssim as ssim
from sklearn import preprocessing
import matplotlib.pyplot as plt


# ### Setting up the arguments

class Arguments:
    def __init__(self,batch_size,data_path,center_fractions,accelerations,challenge,sample_rate,resolution,resume,learning_rate,epoch,reluslope,exp_dir,checkpoint):
        self.batch_size = batch_size
        self.data_path = data_path
        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.challenge = challenge
        self.sample_rate = sample_rate
        self.resolution = resolution
        self.resume = resume
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.reluslope = reluslope
        self.exp_dir = exp_dir
        self.checkpoint = checkpoint
        
# ### Custom dataset class

import pathlib
import random

import h5py
from torch.utils.data import Dataset


class SliceData(Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(self, root, transform, challenge, sample_rate=1):
        """
        Args:
            root (pathlib.Path): Path to the dataset.
            transform (callable): A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace', 'target',
                'attributes', 'filename', and 'slice' as inputs. 'target' may be null
                for test data.
            challenge (str): "singlecoil" or "multicoil" depending on which challenge to use.
            sample_rate (float, optional): A float between 0 and 1. This controls what fraction
                of the volumes should be loaded.
        """
        if challenge not in ('singlecoil', 'multicoil'):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        self.transform = transform
        self.recons_key = 'reconstruction_esc' if challenge == 'singlecoil' else 'reconstruction_rss'
        self.challenge = challenge

        self.examples = []
        files = list(pathlib.Path(root).iterdir())
        if sample_rate < 1:
            random.shuffle(files)
            num_files = round(len(files) * sample_rate)
            files = files[:num_files]
        for fname in sorted(files):
            kspace = h5py.File(fname, 'r')['kspace']
            num_slices = kspace.shape[0]
            self.examples += [(fname, slice) for slice in range(num_slices)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice = self.examples[i]
        with h5py.File(fname, 'r') as data:
            kspace = data['kspace'][slice]
            target = data[self.recons_key][slice] if self.recons_key in data else None
            return self.transform(kspace, fname.name, target, self.challenge)


# ### Data Transform (return original and masked kspace)

from data import transforms
class DataTransform:
    """
    Data Transformer for training DAE.
    """

    def __init__(self, mask_func, resolution, use_seed=True):
        self.mask_func = mask_func
        self.use_seed = use_seed
        self.resolution = resolution

    def __call__(self, kspace, fname, target, challenge):
        original_kspace = transforms.to_tensor(kspace)
        # Apply mask
        seed = None if not self.use_seed else tuple(map(ord, fname))
        masked_kspace, mask = transforms.apply_mask(original_kspace, self.mask_func, seed)
        
         # Inverse Fourier Transform to get zero filled solution
        image = transforms.ifft2(masked_kspace)
        # Crop input image
        image = transforms.complex_center_crop(image, (self.resolution, self.resolution))
        # Absolute value
        image = transforms.complex_abs(image)
        # Apply Root-Sum-of-Squares if multicoil data
        if challenge == 'multicoil':
            image = transforms.root_sum_of_squares(image)
        # Normalize input
        image, mean, std = transforms.normalize_instance(image, eps=1e-11)
        
        target = transforms.to_tensor(target)
        # Normalize target
        target = transforms.normalize(target, mean, std, eps=1e-11)
        target = target.clamp(-6, 6)
        
        return original_kspace, masked_kspace, target, mask


# ### Creating data loaders

from common.subsample import MaskFunc
from torch.utils.data import DataLoader
def create_datasets(args):
    train_mask = MaskFunc(args.center_fractions, args.accelerations)
    dev_mask = MaskFunc(args.center_fractions, args.accelerations)

    train_data = SliceData(
        root=args.data_path + 'singlecoil_train',
        transform=DataTransform(train_mask, args.resolution),
        sample_rate=args.sample_rate,
        challenge=args.challenge
    )
    dev_data = SliceData(
        root=args.data_path + 'singlecoil_val',
        transform=DataTransform(dev_mask, args.resolution, use_seed=True),
        sample_rate=args.sample_rate,
        challenge=args.challenge,
    )
    return dev_data, train_data


def create_data_loaders(args, if_shuffle = True):
    dev_data, train_data = create_datasets(args)
    
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=if_shuffle, 
        pin_memory=True,
    )
    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        pin_memory=True,
    )
    return train_loader, dev_loader

# #### Some utility function

import matplotlib.pyplot as plt

def kspacetoimage(kspace, args):
    # Inverse Fourier Transform to get zero filled solution
    image = transforms.ifft2(kspace)
        # Crop input image
    image = transforms.complex_center_crop(image, (args.resolution, args.resolution))
        # Absolute value
    image = transforms.complex_abs(image)
        # Apply Root-Sum-of-Squares if multicoil data
    if args.challenge == 'multicoil':
        image = transforms.root_sum_of_squares(image)
        # Normalize input
    image, mean, std = transforms.normalize_instance(image, eps=1e-11)
    image = image.clamp(-6, 6)

    return image

def plotimage(image):
    plt.imshow(np.array(image))
    plt.show()
    
def transformshape(kspace):
    s = kspace.shape
    kspace = np.reshape(kspace , (s[0],s[3],s[1],s[2]))
    return kspace

def transformback(kspace):
    s = kspace.shape
    kspace = np.reshape(kspace , (s[0],s[2],s[3],s[1]))
    return kspace

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def compare_images(imageA, imageB,imageC,writer,iteration):
    mb = mse(imageA, imageB)
    sb = ssim(imageA, imageB)
    mc = mse(imageA, imageC)
    sc = ssim(imageA, imageC)

    fig = plt.figure()
    plt.suptitle("Target vs network MSE: %.2f, SSIM: %.2f" % (mb, sb)+" Target vs Zeroimputed MSE: %.2f, SSIM: %.2f" % (mc, sc))
     
    ax = fig.add_subplot(2, 2, 1)
    plt.imshow(imageA)
    plt.axis("off")

    ax = fig.add_subplot(2, 2, 2)
    plt.imshow(imageB)
    plt.axis("off")
    
    ax = fig.add_subplot(2, 2, 3)
    plt.imshow(imageA)
    plt.axis("off")

    ax = fig.add_subplot(2, 2, 4)
    plt.imshow(imageC)
    plt.axis("off")
    
    plt.show()
    writer.add_figure('Comparision', fig, global_step = iteration)    

def compareimageoutput(target,masked_kspace,outputkspace,mask,writer,iteration):
    unmask = np.where(mask==1.0, 0.0, 1.0)
    unmask = transforms.to_tensor(unmask)
    unmask = unmask.float()
    output = transformback(outputkspace.data.cpu())
    output = output * unmask
    output = output + masked_kspace.data.cpu()
    imageA = np.array(target)[0]
    imageB = np.array(kspacetoimage(output))[0]
    imageC = np.array(kspacetoimage(masked_kspace.data.cpu()))[0]
    compare_images(imageA,imageB,imageC,writer,iteration)

def onormalize(original_kspace, mean, std, eps=1e-11):
    #getting image from masked data
    image = transforms.ifft2(original_kspace)
    #normalizing the image
    nimage = transforms.normalize(image, mean, std, eps=1e-11)
    #getting kspace data from normalized image
    original_kspace_fni = transforms.ifftshift(nimage, dim=(-3, -2))
    original_kspace_fni = torch.fft(original_kspace_fni, 2)
    original_kspace_fni = transforms.fftshift(original_kspace_fni, dim=(-3, -2)) 
    original_kspace_fni = transforms.normalize(original_kspace, mean, std, eps=1e-11)
    return original_kspace_fni

def mnormalize(masked_kspace):
    #getting image from masked data
    image = transforms.ifft2(masked_kspace)
    #normalizing the image
    nimage, mean, std = transforms.normalize_instance(image, eps=1e-11)
    #getting kspace data from normalized image
    maksed_kspace_fni = transforms.ifftshift(nimage, dim=(-3, -2))
    maksed_kspace_fni = torch.fft(maksed_kspace_fni, 2)
    maksed_kspace_fni = transforms.fftshift(maksed_kspace_fni, dim=(-3, -2))
    maksed_kspace_fni, mean, std = transforms.normalize_instance(masked_kspace, eps=1e-11)
    return maksed_kspace_fni,mean,std

def nkspacetoimage(args, kspace_fni, mean, std, eps=1e-11):
    #nkspace to image
    assert kspace_fni.size(-1) == 2
    image = transforms.ifftshift(kspace_fni, dim=(-3, -2))
    image = torch.ifft(image, 2)
    image = transforms.fftshift(image, dim=(-3, -2))
    #denormalizing the nimage
    image = (image * std)+mean
    image = image[0]
    
    image = transforms.complex_center_crop(image, (args.resolution, args.resolution))
    # Absolute value
    image = transforms.complex_abs(image)
    # Normalize input
    image, mean, std = transforms.normalize_instance(image, eps=1e-11)
    image = image.clamp(-6, 6)
    return image

def normalize(data):
    a = np.array(data[0,:,:,0])**2 + np.array(data[0,:,:,1])**2
    divisor = math.sqrt(a.max())
    data = data/divisor
    return data,divisor

def save_model(args, exp_dir, epoch, model, optimizer, best_dev_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_dev_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir + '/model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir + '/model.pt', exp_dir + '/best_model.pt')
    
def load_model(checkpoint_file, model):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    autoencoder = model
    autoencoder.load_state_dict(checkpoint['model'])
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=args.learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint, autoencoder, optimizer