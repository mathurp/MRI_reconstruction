import utils
import torch.nn as nn
from torch.nn import functional as F
import torch
from torch.autograd import Variable
import argparse
from tensorboardX import SummaryWriter

# **The Denoising Autoencoder**

parser = argparse.ArgumentParser()

parser.add_argument("--data-path", help="The path of the directory to the dataset", required=True)
parser.add_argument("--mode", help="Choice of space to find the nearest neighbour", choices=['image', 'kspace'], default='image')
parser.add_argument("--center-fractions", nargs='+', default=[0.08, 0.04])
parser.add_argument("--accelerations", nargs='+', default=[4, 8])
parser.add_argument("--resolution", default=320, type=int)
parser.add_argument("--sample-rate", default=1.)
parser.add_argument("--challenge", default="singlecoil", choices=["singlecoil", "multicoil"])
parser.add_argument("--batch-size", default=1, type=int)
parser.add_argument("--learning-rate", default=0.0001, type=float)
parser.add_argument("--epoch", default=10, type=int)
parser.add_argument("--reluslope", default=0.2, type=float)
parser.add_argument("--checkpoint", default='DAEcheckpoint/best_model.pt')
parser.add_argument("--exp-dir", default='DAEcheckpoint')
parser.add_argument("--resume", default=False, type=bool, choices=[True, False])
args = parser.parse_args()


# args = utils.Arguments(1,'/media/ranka47/9E3C0D4A3C0D1EC1/fastMRI/',[0.08, 0.04],[4, 8],'singlecoil',1.,320)
train_loader, dev_loader = utils.create_data_loaders(args)

kernel_size = 51
padding_size = 25

class DenoisingAutoencoder(nn.Module):
    
    def __init__(self):
    
        super(DenoisingAutoencoder, self).__init__()
                                                            # 640 x 386 x 2 (input)  476160 activations
        #Conv2d(in_channels, out_channels, kernel_size,padding) 
        self.conv1e = nn.Conv2d(2, 14, kernel_size, padding=padding_size)        # 640 x 386 x 14
        self.conv2e = nn.Conv2d(14, 28, kernel_size, padding=padding_size)       # 640 x 386 x 28
        self.mpl1   = nn.MaxPool2d(2)  # 320 x 186 x 28
        self.conv3e = nn.Conv2d(28, 56, kernel_size, padding=padding_size)       # 320 x 193 x 56
        self.mpl2   = nn.MaxPool2d(2)  # 160 x 93 x 56     833280 activations
        
        self.conv4d = nn.ConvTranspose2d(56, 28, kernel_size, padding=padding_size)            # 320 x 192 x 28
        
        self.conv5d = nn.ConvTranspose2d(28, 14, kernel_size, padding=padding_size)            # 640 x 384 x 14   
        
        self.conv6d = nn.ConvTranspose2d(14, 2, kernel_size, padding=padding_size)             # 640 x 384 x 2
        
        self.conv7d = nn.ConvTranspose2d(14, 2, kernel_size, padding=(padding_size, padding_size - 2))             # 640 x 386 x 2
        
    
    def forward(self, x):
        # Encoder
        x = self.conv1e(x)
        x = F.leaky_relu(x,negative_slope=args.reluslope)
        
        x = self.conv2e(x)
        x = F.leaky_relu(x,negative_slope=args.reluslope)
        
        x = self.mpl1(x)
        
        x = self.conv3e(x)
        x = F.leaky_relu(x,negative_slope=args.reluslope)
        
        temp = x.shape[3]
        x = self.mpl2(x)
        
        # Decoder
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        x = self.conv4d(x)
        x = F.leaky_relu(x,negative_slope=args.reluslope)
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        x = self.conv5d(x)
        x = F.leaky_relu(x,negative_slope=args.reluslope)
        
        if temp%2==0:
            x = self.conv6d(x)
        else:
            x = self.conv7d(x)
        
        x = F.leaky_relu(x,negative_slope=args.reluslope)
        
        return x

train_loss = []
valid_loss = []
# output_file = open("log.txt", "w", buffering=1)
print('Total number of epochs: %d\n' % args.epoch)
print('Total number of training iterations: %d\n' % len(train_loader))
print('Total number of validation iterations: %d\n' % len(dev_loader))

# autoencoder = DenoisingAutoencoder()
writer = SummaryWriter(log_dir=args.exp_dir+'/summary')
autoencoder = DenoisingAutoencoder().cuda()
parameters = list(autoencoder.parameters())
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)

print("Model's state_dict:")
for param_tensor in autoencoder.state_dict():
    print(param_tensor, "\t", autoencoder.state_dict()[param_tensor].size())

print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

if args.resume:
    checkpoint, autoencoder, optimizer = utils.load_model(args.checkpoint, autoencoder)
    args = checkpoint['args']
    best_val_loss = checkpoint['best_val_loss']
    start_epoch = checkpoint['epoch']
    del checkpoint
else:
    best_val_loss = 1e9
    start_epoch = 0

for i in range(start_epoch,args.epoch):
    print("Epoch: ",i+1)
    # model training
    total_loss = 0.0
    total_val_loss = 0.0
    global_step = i * len(train_loader)
    
    autoencoder.train()
    for j,data in enumerate(train_loader):
        original_kspace,masked_kspace,target, mask = data
        #normalizing the kspace
        nmasked_kspace,mdivisor = utils.normalize(masked_kspace)
        noriginal_kspace,odivisor = utils.normalize(original_kspace)
        
        #transforming the input according dimention and type 
        noriginal_kspace,nmasked_kspace = utils.transformshape(noriginal_kspace), utils.transformshape(nmasked_kspace)
        nmasked_kspace = Variable(nmasked_kspace).cuda()
        noriginal_kspace = Variable(noriginal_kspace).cuda()
        
        #setting up all the gradients to zero
        optimizer.zero_grad()
        
        #forward pass
        outputkspace = autoencoder(nmasked_kspace)        
        #finding the loss
        loss = loss_func(outputkspace, noriginal_kspace)
        #backward pass
        loss.backward()
        optimizer.step()
        
        if j % 1000 == 0:
            avg_loss = loss.data.item()/(j+1)
            print('Avg training loss: ',avg_loss,' Training loss: ',loss.data.item(), ' iteration :', j+1)
            
        if j % 5000 == 0:
            utils.compareimageoutput(target,masked_kspace,outputkspace,mask,writer,global_step + j+1, args)
        
        total_loss += loss.data.item()
        writer.add_scalar('TrainLoss', loss.data.item()*1000, global_step + j+1)
        
    train_loss.append(total_loss/len(train_loader))
    
    # validation loss
    global_step = i * len(dev_loader)
    autoencoder.eval()
    for j,data in enumerate(dev_loader):
        original_kspace,masked_kspace,target, mask = data
        #normalizing the kspace
        nmasked_kspace,mdivisor = utils.normalize(masked_kspace)
        noriginal_kspace,odivisor = utils.normalize(original_kspace)
        
        #transforming the input according dimention and type 
        noriginal_kspace,nmasked_kspace = utils.transformshape(noriginal_kspace), utils.transformshape(nmasked_kspace)
        nmasked_kspace = Variable(nmasked_kspace).cuda()
        noriginal_kspace = Variable(noriginal_kspace).cuda()
        
        #forward pass
        outputkspace = autoencoder(nmasked_kspace)        
        #finding the loss
        loss = loss_func(outputkspace, noriginal_kspace)
        
        writer.add_scalar('ValidationLoss', loss.data.item()*1000, global_step + j+1)
        
        if (j) % 1000 == 0:
            avg_loss = loss.data.item()/(j+1)
            print('Avg ValidationLoss loss: ',avg_loss,' Validation loss: ',loss.data.item(), ' iteration :', j+1)
            
        if (j) % 5000 == 0:
            utils.compareimageoutput(target,masked_kspace,outputkspace,mask,writer,global_step + j+1, args)
        
        total_val_loss += loss.data.item()
    valid_loss.append(total_val_loss / len(dev_loader))
    
    print('saving')
    is_new_best = valid_loss[-1] < best_val_loss
    best_val_loss = min(best_val_loss, valid_loss[-1])
    print("best val loss :",best_val_loss)
    utils.save_model(args, args.exp_dir, i, autoencoder, optimizer, best_val_loss, is_new_best)
writer.close()
    


# # # The code below is just for person use (inference and testing)

# np.savetxt('originalkspace_a.txt', original_kspace[0,:,:,0])
# np.savetxt('originalkspace_b.txt', original_kspace[0,:,:,1])

# ####################
# i = original_kspace[0,:,:,0]
# p = original_kspace[0,:,:,1]
# plt.plot(np.array(i))
# plt.show()
# plt.plot(np.array(p))
# plt.show()

# i = noriginal_kspace[0,0,:,:]
# p = noriginal_kspace[0,1,:,:]
# plt.plot(np.array(i))
# plt.show()
# plt.plot(np.array(p))
# plt.show()

# i = masked_kspace[0,0,:,:].data.cpu()
# p = masked_kspace[0,1,:,:].data.cpu()
# plt.plot(np.array(i))
# plt.show()
# plt.plot(np.array(p))
# plt.show()

# i = nmasked_kspace[0,0,:,:].data.cpu()
# p = nmasked_kspace[0,1,:,:].data.cpu()
# plt.plot(np.array(i))
# plt.show()
# plt.plot(np.array(p))
# plt.show()


# np.sum(np.array(original_kspace[0,:,:,0]) == 0)


# for i in range(epoch):
#     print("Epoch: ",i)
#     # model training
#     total_loss = 0.0
#     total_iter = 0

#     autoencoder.train()
#     for i,data in enumerate(train_loader):
#         original_kspace,masked_kspace,target, mask = data
#         #noriginal_kspace,nmasked_kspace = normalize(original_kspace),normalize(masked_kspace)
#         original_width = original_kspace.shape[2]
        
#         inputdata = Variable(transformshape(original_kspace)).cuda()
#         masked_kspace = Variable(transformshape(masked_kspace)).cuda()
#         optimizer.zero_grad()
        
#         outputdata = autoencoder(masked_kspace)
#         plotimage(kspacetoimage(original_kspace, args)[0])
#         break
#     break

# ####################
# unmask = np.where(mask==1.0, 0.0, 1.0)
# unmask = transforms.to_tensor(unmask)
# unmask = unmask.float()
# output = transformback(outputkspace.data.cpu())

# list(autoencoder.parameters())

# list(autoencoder.parameters())

# train_loss = []
# valid_loss = []
# print('Total number of epochs:', epoch)
# print('Total number of training iterations: ',len(train_loader))
# print('Total number of validation iterations: ',len(dev_loader))

# for i in range(epoch):
#     print("Epoch: ",i)
#     # model training
#     total_loss = 0.0
#     total_iter = 0

#     autoencoder.train()
#     for i,data in enumerate(train_loader):
#         original_kspace,masked_kspace,target, mask = data
#         original_width = original_kspace.shape[2]
        
#         inputdata = Variable(transformshape(original_kspace)).cuda()
#         masked_kspace = Variable(transformshape(masked_kspace)).cuda()
#         optimizer.zero_grad()
        
#         outputdata = autoencoder(masked_kspace)
#         plotimage(kspacetoimage(original_kspace, args))
#         break
#     break

# plt.plot(np.unique(np.array(noriginal_kspace.data.cpu())))

# np.unique(np.array(outputkspace.data.cpu()))

# outputkspace = autoencoder(nmasked_kspace)

# loss_func(outputkspace, noriginal_kspace)

# compareimageoutput(target,masked_kspace,outputkspace,mask)

# ####################
# plotimage(kspacetoimage(transforms.to_tensor(transformback(transformshape(outputdata)[:,:,:,1:1+original_width])), args))
# max=0
# v=0
# for i,data in enumerate(train_loader):
#     original_kspace,masked_kspace,target, mask = data
#     s = original_kspace.shape
#     if s[2]>max:
#         max = s[2]
#         v = s
# print(max)
# print(v)
# plt.imshow(np.array(masked_kspace[0,250:400,150:250,0]))
# plt.show()
# plt.imshow(np.array(original_kspace[0,250:400,150:250,0]))
# plt.show()
# print(np.array(mask).shape)

# ####################
# # Save the model
# torch.save(autoencoder.state_dict(), "./5.autoencoder.pth")

# ####################
# fig = plt.figure(figsize=(10, 7))
# plt.plot(train_loss, label='Train loss')
# plt.plot(valid_loss, label='Validation loss')
# plt.legend()
# plt.show()

# ####################
# import random

# img, _ = random.choice(cifar10_valid)
# img    = img.resize_((1, 3, 32, 32))
# noise  = torch.randn((1, 3, 32, 32)) * noise_level
# img_n  = torch.add(img, noise)

# img_n = Variable(img_n).cuda()
# denoised = autoencoder(img_n)


# show_img(img[0].numpy(), img_n[0].data.cpu().numpy(), denoised[0].data.cpu().numpy())

# # Visualize the first image of the last batch in our validation set
#     orig = image[0].cpu()
#     noisy = image_n[0].cpu()
#     denoised = output[0].cpu()

#     orig = orig.data.numpy()
#     noisy = noisy.data.numpy()
#     denoised = denoised.data.numpy()

#     print("Iteration ", i+1)
#     show_img(orig, noisy, denoised)

# ####################
# learning_rate = 0.001
# epoch = 10
# class DenoisingAutoencoder(nn.Module):
    
#     def __init__(self):
    
#         super(DenoisingAutoencoder, self).__init__()
#                                                             # 640 x 372 x 2 (input)  476160 activations
#         self.conv1e = nn.Conv2d(2, 24, 3, padding=2)        # 640 x 372 x 24
#         self.conv2e = nn.Conv2d(24, 48, 3, padding=2)       # 640 x 372 x 48
#         self.conv3e = nn.Conv2d(48, 96, 3, padding=2)       # 640 x 372 x 96
#         self.conv4e = nn.Conv2d(96, 128, 3, padding=2)      # 640 x 372 x 128
#         self.conv5e = nn.Conv2d(128, 256, 3, padding=2)     # 640 x 372 x 256
#         self.mp1e   = nn.MaxPool2d(2, return_indices=True)  # 640 x 372 x 256       60948480 acivations

#         self.mp1d = nn.MaxUnpool2d(2)
#         self.conv5d = nn.ConvTranspose2d(256, 128, 3, padding=2)
#         self.conv4d = nn.ConvTranspose2d(128, 96, 3, padding=2)
#         self.conv3d = nn.ConvTranspose2d(96, 48, 3, padding=2)
#         self.conv2d = nn.ConvTranspose2d(48, 24, 3, padding=2)
#         self.conv1d = nn.ConvTranspose2d(24, 3, 2, padding=2)
    
#     def forward(self, x):
#         # Encoder
#         x = self.conv1e(x)
#         x = F.relu(x)
#         x = self.conv2e(x)
#         x = F.relu(x)
#         x = self.conv3e(x)
#         x = F.relu(x)
#         x = self.conv4e(x)
#         x = F.relu(x)
#         x = self.conv5e(x)
#         x = F.relu(x)
#         x, i = self.mp1e(x)
        
#          # Decoder
#         x = self.mp1d(x, i)
#         x = self.conv5d(x)
#         x = F.relu(x)
#         x = self.conv4d(x)
#         x = F.relu(x)
#         x = self.conv3d(x)
#         x = F.relu(x)
#         x = self.conv2d(x)
#         x = F.relu(x)
#         x = self.conv1d(x)
#         x = F.relu(x)
        
#         return x

# autoencoder = DenoisingAutoencoder().cuda()
# parameters = list(autoencoder.parameters())
# loss_func = nn.MSELoss()
# optimizer = torch.optim.Adam(parameters, lr=learning_rate)

# self.conv1e = nn.Conv2d(2, 14, 3, padding=1)        # 640 x 372 x 14
#         self.conv2e = nn.Conv2d(14, 28, 3, padding=1)       # 640 x 372 x 28
#         self.mpl1   = nn.MaxPool2d(2, return_indices=True)  # 320 x 186 x 28
#         self.conv3e = nn.Conv2d(28, 56, 3, padding=1)       # 320 x 186 x 56
#         self.mpl2   = nn.MaxPool2d(2, return_indices=True)  # 160 x 93 x 56     833280 activations
        
#         self.conv4d = nn.ConvTranspose2d(56, 28, 3, padding=1)            # 320 x 186 x 28
#         self.conv5d = nn.ConvTranspose2d(28, 14, 3, padding=1)            # 640 x 372 x 14   
#         self.conv6d = nn.ConvTranspose2d(14, 2, 3, padding=1)             # 640 x 372 x 2
        
#         def forward(self, x):
#         # Encoder
#         x = self.conv1e(x)
#         x = F.relu(x)
        
#         x = self.conv2e(x)
#         x = F.relu(x)
        
#         x, i = self.mpl1(x)
        
#         x = self.conv3e(x)
#         x = F.relu(x)
        
#         x, j = self.mpl2(x)
        
#         # Decoder
        
#         x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
#         x = self.conv4d(x)
#         x = F.relu(x)
        
#         x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
#         x = self.conv5d(x)
#         x = F.relu(x)
        
#         x = self.conv6d(x)
#         x = F.relu(x)
        
#         return x

# def explainmasking():
#     masked = original_kspace*mask
#     maskedt = original_kspace*unmask
#     total = masked + maskedt
#     plotimage(kspacetoimage(masked, args))
#     plotimage(kspacetoimage(maskedt, args))
#     plotimage(kspacetoimage(total, args))

