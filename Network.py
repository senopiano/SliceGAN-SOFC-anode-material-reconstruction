'''Project: SliceGAN
   Part: Network
   Author: lwj
   Date: 2024/2/26'''


# Batch size of D & G
batch_size = 8 * 64
D_batch_size = 4 * 64

# Number of workers for dataloader
workers = 2

# The gpu available
ngpu = 1

# Load data to dataloader
dataloader = torch.utils.data.DataLoader(dataset , batch_size = D_batch_size , shuffle = True , num_workers = workers)

# Choose the device
device = torch.device('cuda:0')

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Define the Generator

# Size of the feature maps
ngf = 64

# Size of the input z vector
nz = 100

# Number of channels in the training images
nc = 3

class Generator(nn.Module):
   def __init__ (self , ngpu):
      super(Generator, self).__init__()
      self.ngpu = ngpu
      self.main = nn.Sequential(
         # input is Z, going into a convolution
         nn.ConvTranspose3d(nz, ngf * 8, 4, 2, 2, bias=False),
         nn.BatchNorm3d(ngf * 8),
         nn.ReLU(True),
         # state size. (ngf*8) x 6 x 6 x 6
         nn.ConvTranspose3d(ngf * 8, ngf * 4, 4, 2, 2, bias=False),
         nn.BatchNorm3d(ngf * 4),
         nn.ReLU(True),
         # state size. (ngf*4) x 10 x 10 x 10
         nn.ConvTranspose3d(ngf * 4, ngf * 2, 4, 2, 2, bias=False),
         nn.BatchNorm3d(ngf * 2),
         nn.ReLU(True),
         # state size. (ngf*2) x 18 x 18 x 18
         nn.ConvTranspose3d(ngf * 2, ngf, 4, 2, 2, bias=False),
         nn.BatchNorm3d(ngf),
         nn.ReLU(True),
         # state size. (ngf) x 34 x 34 x 34
         nn.ConvTranspose3d(ngf, nc, 4, 2, 3, bias=False),
         nn.Softmax(dim=1)
         # state size. (nc) x 64 x 64 x 64
      )
   def forward(self , x):
      return self.main(x)

# Define the Discriminator
   
# Size of feature maps in Discriminator
ndf = 64

class Discriminator(nn.Module):
   def __init__(self, ngpu):
      super(Discriminator, self).__init__()
      self.ngpu = ngpu
      self.main = nn.Sequential(
         # input is (nc) x 64 x 64
         nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
         nn.BatchNorm2d(ndf),
         nn.LeakyReLU(0.2, inplace=True),
         # state size. (ndf) x 32 x 32
         nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
         nn.BatchNorm2d(ndf * 2),
         nn.LeakyReLU(0.2, inplace=True),
         # state size. (ndf*2) x 16 x 16
         nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
         nn.BatchNorm2d(ndf * 4),
         nn.LeakyReLU(0.2, inplace=True),
         # state size. (ndf*4) x 8 x 8
         nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
         nn.BatchNorm2d(ndf * 8),
         nn.LeakyReLU(0.2, inplace=True),
         # state size. (ndf*8) x 4 x 4
         nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
         nn.Sigmoid()
      )

   def forward(self, x):
      return self.main(x)

netG = Generator(ngpu).to(device)    
netD = Discriminator(ngpu).to(device)
netG.apply(weights_init)
netD.apply(weights_init)