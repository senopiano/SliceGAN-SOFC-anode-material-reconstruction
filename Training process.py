'''Project: SliceGAN
   Part: Training
   Author: lwj
   Date: 2024/2/26-2024/2/28'''

import random
import torch.optim as optim

# Set random seed for reproducibility.
seed = 500
random.seed(seed)
torch.manual_seed(seed)

#  Choose the loss fuction
criterion = nn.BCELoss()

# Setup the optimizer
optimD = optim.Adam(netD.parameters() , lr = 0.0001 , betas = (0.9 , 0.999))
optimG = optim.Adam(netG.parameters() , lr = 0.0001 , betas = (0.9 , 0.999))

# Create the noise
fixed_noise = torch.randn(4 , nz , 4 , 4 , 4 , device = device)

# Training process

# Label
real_label = 0.9
fake_label = 0

img_list = []
G_losses = []
D_losses = []
iters = 0

for epoch in range(100):
   for i , data in enumerate(dataloader , 0):
      # Train the netD
      noise = torch.randn(4 , nz , 4 , 4 , 4 , device = device)
      fake = netG(noise).detach()
      for dim , (d1 , d2 ,d3) in enumerate(zip([2, 3, 4], [3, 2, 2], [4, 4, 3])):
         # Use real data
         optimD.zero_grad()                  
         real_cpu = data[0].to(device)
         b_size = real_cpu.size(0)
         # print(real_cpu.size(3))
         label = torch.full((b_size,) , real_label , dtype = torch.float , device = device)     
         output_r = netD(real_cpu).view(-1)
         # print(output_r.shape)                     
         loss_real = criterion(output_r , label)
         # loss_real.backward()
         D_x = output_r.mean().item()
         # Use fake data
         fake_perm = fake.permute(0, d1, 1, d2, d3).reshape(D_batch_size, nc, 64, 64)
         output_f = netD(fake_perm).view(-1)
         label_f = torch.full((D_batch_size,) , fake_label , dtype = torch.float , device = device)
         loss_fake = criterion(output_f , label_f)
         # loss_fake.backward()
         D_G_z1 = output_f.mean().item()
         loss_D = loss_real + loss_fake
         loss_D.backward()
         optimD.step()

      # Train the netG
      optimG.zero_grad()
      noise = torch.randn(8 , nz , 4 , 4 , 4 , device = device)
      fake_data = netG(noise)
      loss_G = 0
      for dim , (d1 , d2 ,d3) in enumerate(zip([2, 3, 4], [3, 2, 2], [4, 4, 3])):         
         fake_data_perm = fake_data.permute(0, d1, 1, d2, d3).reshape(batch_size, nc, 64, 64)
         output_G = netD(fake_data_perm).view(-1)
         label_f = torch.full((batch_size,) , real_label , dtype = torch.float , device = device)
         loss_G = loss_G + criterion(output_G , label_f)
         D_G_z2 = output_G.mean().item()
      loss_G.backward()
      optimG.step()
      
      if i % 500 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, 100, i, len(dataloader),
                     loss_D.item(), loss_G.item(), D_x, D_G_z1, D_G_z2))
      G_losses.append(loss_G.item())
      D_losses.append(loss_D.item())
             
      iters += 1
   if epoch % 10 == 0:
      torch.save(netG.state_dict() , 'netG_epoch_{}.pt'.format(epoch))   

torch.save(netG.state_dict() , 'netG.pt')
torch.save(netD.state_dict() , 'netD.pt')