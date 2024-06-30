'''Project: SliceGAN
   Part: Postprocessing
   Author: lwj
   Date: 2024/2/28'''


import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn

# Set random seed for reproducibility.
seed = 500
random.seed(seed)
torch.manual_seed(seed)
# Test the model
m_state_dict = torch.load('netG.pt')
new_model = Generator(ngpu).to(device)
new_model.load_state_dict(m_state_dict)
fixed_noise = torch.randn(4 , nz , 4 , 4 , 4 , device = device)
predict = new_model(fixed_noise)
# Output the generated figure
fig = plt.figure()
axes = []
i = 0
for dim , (d1 , d2 ,d3) in enumerate(zip([3, 2, 4], [2, 3, 2], [4, 4, 3])):
    predict_show = predict.permute(0, d1, 1, d2, d3).reshape(64*4, nc, 64, 64)
    device = 'cpu'
    predict_show = predict_show.to(device)
    predict_show = predict_show.detach().numpy()
    phase2 = np.zeros([64 , 64 , 64])
    phase3 = np.zeros([64 , 64 , 64])
    p1 = np.array(predict_show[0][0])
    p2 = np.array(predict_show[0][1])
    p3 = np.array(predict_show[0][2])
    phase2[(p2 > p1) & (p2 > p3)] = 128  # spheres, grey
    phase3[(p3 > p2) & (p3 > p1)] = 255  # binder, white
    output_img = np.int_(phase2+phase3)
    axes.append(fig.add_subplot(2 , 2 , i + 1))
    plt.imshow(output_img[: , : , 45])
    i += 1