# SliceGAN-SOFC-anode-material-reconstruction
In this project, we use SliceGAN to reconstruct the SOFC anode material from 2D images to 3D.
## Source Images
The data set is composed of a big SEM picture which has already been segmented. You can directly visit https://edx.netl.doe.gov/dataset/sofc-microstructures-hsu-epting-mahbub-jps-2018 to have a try. Also, you can download the original pictures and then processed by *Avizo*.

![image](https://github.com/senopiano/SliceGAN-SOFC-anode-material-reconstruction/blob/main/anode.png)
## SliceGAN network
You can download Network.py to see the code. The structure of the whole network is shown as below.

![image](https://github.com/senopiano/SliceGAN-SOFC-anode-material-reconstruction/blob/main/sliceGAN%20structure.png)
## Generated images
After the training process, we can use the netG to generate pictures. The input is a random Gaussian noise and the output is a 3D array. Turning the array to a series of RGB pictures, you can visualize these pictures with some apps like *Avizo* / *ImageJ*...
The following 6 images are generated by the trained netG. These are slices of the 3D array of 3 directions. Slice directions of the colorful pictures are *x*, *y*, *z* respectively.

![image](https://github.com/senopiano/SliceGAN-SOFC-anode-material-reconstruction/blob/main/output.png)
## Determination of porosity
Input the generated images into *Avizo*. Apply the functions of threshold segmentation, filtering noise reduction and porosity measurement. The result is shown as below.
| Source |  Porosity(%) |
| ---------- | :-----------:  |
| Generated | 20.48 |
|  Original | 20.63 |
|   Error   | 0.73% |
