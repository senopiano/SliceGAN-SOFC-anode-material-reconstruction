# SliceGAN-SOFC-anode-material-reconstruction
In this project, we use SliceGAN to reconstruct the SOFC anode material from 2D images to 3D.
## Source Images
The data set is composed of a big SEM picture which has already been segmented. You can directly visit *https://edx.netl.doe.gov/dataset/sofc-microstructures-hsu-epting-mahbub-jps-2018* to have a try. Also, you can download the original pictures and then processed by *Avizo*.

![image](anode_segmented_tiff_z050.png)
## SliceGAN network
You can download *Network.py* to see the code. The structure of the whole network is shown as below.

![image](https://github.com/senopiano/SliceGAN-SOFC-anode-material-reconstruction/blob/main/sliceGAN%20structure.png)
