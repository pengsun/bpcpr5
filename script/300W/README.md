This folder contains the scripts for training on dataset 300W.

### Scripts for Training
`tr_xxx_xxx.m` are the files that really do the training. Currently the best results are got with `tr_ubuntu_sumell_3.m`. See the paper TODO.

### Data Protocol
Training data are stored in a Matlab `*.mat` file including the image `I` and the pose `p`:
#### The image
`I`: size = [H,W,3,N] where H the height, W the width, N the total number of training instances. For the image `j`, `I(:,:,1,j)` is the gray scale image, `I(:,:,2,j)` and `I(:,:,3,j)` are the x-direction (along the width) and y-direction (along the height) gradients (**BE CAREFUL, IT'S NOT A COLOR IMAGE!**), respectively. In our experiments we got the two gradient maps by simply calling the Matlab function `imgradientxy` by default options. H = W = 192 with normalization beforehand. The pixel value is `single` and normalized to the range `[0,1]`.
#### The pose
`p`: size = [2,L,N] where L the number of landmarks (=68 for dataset 300W), N the number of training instances. For the pose for image `j`, `p(1,:,j)` stores the x-coordinates (along the width) for all the L landmarks, while `p(2,:,j)` stores the y-coordinates. The coordinate value is `single` and normalized to the range `[0,1]`.
