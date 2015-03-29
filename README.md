# bpcpr: Cascade Pose Regression with Back Propagation
A neural network formulation for Cascaded Pose Regression (CPR) applied to Face Alignment. Implement the algorithm in [2] which is almost the same with [1], where the parameters jointly tuned for the whole CPR algorithmic pipeline. See e.g., [3] for original CPR based face alignment.

## Install
1. Install [the Matlab DAG network](https://github.com/pengsun/MatConvDAG) by following the instruction therein.
2. Run the script `setup_path.m`. Note that you need modify the code before running to specify the path of the required third party toolbox.

## Folder Layout
`tf_*.m` and `tfw_*.m`: transformer for the Graph Transformer Network (GTN), a.k.a. DAG netowrk. In this project we view the whole network as a transformer that composites many sub-transformers

`convdag_bpcpr.m`: A thin wrapper of the whole network, managing training, testing, etc.

`peek.m`: the observer (a Design Pattern) for `convdag_bpcpr.m`, managing model saving, training loss plotting on the fly, etc.

`mex`: C and CUDA C code

`util`: helper functions

`script`: scripts for training

`chk_rst`: scripts for results inspection

## References
[1]. Baoguang Shi, Xiang Bai, Wenyu Liu, Jingdong Wang, "Deep Regression for Face Alignment", arXiv:1409.5230, 2014

[2]. Peng Sun, James K. Min, Guanglei Xiong. Globally Tuned Cascade Pose Regression via Back Propagation with Application in 2D Face Pose Estimation and Heart Segmentation in 3D CT Images

[3]. Ren, Shaoqing, Cao, Xudong, Wei, Yichen, and Sun, Jian. Face alignment at 3000 fps via regressing local binary features. CVPR 2014
