# MHE Layers

This repository contains the code for the numerical example in the paper ![Learning-based Moving Horizon Estimation through Differentiable Convex Optimization Layers](https://arxiv.org/abs/2109.03962) presented at the ![4th Learning for Dynamics & Control](https://l4dc.stanford.edu/) conference.
The paper presents an formulation of an MHE as a differentiable optimization layer, allowing for combined state estimation and online estimator tuning. 
The implementation is based on ![Pytorch](https://pytorch.org/), the ![CvxpyLayers](https://github.com/cvxgrp/cvxpylayers) package, and uses ![pytorch-sqrtm](https://github.com/steveli/pytorch-sqrtm) to enable differentation through matrix square-roots. The implementation was tested using Python3.8 with packages as included in the requirements.txt file.

## Setup Instructions
* Clone the repository with submodule:

        git clone --recurse-submodules git@github.com:IntelligentControlSystems/mhe-layers.git

* Setup a virtual environment with requirements as stated in requirements.txt:

        python3.8 -m venv mhe-layers-env
        source mhe-layers-env/bin/activate
        cd mhe-layers
        pip install -r requirements.txt
        
* Open the ipython notebook:

        jupyter notebook numerical-example.ipynb
 
## Abstract
To control a dynamical system it is essential to obtain an accurate estimate of the current system state based on uncertain sensor measurements and existing system knowledge.
 An optimization-based moving horizon estimation (MHE) approach uses a dynamical model of the system, and further allows for integration of physical constraints on system states and uncertainties, to obtain a trajectory of state estimates.
 In this work, we address the problem of state estimation in the case of constrained linear systems with parametric uncertainty.
 The proposed approach makes use of differentiable convex optimization layers to formulate an MHE state estimator for systems with uncertain parameters.
 This formulation allows us to obtain the gradient of a squared and regularized output error, based on sensor measurements and state estimates, with respect to the current belief of the unknown system parameters.
 The parameters within the MHE problem can then be updated online using stochastic gradient descent (SGD) to improve the performance of the MHE.
 In a numerical example of estimating temperatures of a group of manufacturing machines, we show the performance of tuning the unknown system parameters and the benefits of integrating physical state constraints in the MHE formulation.

## Reference

Simon Muntwiler, Kim P. Wabersich, and Melanie N. Zeilinger. Learning-based Moving Horizon Estimation through Differentiable Convex Optimization Layers. In *Proceedings of the 4th Conference on Learning for Dynamics and Control*, vol 168. Proceedings of Machine Learning Research, 2020.
