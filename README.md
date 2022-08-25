This program is open source under the BSD-3 License.
Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and
the following disclaimer.
 
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
and the following disclaimer in the documentation and/or other materials provided with the
distribution.
 
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse
or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

# OpenFWI Benchmarks
> Pytorch Implementation on OpenFWI 2D datasets


## About

This repository officially supports the reproducibility of OpenFWI Benchmarks[[1]](#ref1). The mateirals will evolve with the further development of OpenFWI.

For the time being, it contains the codes for training and testing InversionNet[[2]](#ref2) and VelocityGAN[[3]](#ref3), and covers 10 datasets in __Vel family__, __Fault family__, and __Style family__.

## Prepare Data
First download any dataset from our [website](https://openfwi-lanl.github.io/docs/data.html#vel) and unzip it into your local directory.

### Load a pair of velocity map and seismic data
For any dataset in _Vel, Fault, Style_  family, the data is saved as `.npy` files, each file contains a batch of 500 samples. `datai.npy` refers to the `i-th` sample of seismic data. To load data and check:
```bash
import numpy as np
# load seismic data
seismic_data = np.load('data1.npy')
print(seismic_data.shape) #(500,5,1000,70)
# load velocity map
velocity_map = np.load('model1.npy')
print(velocity_map.shape) #(500,1,70,70)
```

### Prepare training and testing set
Note that there are many ways of organizing training and testing dataset, as long as it is compatible with the [DataLoader module](https://pytorch.org/docs/stable/data.html) in pytorch. Whichever way you choose, please refer to the following table for the train/test split.

| Dataset      | Train / test Split | Corresponding `.npy` files |
| ----------- | ----------- | ------------ |
| Vel Family     | 24k / 6k     | data(model)1-48.npy / data(model)49-60.npy |
| Fault Family   | 48k / 6k     | data(model)1-96.npy / data(model)97-108.npy |
| Style Family   | 60k / 7k     | data(model)1-120.npy / data(model)121-134.npy |


A convenient way of loading the data is to use a `.txt` file containing the _location+filename_ of all `.npy` files, parse each line of the `.txt` file and push to the dataloader. Take **flatvel-A** as an exmaple, we create `flatvel-a-train.txt`, organized as the follows, and same for `flatvel-a-test.txt`. 
```bash
Dataset_directory/data1.npy
Dataset_directory/data2.npy
...
Dataset_directory/data48.npy
```

**To save time, you can download all the text files from the `splitting_files` folder and change to your own directory.**

## Reproduce the OpenFWI Benchmarks
> For InversionNet and VelocityGAN, the current version supports training with a single GPU. For UPFWI and InversionNet3D, multiple-GPU is necessary due to the computation cost. 


### Environment setup
The following packages are required:
- pytorch v1.7.1
- torchvision v0.8.2
- scikit learn
- numpy
- matplotlib (for visualization)

If you other versions of pytorch and torchvision, please make sure they align.

### InversionNet
To train from scratch on Flatvel-A dataset with $\ell_1$ loss,  run the following codes:
```
python train.py -ds flatvel-a -n YOUR_DIRECTORY -m InversionNet -g2v 0 --tensorboard -t flatvel_a_train.txt -v flatvel_a_val.txt
```

`-ds` specifies the dataset, `-n` creates the folder containing the saved model other log files, `-g2v` sets the coefficient of $\ell_2$ loss to be zero, `-t` and `-v` assign the training data and test data loading files.

To continue training from a saved checkpoint, run the following codes:
```
python train.py -ds flatvel-a -n YOUR_DIRECTORY -r CHECKPOINT.PTH -m InversionNet -g2v 0 --tensorboard -t flatvel_a_train.txt -v flatvel_a_val.txt
```

Please refer to the details of the codes if you would like to change other parameters (*learning rate,* etc.). These commands suffice to reproduce the OpenFWI benchmarks.

The last step would be testing, where we include the visualization. Also we borrow the implementation of SSIM metric from [pytorch-ssim](https://github.com/Po-Hsun-Su/pytorch-ssim). Please make sure that `pytorch-ssim.py` and `rainbow256.npy` are placed together with others.

```
python test.py -ds flatvel-a -n YOUR_DIRECTORY -m InversionNet -v flatvel_a_val.txt -r CHECKPOINT.PTH --vis -vb 2 -vsa 3
```

`--vis` enables the visualization and creates a folder with the figures, you may also change the amount of velocity maps by playing with `-vb` and `-vsa`.

### VelocityGAN
The code logic of VelocityGAN is almost identical the that of InversionNet.

To train from scratch on Flatvel-A dataset with $\ell_1$ loss, run the following codes:
```
python gan_train.py -ds flatvel-a -n YOUR_DIRECTORY -m InversionNet -g2v 0 --tensorboard -t flatvel_a_train.txt -v flatvel_a_val.txt
```
To continue training from a saved checkpoint, run the following codes:
```
python gan_train.py -ds flatvel-a -n YOUR_DIRECTORY -r CHECKPOINT.PTH -m InversionNet -g2v 0 --tensorboard -t flatvel_a_train.txt -v flatvel_a_val.txt
```
The command for testing is the same with InversionNet


## Future Updates
- We will release the training configuration of Kimberlina-CO2 dataset very soon.
- We will improve the instruction with illustrations and other necessary details
- The codes of UPFWI and InversionNet3D is pending approval, they will be added to this repo once approved.

## References
<a id="ref1">[1]</a> 
Deng, Chengyuan, et al. "OpenFWI: Benchmark Seismic Datasets for Machine Learning-Based Full Waveform Inversion." arXiv preprint arXiv:2111.02926 (2021).

<a id="ref2">[1]</a> 
Wu, Yue, and Youzuo Lin. "InversionNet: An efficient and accurate data-driven full waveform inversion." IEEE Transactions on Computational Imaging 6 (2019): 419-433.

<a id="ref3">[1]</a> 
Zhang, Zhongping, and Youzuo Lin. "Data-driven seismic waveform inversion: A study on the robustness and generalization." IEEE Transactions on Geoscience and Remote sensing 58.10 (2020): 6900-6913.
