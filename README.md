# Learning residual coding for point clouds



 * Authors: Davi Lazzarotto and Touradj Ebrahimi
 * Research group: Multimedia Signal Processing Group (MMSPG)
 * Affiliation: École Polytechnique Fédérale de Lausanne (EPFL)

The source code of the learned residual coding module proposed in [1] is released in this repository. We also release all trained models presented in the paper. 



## Prerequisites

The following software version were employed during code development and are necessary for training and testing the model:

* Python 3.8.10
* TensorFlow 2.5.0
* [TensorFlow Compression 2.2](https://github.com/tensorflow/compression)
* Pyntcloud 0.1.4
* PyYAML 5.4.1
* tqdm 4.59.0

The metrics are computed with the mpeg-pcc-dmetric utility (release 0.13.5) which can be found at the MPEG Gitlab repository at this [link](http://mpegx.int-evry.fr/software/MPEG/PCC/mpeg-pcc-dmetric). An account at http://mpegx.int-evry.fr/software/ and special permission are needed to access this tool. 

## Quickstart

#### First steps

In order to reproduce the results from [1], the source code contained at ```src``` must be executed using the configuration files from ```src/cfg```.  

A subset of the test set used at [1] can be downloaded following the instructions from the **Dataset** section . The folder ```dataset``` should be placed in the root of the repository. Also, the code from the mpeg-pcc-dmetric utility must be built and the generated executable placed in a folder named ```tools``` with the name ```pc_error_d```.

#### Reproducting results with pre-trained models

After installing all dependencies and performing the first steps, go to ```src``` and execute the following command to compress the residual representation for the dataset with all released models and compute the metrics:

```python main.py cfg/compress_and_evaluate/compress_all.yml```

Alternatively, to employ only the residual compression models with the values of lambda selected in Figures 6 and 7 of [1], run:

```python main.py cfg/compress_and_evaluate/compress_selected.yml```

#### Training models from scratch

If you desire to train all models from scratch, please run:

```python main.py cfg/train_and_evaluate/train_all.yml```

Alternatively, to employ only the residual compression models with the values of lambda selected in Figures 6 and 7 of [1], run:

```python main.py cfg/train_and_evaluate/train_selected.yml```

Note that a training dataset containing reference and distorted point cloud blocks should be provided for this effect. Also, the patterns for these blocks partitions should be written at the configuration files from ```src/cfg/train_and_evaluate/trisoup_multi_levels``` and ```src/cfg/train_and_evaluate/octree_multi_levels```.

## Components of this repository

#### Source code

The source code for the residual coding module is entirely contained in the ```src``` directory. A short description of each file can be seen below:

* compress.py: Residual compression and decompression of point cloud blocks and dataset.
* compression_model.py: Neural network architecture of residual coding module.
* dataset.py: Assembling datasets.
* eval_pc.py: Evaluation of bit-rate and distortion of decompressed point cloud models.
* focal_loss.py: Computing the focal loss.

* main.py: Entry point for command line execution of code.
* merge.py: Merging point cloud blocks and binary files into one single file per point cloud.
* pc_io.py: Loading and writing point clouds from disk.
* train.py: Training residual coding module.

#### Configuration files

The configuration files contain information used by the source code to train and test the residual coding module and are all placed at ```src/cfg```. The files under ```compress_and_evaluate``` use the pre-trained models to compress the test set and compute the objective metrics on the distorted point cloud models, while the files under ```train_and_evaluate``` train the models from scratch. 

A short description about each parameter is included in the configuration files. The parameter ```command``` determines the actions to be executed. All possible values for this parameter are listed below:

* multi_config: Execute all the configuration files listed by the parameter ```yml_files```
* train: Train residual coding module.
* compress_one: Compress one point cloud block.
* decompress_one: Decompress one point cloud block.
* compress_decompress_one: Compress and decompress one point cloud block.
* compress_dataset: Compress and decompress all point cloud blocks from test set.
* merge_bin: Merge binary files from point cloud blocks into one file for point cloud model by concatenating them into one bitstream.
* merge_dec: Merge ply files form point cloud blocks into one file for point cloud model by concatenating the points after offsetting them according to the block position.
* eval: Compute the objective metrics on the point clouds from the test set and their bit-rate, writing the values in a csv file.
* train_and_evaluate: Train residual coding module, compress and decompress test set, merge blocks and evaluates the generated point clouds.
* compress_and_evaluate: Compress and decompress test set, merge blocks and evaluates the generated point clouds.

#### Pre-trained models

The residual coding modules trained with the settings presented in [1] are included in the directory ```output```. All models are also accompanied a csv file listing the obtained objective metric values and bit-rates for the test set used in [1].

## Dataset 

The test dataset can be downloaded from the following **FTP** by using dedicated FTP clients, such as FileZilla or FireFTP (we recommend to use [FileZilla](https://filezilla-project.org/)):

Protocol: FTP
FTP address: tremplin.epfl.ch
Username: datasets@mmspgdata.epfl.ch
Password: ohsh9jah4T
FTP port: 21

After you connect, choose the **learned_residual_pcc_test_set** folder from the remote site, and download the relevant material. The total size of the provided data is ~262 MB. 

Please read the README files for further information on the structure and the usage of the material.

## License

This repository is licensed under the GPL-3.0 License. However, some files of the source code were adapted from two other repositories and are therefore licensed with the same license as the adapted code. The repositories, their respective licenses and related source code files are listed below:

* Name: Learning Convolutional Transforms for Point Cloud Geometry Compression
* License: MIT License
* License file: LICENSE-a
* Related files: dataset.py, focal_loss.py, pc_io.py



* Name: TensorFlow Compression
* License: Apache-2.0
* License file: LICENSE-b
* Related files: compression_model.py, train.py

## Conditions of use

If you wish to use this software in your research, we kindly ask you to cite [1].

## References

[1] Davi Lazzarotto and Touradj Ebrahimi "Learning residual coding for point clouds", Proc. SPIE 11842, Applications of Digital Image Processing XLIV, 118420S (1 August 2021);

```
@inproceedings{,
author = {Davi Lazzarotto and Touradj Ebrahimi},
title = {{Learning residual coding for point clouds}},
volume = {11842},
booktitle = {Applications of Digital Image Processing XLIV},
editor = {Andrew G. Tescher and Touradj Ebrahimi},
organization = {International Society for Optics and Photonics},
publisher = {SPIE},
pages = {223 -- 235},
keywords = {Point clouds, Residual coding, Learning-based compression},
year = {2021},
URL = {https://doi.org/10.1117/12.2597814}
}
```

