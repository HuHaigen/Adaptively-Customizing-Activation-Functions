# Adaptively Customizing Activation Functions for Various Layers

> **The Paper Links**: [IEEE](https://ieeexplore.ieee.org/document/9675813).  
> **Authors:** [Haigen Hu](), [Aizhu Liu](), [Qiu Guan](), [Xiaoxin Li](), [Shengyong Chen](), [Qianwei Zhou]()

## Abstract

To enhance the nonlinearity of neural networks and increase their mapping abilities between the inputs and response variables, activation functions play a crucial role to model more complex relationships and patterns in the data. In this work, a novel methodology is proposed to adaptively customize activation functions only by adding very few parameters to the traditional activation functions such as Sigmoid, Tanh, and ReLU. To verify the effectiveness of the proposed methodology, some theoretical and experimental analysis on accelerating the convergence and improving the performance is presented, and a series of experiments are conducted based on various network models (such as AlexNet, VGGNet, GoogLeNet, ResNet and DenseNet), and various datasets (such as CIFAR10, CIFAR100, miniImageNet, PASCAL VOC and COCO) . To further verify the validity and suitability in various optimization strategies and usage scenarios, some comparison experiments are also implemented among different optimization strategies (such as SGD, Momentum, AdaGrad, AdaDelta and ADAMs) and different recognition tasks like classification and detection. The results show that the proposed methodology is very simple but with significant performance in convergence speed, precision and generalization, and it can surpass other popular methods like ReLU and adaptive functions like Swish in almost all experiments in terms of overall performance.

## CNNs for CIFAR10 classification

An implementation of CNNs for CIFAR10 classification using tensorflow. To adapt to the size of CIFAR10, we adjusted some parameters of the network. And it is easy to adapt it on other datasets.

- LeNet
- AlexNet
- VGG16
- GoogLeNet
- ResNet50

## Requirements

- python 3.6.3
- tensorflow 1.13.1
- numpy 1.16.3
- CIFAR10 can be download [here][1]. The path to ‘cifar-10-batches-py’ can be specified with the optional parameter ‘--dataset_dir’, which by default is placed in the root directory.

## Train and Test

Here I only iterate 20 epoches (10000 steps), you can increase the number of iterations by using the last trained model to achieve higher accuracy. Besides, you can also change `learning rate` and `steps` in ‘main.py’.

```shell
# Train and test by default.
$ python main.py

# Train with optional patameters and test.
$ python main.py --model_type [LeNet/AlexNet/VGG16/GoogLeNet/ResNet50] \
                 --dataset_dir [Path to cifar-10-batches-py] \
                 --model_dir [A .ckpt file of pretrained model or A folder for saving model]
```

## Use GPU

- CUDA 8.0.61
- CUDNN 5.1
- tensorflow_gpu 1.2.0

```shell
# Chose GPU to use
$ CUDA_VISIBLE_DEVICES=0 python main.py (optional patameters...)
```

## Logs

```shell
tensorboard --logdir=/logs
```

## Citation

Please cite our paper if you find the work useful:

```
@ARTICLE{Hu2022Adaptively,
  author={Hu, Haigen and Liu, Aizhu and Guan, Qiu and Qian, Hanwang and Li, Xiaoxin and Chen, Shengyong and Zhou, Qianwei},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  title={Adaptively Customizing Activation Functions for Various Layers},
  year={2023},
  volume={34},
  number={9},
  pages={6096-6107},
  keywords={Neural networks;Training;Shape;Adaptive systems;Mathematical models;Learning systems;Deep learning;Adaptable parameters;adaptive activation function;deep learning;various layers},
  doi={10.1109/TNNLS.2021.3133263}
}
```

```
@INPROCEEDINGS{Liu2020Exploring,
  author={Liu, Aizhu and Hu, Haigen and Qiu, Tian and Zhou, Qianwei and Guan, Qiu and Li, Xiaoxin},
  booktitle={2020 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  title={Exploring Optimal Adaptive Activation Functions for Various Tasks},
  year={2020},
  volume={},
  number={},
  pages={2290-2297},
  keywords={Adaptive systems;Task analysis;Biological neural networks;Convergence;Neurons;Shape;Recurrent neural networks;Activation functions;Adaptable parameters;Various tasks},
  doi={10.1109/BIBM49941.2020.9313386}
}
```

[1]: https://www.cs.toronto.edu/~kriz/cifar.html
