# ECE1512 2022F Project B - Enrique Torres Sanchez
Project B contains the implementation details and result files for the Dataset Distillation: A Data-Efficient Learning Framework project for course ECE1512: Digital Image Processing

The Task1 folder contains the implementations for the following sections:
- Task1_2a_3.py contains the implementation of both from scratch vanilla and from scratch synthetic dataset training, with FLOPs per second and compute time measuring
- Task1_2bcde.py contains the implementation from the GitHub repo https://github.com/VICO-UoE/DatasetCondensation, to generate synthetic datasets based on CIFAR10 and MNIST, both from randomly sampled original images and from Gaussian noise. The implementation also provides a way to store a trainable version of the synthetic dataset, as well as a visualization of the generated synthetic images.
- Task1_4.py contains an implementation of a simulated continual learning scenario, where a model is first given half of the classes of the CIFAR10 dataset, then the other half. Two versions of the model are evaluated: 1) as is after seeing the two halfs of the dataset sequentially; 2) trained on the synthetic CIFAR10 dataset after seeing the two halfs of the original dataset sequentially

The implementation given on Task1_2bcde.py is based on the paper by Zhao et al. (https://arxiv.org/abs/2006.05929)

The Task2 folder contains the implementation for the following sections:
- Task2_1_distill.py contains the implementation from the GitHub repo https://github.com/GeorgeCazenavette/mtt-distillation, to generate synthetic datasets based on CIFAR10 and MNIST, both from randomly sampled original images and from Gaussian noise. The implementation also provides a way to store a trainable version of the synthetic dataset, as well as a visualization of the generated synthetic images.
- Task2_1.py contains the implementation of both from scratch vanilla and from scratch synthetic dataset training, with FLOPs per second and compute time measuring

The implementation given on Task2_1_distill.py is based on the paper by Cazenavette et al. (https://arxiv.org/abs/2203.11932)