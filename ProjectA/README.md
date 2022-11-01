# ECE1512 2022F Project A - Enrique Torres Sanchez
Project A contains the implementation details and result graphs for the Knowledge Distillation for Building Lightweight Deep Learning Models in Visual Classification Tasks project

The Task1 folder contains the Jupyter Notebook implementation of Task1 (Task1.ipynb), knowledge distillation. To fully run the code, the following packages are required:
- Tensorflow (CPU or GPU)
- Tensorflow Datasets
- NumPy
- SciPy
- MatPlotLib
- Seaborn
- Keras FLOPs

The notebook contains implementation details for the original knowledge distillation work (teacher - student model training) by Hinton et al. (https://arxiv.org/abs/1503.02531) as well as the novel work by Kobayashi (https://openaccess.thecvf.com/content/WACV2022/papers/Kobayashi_Extractive_Knowledge_Distillation_WACV_2022_paper.pdf).

The Task2 folder contains the Jupyter Notebook implementation of Task2 (Task2.ipynb), transfer learning and knowledge distillation. To fully run the code, the following packages are required:
- Tensorflow (CPU or GPU)
- Tensorflow Datasets
- NumPy
- SciPy
- MatPlotLib
- Seaborn
- Keras FLOPs

The notebook contains implementation details to perform transfer learning on ResNet50V2 (teacher model) and MobileNetV2 (student model) for the MHIST data-set. Moreover, it includes results graphing for the models' loss and AUC-ROC, and an implementation of transfer learning plus knowledge distillation from the teacher model to the student model.