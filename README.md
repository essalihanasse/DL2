# Deep Learning Project: RBM, DBN, and DNN Implementation

This project implements Restricted Boltzmann Machines (RBM), Deep Belief Networks (DBN), and Deep Neural Networks (DNN) for classification of handwritten digits from the MNIST dataset. It also includes analysis on the Binary AlphaDigits dataset for validation of the generative models.

## Requirements

- Python 3.8+
- NumPy
- TensorFlow 2.x
- Matplotlib
- SciPy

## Dataset Setup

Before running the code, download the required datasets:

1. **Binary AlphaDigits**: Download from [Kaggle](https://www.kaggle.com/datasets/angevalli/binary-alpha-digits?select=binaryalphadigs.mat) and place the `binaryalphadigs.mat` file in the project root directory.

2. **MNIST**: The MNIST dataset will be automatically downloaded through TensorFlow/Keras.

## Project Structure

- `main.py`: Main script that runs the complete analysis for both datasets
- `models.py`: Implementations of RBM, DBN, and DNN models
- `utils.py`: Utility functions for data loading and model operations
- `test_rbm_dbn.py`: Test script for RBM and DBN models on Binary AlphaDigits
- `main_dnn_mnist.py`: Detailed analysis of DNN performance on MNIST dataset

## How to Run

### Basic Usage

Run the complete analysis:

```bash
python main.py
```

### Specific Analyses

Run only the Binary AlphaDigits analysis:

```bash
python main.py --alphadigits
```

Run only the MNIST analysis:

```bash
python main.py --mnist
```

### Individual Tests

Test RBM and DBN on Binary AlphaDigits:

```bash
python test_rbm_dbn.py
```

Run the detailed DNN analysis on MNIST:

```bash
python main_dnn_mnist.py
```

## Output

The analysis generates various plots and visualizations that are saved in the `results` directory:

1. **RBM/DBN Image Generation**: Generated images from trained RBM and DBN models
2. **Error Analysis**: Plots showing error rates for different model configurations:
   - Error rate vs number of hidden layers
   - Error rate vs number of neurons per layer
   - Error rate vs number of training samples

## Model Implementations

### Restricted Boltzmann Machine (RBM)

The RBM implementation includes:
- Contrastive Divergence (CD-k) learning algorithm
- Gibbs sampling for generating samples
- Visualization of learned features

### Deep Belief Network (DBN)

The DBN implementation includes:
- Greedy layer-wise pre-training
- Multiple RBM layers stacked together
- Top-down generative sampling

### Deep Neural Network (DNN)

The DNN implementation includes:
- Pre-training with DBN
- Fine-tuning with backpropagation
- Softmax classification layer

## Results

The main goal of this project is to compare the performance of:
1. DNNs pre-trained with unsupervised learning (RBM/DBN)
2. DNNs initialized randomly

The comparison is done across different configurations:
- Varying number of hidden layers
- Varying number of neurons per layer
- Varying amount of training data

The results demonstrate the advantage of unsupervised pre-training, especially with limited training data or deeper architectures.