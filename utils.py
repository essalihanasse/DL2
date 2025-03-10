import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_energy(v, h, W, b, c):
    return -np.dot(v, b) - np.dot(h, c) - np.dot(np.dot(v, W), h)

def sample_bernoulli(probs):
    return (probs > np.random.random(probs.shape)).astype(float)

def entree_sortie_RBM(v, W, c):
    return sigmoid(np.dot(v, W) + c)

def sortie_entree_RBM(h, W, b):
    return sigmoid(np.dot(h, W.T) + b)

def gibbs_step(v, W, b, c):
    h_probs = entree_sortie_RBM(v, W, c)
    h = sample_bernoulli(h_probs)
    v_probs = sortie_entree_RBM(h, W, b)
    v_new = sample_bernoulli(v_probs)
    return v_new, h, h_probs


def lire_alpha_digit(data,L=list(range(10))):
    X=data['dat'][L[0]]
    for i in range(1,len(L)) :
        X_bis=data['dat'][L[i]]
        X=np.concatenate((X,X_bis),axis=0)
    n=X.shape[0]
    X=np.concatenate(X).reshape((n,-1))
    return X
def display_images(images, size):
    for image in images:
        image = image.reshape(size)
        plt.imshow(image, cmap='gray')
        plt.show()
# def load_mnist():
#     """
#     Load MNIST dataset.
    
#     Returns:
#         tuple: (X_train, y_train, X_test, y_test)
#     """
#     # Load data from keras
#     (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    
#     # Reshape and normalize
#     X_train = X_train.reshape(-1, 784).astype('float32') / 255.0
#     X_test = X_test.reshape(-1, 784).astype('float32') / 255.0
    
#     # Binarize
#     X_train = (X_train > 0.5).astype('float32')
#     X_test = (X_test > 0.5).astype('float32')
    
#     # One-hot encode labels
#     y_train_onehot = keras.utils.to_categorical(y_train, 10)
#     y_test_onehot = keras.utils.to_categorical(y_test, 10)
    
#     return X_train, y_train_onehot, X_test, y_test_onehot, y_train, y_test