import numpy as np
from utils import *
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from scipy.io.matlab import loadmat

class RestrictedBoltzmannMachine:
    def __init__(self, n_visible, n_hidden, learning_rate=0.01):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.W = np.random.normal(0, 0.01, (n_visible, n_hidden))
        self.b = np.zeros(n_visible)
        self.c = np.zeros(n_hidden) 
    
    def train(self, data, n_epochs=100, batch_size=10, k=1):
        n_samples = data.shape[0]
        
        for epoch in range(n_epochs):
            np.random.shuffle(data)
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                batch = data[i:i+batch_size]
                self._contrastive_divergence(batch, k)
    
    def _contrastive_divergence(self, batch, k):
        """
        Perform k-step contrastive divergence update.
        
        Args:
            batch (ndarray): Mini-batch of training data
            k (int): Number of Gibbs sampling steps
        """
        # Positive phase
        h_probs = entree_sortie_RBM(batch, self.W, self.c)
        pos_associations = np.dot(batch.T, h_probs)
        
        # Negative phase / Gibbs sampling
        v_model = batch.copy()
        for _ in range(k):
            v_model, h_model, h_probs_model = gibbs_step(v_model, self.W, self.b, self.c)
        
        neg_associations = np.dot(v_model.T, h_probs_model)
        
        # Update parameters
        self.W += self.learning_rate * (pos_associations - neg_associations) / batch.shape[0]
        self.b += self.learning_rate * np.mean(batch - v_model, axis=0)
        self.c += self.learning_rate * np.mean(h_probs - h_probs_model, axis=0)
    def generate_image_RBM(self, n_images, n_iter, size_img=None):
        v = np.random.binomial(1, 0.5, (n_images, self.n_visible))
        for _ in range(n_iter):
            v,_,_=gibbs_step(v,self.W,self.b, self.c)
        if size_img is not None:
            plt.figure(figsize=(12, 3))
            for i in range(n_images):
                plt.subplot(1, n_images, i+1)
                plt.imshow(v[i].reshape(size_img), cmap='gray')
                plt.axis('off')
            plt.tight_layout()
            plt.show()
    
        return v


    
class DeepBeliefNetwork:
    def __init__(self, hidden_sizes, n_visible, learning_rate=0.01):
        self.sizes = [n_visible] + hidden_sizes
        self.n_visible = n_visible
        self.learning_rate = learning_rate
        self.rbms = []
        
    def train(self, data, num_epochs):
        current_input = data
        
        # Train each RBM layer
        for i in range(len(self.sizes) - 1):
            print(f"Training RBM layer {i+1}")
            
            # Initialize and train current RBM
            rbm = RestrictedBoltzmannMachine(
                n_visible=self.sizes[i],
                n_hidden=self.sizes[i+1],
                learning_rate=self.learning_rate
            )
            
            rbm.train(current_input, n_epochs=num_epochs)
            self.rbms.append(rbm)
            
            # Transform data for next layer
            current_input = entree_sortie_RBM(
                current_input, 
                rbm.W, 
                rbm.c
            )
            
        # Store final weights and biases
        self.Ws = [rbm.W for rbm in self.rbms]
        self.cs = [rbm.c for rbm in self.rbms]
    def generer_img_DBN(self,n_images=5, n_steps=1000, size_img=None):
        top_rbm = self.rbms[-1]
        h = np.random.binomial(1, 0.5, (n_images, top_rbm.n_hidden))
        v = sample_bernoulli(sigmoid(np.dot(h, top_rbm.W.T) + top_rbm.b))
        for _ in range(n_steps):
            h_probs = sigmoid(np.dot(v, top_rbm.W) + top_rbm.c)
            h = sample_bernoulli(h_probs)
            v_probs = sigmoid(np.dot(h, top_rbm.W.T) + top_rbm.b)
            v = sample_bernoulli(v_probs)
        for i in range(len(self.rbms) - 2, -1, -1):
            rbm = self.rbms[i]
            v_probs = sigmoid(np.dot(v, rbm.W.T) + rbm.b)
            v = sample_bernoulli(v_probs)
        if size_img is not None:
            plt.figure(figsize=(12, 3))
            for i in range(n_images):
                plt.subplot(1, n_images, i+1)
                plt.imshow(v[i].reshape(size_img), cmap='gray')
                plt.axis('off')
            plt.tight_layout()
            plt.show()
        return v

class DeepNeuralNetworkKeras:
    def __init__(self, hidden_sizes, n_visible, n_classes, learning_rate=0.01):
        self.sizes = [n_visible] + hidden_sizes + [n_classes]
        self.n_layers = len(self.sizes) - 1
        self.learning_rate = learning_rate
        self.model = None
        self.weights_initialized = False
        self._create_model()
    
    def _create_model(self):
        """Create the Keras model with the specified architecture"""
        model = models.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=(self.sizes[0],)))
        
        # Hidden layers
        for i in range(len(self.sizes) - 2):
            model.add(layers.Dense(self.sizes[i+1], activation='sigmoid'))
        
        # Output layer with softmax activation
        model.add(layers.Dense(self.sizes[-1], activation='softmax'))
        
        # Compile the model
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
    
    def pretrain(self, data, num_epochs=100, batch_size=10, k=1):
        """
        Pretrain the DNN using a Deep Belief Network (greedy layer-wise training).
        
        Args:
            data (ndarray): Training data (samples x features)
            num_epochs (int): Number of epochs for each RBM layer
            batch_size (int): Size of mini-batches
            k (int): Number of Gibbs sampling steps for contrastive divergence
        """
        print("Pre-training with DBN...")
        
        # Create a DBN with the same architecture (excluding output layer)
        dbn = DeepBeliefNetwork(
            hidden_sizes=self.sizes[1:-1],
            n_visible=self.sizes[0],
            learning_rate=self.learning_rate
        )
        
        # Train the DBN
        dbn.train(data, num_epochs)
        
        # Get weights and biases from DBN
        weights = []
        biases = []
        
        for i in range(len(dbn.rbms)):
            weights.append(dbn.rbms[i].W)
            biases.append(dbn.rbms[i].c)
        
        # Set weights in Keras model (excluding the output layer)
        for i, (w, b) in enumerate(zip(weights, biases)):
            layer = self.model.layers[i+1]  # +1 because of input layer
            
            # Get current weights from Keras layer
            current_weights = layer.get_weights()
            
            # Set new weights and biases
            layer.set_weights([w, b])
        
        self.weights_initialized = True
        print("Pre-training complete")
    
    def train(self, X, y, n_epochs=100, batch_size=32, validation_data=None, callbacks=None):
        """
        Train the DNN using backpropagation.
        
        Args:
            X (ndarray): Training data
            y (ndarray): Training labels (one-hot encoded)
            n_epochs (int): Number of training epochs
            batch_size (int): Size of mini-batches
            validation_data (tuple): Validation data (X_val, y_val)
            callbacks (list): List of Keras callbacks
            
        Returns:
            history: Keras training history
        """
        history = self.model.fit(
            X, y,
            epochs=n_epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate(self, X, y):
        """
        Evaluate the model on test data.
        
        Args:
            X (ndarray): Test data
            y (ndarray): Test labels (one-hot encoded)
            
        Returns:
            tuple: (loss, accuracy)
        """
        return self.model.evaluate(X, y, verbose=0)
    
    def predict(self, X):
        """
        Generate predictions for input data.
        
        Args:
            X (ndarray): Input data
            
        Returns:
            ndarray: Predicted class probabilities
        """
        return self.model.predict(X, verbose=0)
    
    def save_weights(self, filepath):
        """Save model weights to file"""
        self.model.save_weights(filepath)
    
    def load_weights(self, filepath):
        """Load model weights from file"""
        self.model.load_weights(filepath)

