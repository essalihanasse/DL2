import numpy as np
from utils import (gibbs_step, compute_conditional_hidden,
                   compute_log_likelihood)

class RestrictedBoltzmannMachine:
    def __init__(self, n_visible, n_hidden, learning_rate=0.01):
        """
        Initialize RBM.
        
        Args:
            n_visible (int): Number of visible units
            n_hidden (int): Number of hidden units
            learning_rate (float): Learning rate for gradient ascent
        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        
        # Initialize weights and biases
        self.W = np.random.normal(0, 0.01, (n_visible, n_hidden))
        self.b = np.zeros(n_visible)  # visible bias
        self.c = np.zeros(n_hidden)   # hidden bias
        
        self.log_likelihood_history = []
    
    def train(self, data, n_epochs=100, batch_size=10, k=1):
        """
        Train the RBM using k-step Contrastive Divergence.
        
        Args:
            data (ndarray): Training data (binary)
            n_epochs (int): Number of training epochs
            batch_size (int): Mini-batch size
            k (int): Number of Gibbs sampling steps
        """
        n_samples = data.shape[0]
        
        for epoch in range(n_epochs):
            # Shuffle data
            np.random.shuffle(data)
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                batch = data[i:i+batch_size]
                self._contrastive_divergence(batch, k)
            
            # Compute and store log-likelihood
            if epoch % 10 == 0:
                log_likelihood = float(compute_log_likelihood(data, self.W, self.b, self.c))
                self.log_likelihood_history.append(log_likelihood)
                print(f"Epoch {epoch}, Log-likelihood: {log_likelihood:.2f}")
    
    def _contrastive_divergence(self, batch, k):
        """
        Perform k-step contrastive divergence update.
        
        Args:
            batch (ndarray): Mini-batch of training data
            k (int): Number of Gibbs sampling steps
        """
        # Positive phase
        h_probs = compute_conditional_hidden(batch, self.W, self.c)
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
    
    def generate_samples(self, n_samples=1, n_steps=1000):
        """
        Generate samples from the trained model.
        
        Args:
            n_samples (int): Number of samples to generate
            n_steps (int): Number of Gibbs sampling steps
            
        Returns:
            ndarray: Generated samples
        """
        v = np.random.binomial(1, 0.5, (n_samples, self.n_visible))
        
        for _ in range(n_steps):
            v, _, _ = gibbs_step(v, self.W, self.b, self.c)
            
        return v
class DeepBeliefNetwork:
    def __init__(self, hidden_sizes, n_visible, learning_rate=0.01):
        """
        Initialize DBN.
        
        Args:
            hidden_sizes (list): List of hidden layer sizes
            n_visible (int): Number of visible units
            learning_rate (float): Learning rate for each RBM
        """
        self.sizes = [n_visible] + hidden_sizes
        self.n_visible = n_visible
        self.learning_rate = learning_rate
        self.rbms = []
        self.log_likelihood_history = []
        
    def train(self, data, num_epochs):
        """
        Train the DBN layer by layer.
        
        Args:
            data (ndarray): Training data
            num_epochs (int): Number of epochs for each RBM
        """
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
            current_input = compute_conditional_hidden(
                current_input, 
                rbm.W, 
                rbm.c
            )
            
        # Store final weights and biases
        self.Ws = [rbm.W for rbm in self.rbms]
        self.cs = [rbm.c for rbm in self.rbms]