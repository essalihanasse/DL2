import numpy as np
from models import *
if __name__ == "__main__":
    # Generate synthetic dataset: simple patterns
    n_samples = 1000
    n_visible = 10
    patterns = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0]])
    
    # Create dataset by randomly selecting and adding noise to patterns
    data = []
    for _ in range(n_samples):
        pattern = patterns[np.random.randint(0, len(patterns))]
        # Add noise
        noise = np.random.binomial(1, 0.1, n_visible)
        sample = np.logical_xor(pattern, noise).astype(float)
        data.append(sample)
    
    data = np.array(data)
    
    # Create and train RBM
    rbm = RestrictedBoltzmannMachine(n_visible=10, n_hidden=5)
    rbm.train(data, n_epochs=100, batch_size=64, k=1)
    
    # Generate samples
    generated_samples = rbm.generate_samples(n_samples=10)
    
    # Print results
    print("\nOriginal patterns:")
    print(patterns)
    print("\nGenerated samples:")
    print(generated_samples)
    print("\nFinal log-likelihood:", rbm.log_likelihood_history[-1])