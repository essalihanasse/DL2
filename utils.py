import numpy as np

def sigmoid(x):
    """Compute sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def compute_energy(v, h, W, b, c):
    """
    Compute the energy of the RBM for given visible and hidden states.
    
    Args:
        v (ndarray): Visible layer states
        h (ndarray): Hidden layer states
        W (ndarray): Weight matrix
        b (ndarray): Visible bias
        c (ndarray): Hidden bias
    
    Returns:
        float: Energy of the configuration
    """
    return -np.dot(v, b) - np.dot(h, c) - np.dot(np.dot(v, W), h)

def sample_bernoulli(probs):
    """Sample from Bernoulli distribution."""
    return (probs > np.random.random(probs.shape)).astype(float)

def compute_conditional_hidden(v, W, c):
    """
    Compute conditional probability p(h|v).
    
    Args:
        v (ndarray): Visible layer states
        W (ndarray): Weight matrix
        c (ndarray): Hidden bias
    
    Returns:
        ndarray: Conditional probabilities for hidden units
    """
    return sigmoid(np.dot(v, W) + c)

def compute_conditional_visible(h, W, b):
    """
    Compute conditional probability p(v|h).
    
    Args:
        h (ndarray): Hidden layer states
        W (ndarray): Weight matrix
        b (ndarray): Visible bias
    
    Returns:
        ndarray: Conditional probabilities for visible units
    """
    return sigmoid(np.dot(h, W.T) + b)

def gibbs_step(v, W, b, c):
    """
    Perform one step of Gibbs sampling.
    
    Args:
        v (ndarray): Current visible states
        W (ndarray): Weight matrix
        b (ndarray): Visible bias
        c (ndarray): Hidden bias
    
    Returns:
        tuple: (new visible states, new hidden states, hidden probabilities)
    """
    # Sample hidden units
    h_probs = compute_conditional_hidden(v, W, c)
    h = sample_bernoulli(h_probs)
    
    # Sample visible units
    v_probs = compute_conditional_visible(h, W, b)
    v_new = sample_bernoulli(v_probs)
    
    return v_new, h, h_probs

def compute_log_likelihood(v_data, W, b, c, num_samples=1000):
    """
    Estimate log-likelihood using importance sampling.
    
    Args:
        v_data (ndarray): Training data (visible configurations)
        W (ndarray): Weight matrix
        b (ndarray): Visible bias
        c (ndarray): Hidden bias
        num_samples (int): Number of samples for estimation
    
    Returns:
        float: Estimated log-likelihood
    """
    n_samples = v_data.shape[0]
    log_likelihood = 0
    
    for v in v_data:
        v = v.reshape(1, -1)
        
        # Get exact free energy for data
        h_probs = compute_conditional_hidden(v, W, c)
        free_energy_data = -np.dot(v, b.T) - np.sum(np.log(1 + np.exp(np.dot(v, W) + c)))
        
        # Sample from model for partition function estimation
        v_model = np.random.rand(num_samples, v.shape[1]) > 0.5
        h_probs_model = compute_conditional_hidden(v_model, W, c)
        free_energy_model = -np.dot(v_model, b.T) - np.sum(np.log(1 + np.exp(np.dot(v_model, W) + c)), axis=1)
        
        # Compute log-likelihood
        log_likelihood += free_energy_data - np.log(np.mean(np.exp(-free_energy_model)))
    
    return log_likelihood / n_samples