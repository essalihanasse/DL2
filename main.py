import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from scipy.io.matlab import loadmat
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from scipy.io.matlab import loadmat
import os
import json
from datetime import datetime
import numba
# from numba import jit, cuda, prange
import time

# Import the provided utility functions
from utils import sigmoid, sample_bernoulli, entree_sortie_RBM
from models import RestrictedBoltzmannMachine, DeepBeliefNetwork, DeepNeuralNetworkKeras
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print(f"Using GPU: {physical_devices[0].name}")
else:
    print("No GPU found, using CPU instead")

# Use mixed precision to speed up GPU computation
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Create results directory if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')

def save_results(results, filename):
    """Save experiment results to JSON file"""
    with open(f'results/{filename}.json', 'w') as f:
        json.dump(results, f, indent=2)

def experiment_network_depth(X_train, y_train_onehot, X_test, y_test_onehot):
    """
    Experiment 1: Compare performance of pre-trained vs random initialization
    with varying network depth (number of layers)
    """
    print("\n===== Experiment 1: Varying number of layers =====")

    # Early stopping callback
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    # Define layer configurations to test
    layer_configs = [
        [200],
        [200, 200],
        [200, 200, 200],
        [200, 200, 200, 200]
    ]

    pretrained_error_layers = []
    random_error_layers = []

    for i, hidden_layers in enumerate(layer_configs):
        config_start = time.time()
        print(f"\nTesting configuration with {len(hidden_layers)} hidden layers: {hidden_layers}")

        # Create pretrained model
        pretrained_model = DeepNeuralNetworkKeras(
            hidden_sizes=hidden_layers,
            n_visible=784,
            n_classes=10,
            learning_rate=0.001
        )

        # Create randomly initialized model
        random_model = DeepNeuralNetworkKeras(
            hidden_sizes=hidden_layers,
            n_visible=784,
            n_classes=10,
            learning_rate=0.001
        )

        # Pre-train the pretrained model
        print("Pre-training model with DBN...")
        pretrain_start = time.time()
        pretrained_model.pretrain(X_train, num_epochs=50, batch_size=256, k=1, verbose=1)
        pretrain_time = time.time() - pretrain_start
        print(f"Pre-training completed in {pretrain_time:.2f}s")

        # Train both models
        print("Training pretrained model...")
        train_start = time.time()
        pretrained_history = pretrained_model.train(
            X_train, y_train_onehot,
            n_epochs=100,
            batch_size=256,
            validation_data=(X_test, y_test_onehot),
            callbacks=[early_stopping]
        )
        train_time = time.time() - train_start
        print(f"Pretrained model training completed in {train_time:.2f}s")

        print("Training randomly initialized model...")
        random_start = time.time()
        random_history = random_model.train(
            X_train, y_train_onehot,
            n_epochs=100,
            batch_size=256,
            validation_data=(X_test, y_test_onehot),
            callbacks=[early_stopping]
        )
        random_time = time.time() - random_start
        print(f"Random model training completed in {random_time:.2f}s")

        # Evaluate both models
        _, pretrained_acc = pretrained_model.evaluate(X_test, y_test_onehot)
        _, random_acc = random_model.evaluate(X_test, y_test_onehot)

        pretrained_error = 1 - pretrained_acc
        random_error = 1 - random_acc

        pretrained_error_layers.append(pretrained_error)
        random_error_layers.append(random_error)

        print(f"Pretrained model error rate: {pretrained_error:.4f}")
        print(f"Random model error rate: {random_error:.4f}")

        config_time = time.time() - config_start
        print(f"Configuration {len(hidden_layers)} layers completed in {config_time:.2f}s")

    # Plot the results
    plt.figure(figsize=(10, 6))
    x_labels = [str(len(config)) for config in layer_configs]
    plt.plot(x_labels, pretrained_error_layers, 'b-o', label='Pretrained')
    plt.plot(x_labels, random_error_layers, 'r-o', label='Random initialization')
    plt.xlabel('Number of Hidden Layers')
    plt.ylabel('Error Rate')
    plt.title('Error Rate vs Number of Hidden Layers')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/error_vs_layers.png')
    plt.close()

    return {
        "configs": [len(config) for config in layer_configs],
        "pretrained_errors": pretrained_error_layers,
        "random_errors": random_error_layers
    }

def experiment_neuron_count(X_train, y_train_onehot, X_test, y_test_onehot):
    """
    Experiment 2: Compare performance of pre-trained vs random initialization
    with varying number of neurons per layer
    """
    print("\n===== Experiment 2: Varying number of neurons per layer =====")

    # Early stopping callback
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    # Define neuron configurations to test (2 layers with varying sizes)
    neuron_configs = [
        [100, 100],
        [200, 200],
        [300, 300],
        [500, 500],
        [700, 700]
    ]

    pretrained_error_neurons = []
    random_error_neurons = []

    for hidden_layers in neuron_configs:
        config_start = time.time()
        print(f"\nTesting configuration with hidden layers: {hidden_layers}")

        # Create pretrained model
        pretrained_model = DeepNeuralNetworkKeras(
            hidden_sizes=hidden_layers,
            n_visible=784,
            n_classes=10,
            learning_rate=0.001
        )

        # Create randomly initialized model
        random_model = DeepNeuralNetworkKeras(
            hidden_sizes=hidden_layers,
            n_visible=784,
            n_classes=10,
            learning_rate=0.001
        )

        # Pre-train the pretrained model
        print("Pre-training model with DBN...")
        pretrain_start = time.time()
        pretrained_model.pretrain(X_train, num_epochs=50, batch_size=256, k=1, verbose=1)
        pretrain_time = time.time() - pretrain_start
        print(f"Pre-training completed in {pretrain_time:.2f}s")

        # Train both models
        print("Training pretrained model...")
        train_start = time.time()
        pretrained_model.train(
            X_train, y_train_onehot,
            n_epochs=100,
            batch_size=256,
            validation_data=(X_test, y_test_onehot),
            callbacks=[early_stopping]
        )
        train_time = time.time() - train_start
        print(f"Pretrained model training completed in {train_time:.2f}s")

        print("Training randomly initialized model...")
        random_start = time.time()
        random_model.train(
            X_train, y_train_onehot,
            n_epochs=100,
            batch_size=256,
            validation_data=(X_test, y_test_onehot),
            callbacks=[early_stopping]
        )
        random_time = time.time() - random_start
        print(f"Random model training completed in {random_time:.2f}s")

        # Evaluate both models
        _, pretrained_acc = pretrained_model.evaluate(X_test, y_test_onehot)
        _, random_acc = random_model.evaluate(X_test, y_test_onehot)

        pretrained_error = 1 - pretrained_acc
        random_error = 1 - random_acc

        pretrained_error_neurons.append(pretrained_error)
        random_error_neurons.append(random_error)

        print(f"Pretrained model error rate: {pretrained_error:.4f}")
        print(f"Random model error rate: {random_error:.4f}")

        config_time = time.time() - config_start
        print(f"Configuration {hidden_layers} completed in {config_time:.2f}s")

    # Plot the results
    plt.figure(figsize=(10, 6))
    x_labels = [str(config[0]) for config in neuron_configs]
    plt.plot(x_labels, pretrained_error_neurons, 'b-o', label='Pretrained')
    plt.plot(x_labels, random_error_neurons, 'r-o', label='Random initialization')
    plt.xlabel('Number of Neurons per Layer')
    plt.ylabel('Error Rate')
    plt.title('Error Rate vs Number of Neurons per Layer (2 layers)')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/error_vs_neurons.png')
    plt.close()

    return {
        "configs": [config[0] for config in neuron_configs],
        "pretrained_errors": pretrained_error_neurons,
        "random_errors": random_error_neurons
    }

def experiment_training_size(X_train, y_train_onehot, X_test, y_test_onehot):
    """
    Experiment 3: Compare performance of pre-trained vs random initialization
    with varying number of training samples
    """
    print("\n===== Experiment 3: Varying number of training samples =====")

    # Early stopping callback
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    # Define sample sizes to test
    sample_sizes = [1000, 3000, 7000]

    pretrained_error_samples = []
    random_error_samples = []

    # Fixed network configuration for this experiment
    hidden_layers = [200, 200]

    for size in sample_sizes:
        config_start = time.time()
        actual_size = min(size, X_train.shape[0])
        print(f"\nTesting with {actual_size} training samples")

        # Subsample the data
        indices = np.random.choice(X_train.shape[0], actual_size, replace=False)
        X_train_subset = X_train[indices]
        y_train_subset = y_train_onehot[indices]

        # Create pretrained model
        pretrained_model = DeepNeuralNetworkKeras(
            hidden_sizes=hidden_layers,
            n_visible=784,
            n_classes=10,
            learning_rate=0.001
        )

        # Create randomly initialized model
        random_model = DeepNeuralNetworkKeras(
            hidden_sizes=hidden_layers,
            n_visible=784,
            n_classes=10,
            learning_rate=0.001
        )

        # Pre-train the pretrained model
        print("Pre-training model with DBN...")
        pretrain_start = time.time()
        pretrained_model.pretrain(X_train_subset, num_epochs=50, batch_size=256, k=1, verbose=1)
        pretrain_time = time.time() - pretrain_start
        print(f"Pre-training completed in {pretrain_time:.2f}s")

        # Train both models
        print("Training pretrained model...")
        train_start = time.time()
        pretrained_model.train(
            X_train_subset, y_train_subset,
            n_epochs=100,
            batch_size=256,
            validation_data=(X_test, y_test_onehot),
            callbacks=[early_stopping]
        )
        train_time = time.time() - train_start
        print(f"Pretrained model training completed in {train_time:.2f}s")

        print("Training randomly initialized model...")
        random_start = time.time()
        random_model.train(
            X_train_subset, y_train_subset,
            n_epochs=100,
            batch_size=256,
            validation_data=(X_test, y_test_onehot),
            callbacks=[early_stopping]
        )
        random_time = time.time() - random_start
        print(f"Random model training completed in {random_time:.2f}s")

        # Evaluate both models
        _, pretrained_acc = pretrained_model.evaluate(X_test, y_test_onehot)
        _, random_acc = random_model.evaluate(X_test, y_test_onehot)

        pretrained_error = 1 - pretrained_acc
        random_error = 1 - random_acc

        pretrained_error_samples.append(pretrained_error)
        random_error_samples.append(random_error)

        print(f"Pretrained model error rate: {pretrained_error:.4f}")
        print(f"Random model error rate: {random_error:.4f}")

        config_time = time.time() - config_start
        print(f"Sample size {actual_size} completed in {config_time:.2f}s")

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(sample_sizes, pretrained_error_samples, 'b-o', label='Pretrained')
    plt.plot(sample_sizes, random_error_samples, 'r-o', label='Random initialization')
    plt.xlabel('Number of Training Samples')
    plt.ylabel('Error Rate')
    plt.title('Error Rate vs Number of Training Samples')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/error_vs_samples.png')
    plt.close()

    return {
        "sizes": sample_sizes,
        "pretrained_errors": pretrained_error_samples,
        "random_errors": random_error_samples
    }

def find_best_configuration(X_train, y_train_onehot, X_test, y_test_onehot):
    """
    Find the best network configuration based on previous experiments
    """
    print("\n===== Finding the best configuration =====")

    # Early stopping callback
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    # Based on the previous experiments, test a few promising configurations
    configurations = [
        {"name": "3x500", "hidden_layers": [500, 500, 500]},
        {"name": "4x300", "hidden_layers": [300, 300, 300, 300]},
        {"name": "2x700", "hidden_layers": [700, 700]}
    ]

    best_pretrained_acc = 0
    best_random_acc = 0
    best_pretrained_config = None
    best_random_config = None

    results = []

    for config in configurations:
        config_start = time.time()
        print(f"\nTesting configuration: {config['name']}")

        # Create pretrained model
        pretrained_model = DeepNeuralNetworkKeras(
            hidden_sizes=config['hidden_layers'],
            n_visible=784,
            n_classes=10,
            learning_rate=0.001
        )

        # Create randomly initialized model
        random_model = DeepNeuralNetworkKeras(
            hidden_sizes=config['hidden_layers'],
            n_visible=784,
            n_classes=10,
            learning_rate=0.001
        )

        # Pre-train the pretrained model
        print("Pre-training model with DBN...")
        pretrain_start = time.time()
        pretrained_model.pretrain(X_train, num_epochs=50, batch_size=256, k=1, verbose=1)
        pretrain_time = time.time() - pretrain_start
        print(f"Pre-training completed in {pretrain_time:.2f}s")

        # Train both models
        print("Training pretrained model...")
        train_start = time.time()
        pretrained_model.train(
            X_train, y_train_onehot,
            n_epochs=100,
            batch_size=256,
            validation_data=(X_test, y_test_onehot),
            callbacks=[early_stopping]
        )
        train_time = time.time() - train_start
        print(f"Pretrained model training completed in {train_time:.2f}s")

        print("Training randomly initialized model...")
        random_start = time.time()
        random_model.train(
            X_train, y_train_onehot,
            n_epochs=100,
            batch_size=256,
            validation_data=(X_test, y_test_onehot),
            callbacks=[early_stopping]
        )
        random_time = time.time() - random_start
        print(f"Random model training completed in {random_time:.2f}s")

        # Evaluate both models
        _, pretrained_acc = pretrained_model.evaluate

def run_all_experiments():
    """
    Run all experiments and save results
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Load MNIST data
    print("Loading MNIST data...")
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    # Reshape and binarize the data
    X_train = X_train.reshape(-1, 784).astype('float32')
    X_test = X_test.reshape(-1, 784).astype('float32')

    # Binarize
    X_train = (X_train > 127).astype('float32')
    X_test = (X_test > 127).astype('float32')

    # One-hot encode labels
    y_train_onehot = keras.utils.to_categorical(y_train, 10)
    y_test_onehot = keras.utils.to_categorical(y_test, 10)

    print(f"MNIST data loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")

    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Run experiments
    results = {}

    # Experiment 1: Varying number of layers
    layer_results = experiment_network_depth(X_train, y_train_onehot, X_test, y_test_onehot)
    results["layer_results"] = layer_results
    save_results(layer_results, f"{timestamp}_layer_results")

    # Experiment 2: Varying number of neurons per layer
    neuron_results = experiment_neuron_count(X_train, y_train_onehot, X_test, y_test_onehot)
    results["neuron_results"] = neuron_results
    save_results(neuron_results, f"{timestamp}_neuron_results")

    # Experiment 3: Varying number of training samples
    sample_results = experiment_training_size(X_train, y_train_onehot, X_test, y_test_onehot)
    results["sample_results"] = sample_results
    save_results(sample_results, f"{timestamp}_sample_results")

    # Find best configuration
    best_config = find_best_configuration(X_train, y_train_onehot, X_test, y_test_onehot)
    results["best_config"] = best_config
    save_results(best_config, f"{timestamp}_best_config")

    # Save full results
    save_results(results, f"{timestamp}_all_results")

    print("\nAll experiments completed and results saved!")
    return results

if __name__ == "__main__":
    results = run_all_experiments()