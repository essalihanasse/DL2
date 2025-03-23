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
        self.W = np.random.normal(0, 0.01, (n_visible, n_hidden)).astype(np.float32)
        self.b = np.zeros(n_visible, dtype=np.float32)
        self.c = np.zeros(n_hidden, dtype=np.float32)

    def train(self, data, n_epochs=100, batch_size=10, k=1, verbose=1):
        """Train RBM with optimized mini-batch processing"""
        start_time = time.time()
        n_samples = data.shape[0]
        n_batches = n_samples // batch_size

        for epoch in range(n_epochs):
            epoch_start = time.time()
            # Shuffle data
            indices = np.random.permutation(n_samples)
            shuffled_data = data[indices]

            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                if i + batch_size <= n_samples:
                    batch = shuffled_data[i:i+batch_size]
                    self._contrastive_divergence(batch, k)

            if verbose and (epoch + 1) % 10 == 0:
                epoch_time = time.time() - epoch_start
                print(f"Epoch {epoch + 1}/{n_epochs} completed in {epoch_time:.2f}s")

        if verbose:
            total_time = time.time() - start_time
            print(f"RBM training completed in {total_time:.2f}s")

    def _contrastive_divergence(self, batch, k):
        """
        Perform k-step contrastive divergence update with optimized implementation.
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
        batch_size = batch.shape[0]
        self.W += self.learning_rate * (pos_associations - neg_associations) / batch_size
        self.b += self.learning_rate * np.mean(batch - v_model, axis=0)
        self.c += self.learning_rate * np.mean(h_probs - h_probs_model, axis=0)

    def generate_image_RBM(self, n_images, n_iter, size_img=None):
        """Generate images using the trained RBM"""
        v = np.random.binomial(1, 0.5, (n_images, self.n_visible)).astype(np.float32)
        for _ in range(n_iter):
            v, _, _ = gibbs_step(v, self.W, self.b, self.c)

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

    def train(self, data, num_epochs, batch_size=100, k=1, verbose=1):
        """Train DBN with optimized layer-wise strategy"""
        start_time = time.time()
        current_input = data

        # Train each RBM layer
        for i in range(len(self.sizes) - 1):
            if verbose:
                print(f"Training RBM layer {i+1}/{len(self.sizes)-1}")
                layer_start = time.time()

            # Initialize and train current RBM
            rbm = RestrictedBoltzmannMachine(
                n_visible=self.sizes[i],
                n_hidden=self.sizes[i+1],
                learning_rate=self.learning_rate
            )

            rbm.train(current_input, n_epochs=num_epochs, batch_size=batch_size, k=k, verbose=verbose)
            self.rbms.append(rbm)

            # Transform data for next layer
            current_input = entree_sortie_RBM(current_input, rbm.W, rbm.c)

            if verbose:
                layer_time = time.time() - layer_start
                print(f"Layer {i+1} training completed in {layer_time:.2f}s")

        # Store final weights and biases
        self.Ws = [rbm.W for rbm in self.rbms]
        self.cs = [rbm.c for rbm in self.rbms]

        if verbose:
            total_time = time.time() - start_time
            print(f"DBN training completed in {total_time:.2f}s")

    def generer_img_DBN(self, n_images=5, n_steps=1000, size_img=None):
        """Generate images using the trained DBN"""
        top_rbm = self.rbms[-1]
        h = np.random.binomial(1, 0.5, (n_images, top_rbm.n_hidden)).astype(np.float32)
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
      # Use mixed precision for better GPU performance
      tf.keras.mixed_precision.set_global_policy('mixed_float16')

      model = models.Sequential()

      # First layer (note: we DON'T use a separate Input layer)
      model.add(layers.Dense(
          self.sizes[1],
          activation='sigmoid',
          input_shape=(self.sizes[0],),
          kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
          bias_initializer='zeros'
      ))

      # Remaining hidden layers
      for i in range(1, len(self.sizes) - 2):
          model.add(layers.Dense(
              self.sizes[i+1],
              activation='sigmoid',
              kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
              bias_initializer='zeros'
          ))

      # Output layer with softmax activation
      model.add(layers.Dense(self.sizes[-1], activation='softmax'))

      # Print model summary to help debug
      model.summary()

      # Compile with mixed precision
      optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
      model.compile(
          optimizer=optimizer,
          loss='categorical_crossentropy',
          metrics=['accuracy']
      )

      self.model = model

    def pretrain(self, data, num_epochs=100, batch_size=10, k=1, verbose=1):
      """
      Pretrain the DNN using a Deep Belief Network (greedy layer-wise training).

      Args:
          data (ndarray): Training data (samples x features)
          num_epochs (int): Number of epochs for each RBM layer
          batch_size (int): Size of mini-batches
          k (int): Number of Gibbs sampling steps for contrastive divergence
      """
      start_time = time.time()
      if verbose:
          print("Pre-training with DBN...")

      # Create a DBN with the same architecture (excluding output layer)
      dbn = DeepBeliefNetwork(
          hidden_sizes=self.sizes[1:-1],
          n_visible=self.sizes[0],
          learning_rate=self.learning_rate
      )

      # Train the DBN
      dbn.train(data, num_epochs, batch_size=batch_size, k=k, verbose=verbose)

      # Get weights and biases from DBN
      weights = []
      biases = []

      for i in range(len(dbn.rbms)):
          weights.append(dbn.rbms[i].W)
          biases.append(dbn.rbms[i].c)

      # Set weights in Keras model (excluding the output layer)
      for i, (w, b) in enumerate(zip(weights, biases)):
          if i < len(self.model.layers) - 1:  # Exclude output layer
              layer = self.model.layers[i]

              # Get expected shapes from the layer
              expected_w_shape = layer.get_weights()[0].shape
              expected_b_shape = layer.get_weights()[1].shape

              # Debug the shape mismatch
              if verbose:
                  print(f"Layer {i} (type: {layer.__class__.__name__}):")
                  print(f"  Expected weight shape: {expected_w_shape}")
                  print(f"  DBN weight shape: {w.shape}")
                  print(f"  Expected bias shape: {expected_b_shape}")
                  print(f"  DBN bias shape: {b.shape}")

              # Only set weights if shapes match
              if w.shape == expected_w_shape and b.shape == expected_b_shape:
                  if verbose:
                      print(f"  Setting weights for layer {i}")
                  layer.set_weights([w, b])
              else:
                  print(f"Warning: Shape mismatch in layer {i}, trying alternatives")

                  # Try transposing
                  if w.T.shape == expected_w_shape:
                      print(f"  Setting transposed weights for layer {i}")
                      layer.set_weights([w.T, b])
                  else:
                      print(f"  Cannot resolve shape mismatch automatically for layer {i}")

      self.weights_initialized = True

      if verbose:
          total_time = time.time() - start_time
          print(f"Pre-training completed in {total_time:.2f}s")
    def train(self, X, y, n_epochs=100, batch_size=128, validation_data=None, callbacks=None, verbose=1):
        """
        Train the DNN using backpropagation with performance optimizations.
        """
        # Create early stopping callback if not provided
        if callbacks is None:
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            callbacks = [early_stopping]

        # Train with larger batch size for better GPU utilization
        history = self.model.fit(
            X, y,
            epochs=n_epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose
        )

        return history

    def evaluate(self, X, y):
        """Evaluate the model on test data."""
        return self.model.evaluate(X, y, verbose=0)

    def predict(self, X):
        """Generate predictions for input data."""
        return self.model.predict(X, verbose=0)