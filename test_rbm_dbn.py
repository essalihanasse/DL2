import numpy as np
import matplotlib.pyplot as plt
from scipy.io.matlab import loadmat

# Import the provided utility functions
from utils import sigmoid, lire_alpha_digit, display_images, gibbs_step, sample_bernoulli, entree_sortie_RBM
from models import RestrictedBoltzmannMachine, DeepBeliefNetwork

def analyze_binary_alphadigits():
    """Detailed analysis of RBM and DBN on the Binary AlphaDigits dataset"""
    print("===== ANALYZING BINARY ALPHADIGITS DATASET =====")
    
    # Load Binary AlphaDigits data
    print("Loading Binary AlphaDigits data...")
    try:
        fichier = loadmat("binaryalphadigs.mat")
        size_img = fichier['dat'][0][0].shape
        print(f"Image size: {size_img}")
    except FileNotFoundError:
        print("Error: binaryalphadigs.mat file not found. Please download it from Kaggle.")
        return
    
    X = lire_alpha_digit(fichier, [3,4,5])
    print(f"Data shape: {X.shape}")
    
    # Display some original images
    print("\nDisplaying some original images:")
    indices = np.random.choice(X.shape[0], 5, replace=False)
    plt.figure(figsize=(12, 3))
    for i, idx in enumerate(indices):
        plt.subplot(1, 5, i+1)
        plt.imshow(X[idx].reshape(size_img), cmap='gray')
        plt.title(f"Original {i+1}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Calculate data statistics for comparison with generated samples
    print("\nAnalyzing data statistics:")
    pixel_means = np.mean(X, axis=0)
    pixel_stds = np.std(X, axis=0)
    avg_active_pixels = np.mean(np.sum(X, axis=1))
    
    print(f"Average number of active pixels per image: {avg_active_pixels:.2f}")
    print(f"Pixel activation rate: {np.mean(pixel_means):.4f}")
    
    # Visualize mean activation pattern
    plt.figure(figsize=(6, 6))
    plt.imshow(pixel_means.reshape(size_img), cmap='viridis')
    plt.colorbar(label='Activation probability')
    plt.title('Mean Activation Pattern')
    plt.show()
    
    # Test different hidden layer sizes for RBM
    hidden_units_list = [100, 500,1000]
    training_epochs_list = [500, 1000]
    
    results = []
    
    for hidden_units in hidden_units_list:
        for epochs in training_epochs_list:
            print(f"\n===== Testing RBM with {hidden_units} hidden units, {epochs} epochs =====")
            
            # Initialize RBM with more hidden units
            rbm = RestrictedBoltzmannMachine(
                n_visible=X.shape[1], 
                n_hidden=hidden_units,
                learning_rate=0.01
            )
            
            print(f"Training RBM with {X.shape[0]} samples...")
            print(f"Architecture: {X.shape[1]} visible units, {hidden_units} hidden units")
            
            # Train RBM with more epochs
            rbm.train(X, n_epochs=epochs, batch_size=10, k=1)
            
            # Generate images with RBM
            print(f"\nGenerating images with trained RBM (after {epochs} epochs):")
            generated_images = rbm.generate_image_RBM(n_images=5, n_iter=2000, size_img=size_img)
            
            # Calculate statistics for generated images
            gen_pixel_means = np.mean(generated_images, axis=0)
            gen_avg_active = np.mean(np.sum(generated_images, axis=1))
            
            # Calculate similarity to training data
            mean_diff = np.mean(np.abs(gen_pixel_means - pixel_means))
            active_diff = np.abs(gen_avg_active - avg_active_pixels)
            
            results.append({
                'hidden_units': hidden_units,
                'epochs': epochs,
                'mean_diff': mean_diff,
                'active_diff': active_diff,
                'avg_active_pixels': gen_avg_active
            })
            
            print(f"Average active pixels in generated images: {gen_avg_active:.2f}")
            print(f"Mean pixel difference from original: {mean_diff:.4f}")
    
    # Print summary of results
    print("\n===== Results Summary =====")
    for result in sorted(results, key=lambda x: x['mean_diff']):
        print(f"Hidden units: {result['hidden_units']}, "
              f"Epochs: {result['epochs']}, "
              f"Mean diff: {result['mean_diff']:.4f}, "
              f"Active diff: {result['active_diff']:.2f}, "
              f"Avg active: {result['avg_active_pixels']:.2f}")
    
    # Test best configuration with more Gibbs steps
    best_config = min(results, key=lambda x: x['mean_diff'])
    hidden_units = best_config['hidden_units']
    epochs = best_config['epochs']
    
    print(f"\n===== Testing best configuration with more Gibbs steps =====")
    print(f"Hidden units: {hidden_units}, Epochs: {epochs}")
    
    rbm = RestrictedBoltzmannMachine(
        n_visible=X.shape[1], 
        n_hidden=hidden_units,
        learning_rate=0.01
    )
    
    rbm.train(X, n_epochs=epochs, batch_size=10, k=1)
    
    # Test different Gibbs sampling steps
    gibbs_steps_list = [100, 500, 1000]
    
    plt.figure(figsize=(15, len(gibbs_steps_list) * 3))
    plt.suptitle(f"Effect of Gibbs Sampling Steps on RBM Generation Quality\n"
                f"Hidden Units: {hidden_units}, Training Epochs: {epochs}", fontsize=16)
    
    for i, steps in enumerate(gibbs_steps_list):
        print(f"\nGenerating images with {steps} Gibbs sampling steps:")
        gen_imgs = rbm.generate_image_RBM(n_images=5, n_iter=steps, size_img=None)
        
        for j in range(5):
            plt.subplot(len(gibbs_steps_list), 5, i*5 + j + 1)
            plt.imshow(gen_imgs[j].reshape(size_img), cmap='gray')
            if j == 0:
                plt.ylabel(f"{steps} steps", fontsize=12)
            if i == 0:
                plt.title(f"Sample {j+1}", fontsize=12)
            plt.axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()
    
    # Analyze potential issues with generation quality
    print("\n===== Analyzing Potential Issues with Generation Quality =====")
    
    # Check for mode collapse
    if best_config['avg_active_pixels'] < 0.5 * avg_active_pixels:
        print("Issue detected: Mode collapse - generated images have fewer active pixels than originals")
        print("This suggests the model is not capturing the full distribution of the data")
    
    # Check for oversmoothing
    if mean_diff > 0.2:
        print("Issue detected: Oversmoothing - generated samples are too different from training data")
        print("This suggests the model might need more training or a different architecture")
    
    # Check for missing fine details
    print("\nAnalysis of feature learning:")
    weights = rbm.W
    
    # Visualize some learned features
    plt.figure(figsize=(15, 5))
    plt.suptitle("Sample of Learned RBM Features", fontsize=16)
    
    for i in range(10):
        idx = np.random.randint(0, weights.shape[1])
        feature = weights[:, idx].reshape(size_img)
        
        plt.subplot(2, 5, i+1)
        plt.imshow(feature, cmap='coolwarm')
        plt.title(f"Feature {idx}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()
    
    return rbm, X, size_img, results

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run the analysis
    rbm, X, size_img, results = analyze_binary_alphadigits()