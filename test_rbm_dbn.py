import numpy as np
import matplotlib.pyplot as plt
from scipy.io.matlab import loadmat

# Import the provided utility functions
from utils import sigmoid, lire_alpha_digit, display_images, gibbs_step, sample_bernoulli, entree_sortie_RBM
from models import RestrictedBoltzmannMachine, DeepBeliefNetwork


def binary_alphadigit_study():
    """
    Main function to study the Binary AlphaDigit dataset using RBM and DBN
    
    This function implements the steps mentioned in the requirements:
    1. Specify network parameters
    2. Load data
    3. Train an RBM and verify it can generate similar data
    4. Pre-train a DBN and verify it can generate similar data
    5. Discuss the quality of generated images based on hyperparameters
    """
    print("==== Etude sur Binary AlphaDigit ====")
    
    # 1. Specify network parameters
    print("\n1. Spécification des paramètres du réseau:")
    
    # Network architecture parameters
    rbm_hidden_units = 500
    dbn_hidden_sizes = [500, 200]
    learning_rate = 0.01
    
    # Training parameters
    n_epochs_rbm = 100
    n_epochs_dbn = 100
    batch_size = 10
    k_steps = 1  # Number of steps for contrastive divergence
    
    print(f"Architecture RBM: ?? -> {rbm_hidden_units}")
    print(f"Architecture DBN: ?? -> {' -> '.join(map(str, dbn_hidden_sizes))}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs RBM: {n_epochs_rbm}")
    print(f"Epochs DBN: {n_epochs_dbn}")
    print(f"Batch size: {batch_size}")
    print(f"Contrastive Divergence steps (k): {k_steps}")
    
    # 2. Load data
    print("\n2. Chargement des données:")
    try:
        fichier = loadmat("binaryalphadigs.mat")
        size_img = fichier['dat'][0][0].shape
        print(f"Image size: {size_img}")
    except FileNotFoundError:
        print("Error: binaryalphadigs.mat file not found. Please download it and place it in the current directory.")
        return
    
    # We'll use digits (0-9) for our experiment
    digit_classes = [3,4,5]
    print(f"Classes to use: {digit_classes}")
    
    X = lire_alpha_digit(fichier, digit_classes)
    print(f"Data shape: {X.shape}")
    
    # Display some original images
    print("\nAffichage de quelques images originales:")
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
    pixel_means = np.mean(X, axis=0)
    pixel_stds = np.std(X, axis=0)
    avg_active_pixels = np.mean(np.sum(X, axis=1))
    
    print(f"Average number of active pixels per image: {avg_active_pixels:.2f}")
    print(f"Pixel activation rate: {np.mean(pixel_means):.4f}")
    
    # 3. Train RBM and verify it can generate similar data
    print("\n3. Entraînement de RBM de manière non supervisée:")
    
    # Initialize and train RBM
    rbm = RestrictedBoltzmannMachine(
        n_visible=X.shape[1],
        n_hidden=rbm_hidden_units,
        learning_rate=learning_rate
    )
    
    print(f"Architecture finalisée RBM: {X.shape[1]} -> {rbm_hidden_units}")
    rbm.train(X, n_epochs=n_epochs_rbm, batch_size=batch_size, k=k_steps)
    
    # Generate images with RBM
    print("\nGénération d'images avec le RBM entraîné:")
    generated_images_rbm = rbm.generate_image_RBM(n_images=5, n_iter=1000, size_img=size_img)
    
    # Calculate statistics for generated images
    gen_pixel_means_rbm = np.mean(generated_images_rbm, axis=0)
    gen_avg_active_rbm = np.mean(np.sum(generated_images_rbm, axis=1))
    mean_diff_rbm = np.mean(np.abs(gen_pixel_means_rbm - pixel_means))
    
    print(f"Average active pixels in generated images: {gen_avg_active_rbm:.2f}")
    print(f"Mean pixel difference from original: {mean_diff_rbm:.4f}")
    
    # 4. Pre-train DBN and verify it can generate similar data
    print("\n4. Pré-entraînement du DBN de manière non supervisée:")
    
    # Initialize and train DBN
    dbn = DeepBeliefNetwork(
        hidden_sizes=dbn_hidden_sizes,
        n_visible=X.shape[1],
        learning_rate=learning_rate
    )
    
    print(f"Architecture finalisée DBN: {X.shape[1]} -> {' -> '.join(map(str, dbn_hidden_sizes))}")
    dbn.train(X, num_epochs=n_epochs_dbn, batch_size=batch_size, k=k_steps)
    
    # Generate images with DBN
    print("\nGénération d'images avec le DBN entraîné:")
    generated_images_dbn = dbn.generer_img_DBN(n_images=5, n_steps=1000, size_img=size_img)
    
    # Calculate statistics for generated images
    gen_pixel_means_dbn = np.mean(generated_images_dbn, axis=0)
    gen_avg_active_dbn = np.mean(np.sum(generated_images_dbn, axis=1))
    mean_diff_dbn = np.mean(np.abs(gen_pixel_means_dbn - pixel_means))
    
    print(f"Average active pixels in generated images: {gen_avg_active_dbn:.2f}")
    print(f"Mean pixel difference from original: {mean_diff_dbn:.4f}")
    
    # 5. Discuss the quality of generated images based on hyperparameters
    print("\n5. Discussion de la qualité des images régénérées:")
    
    # Compare RBM and DBN results
    print("\nComparaison RBM vs DBN:")
    print(f"RBM - Average active pixels: {gen_avg_active_rbm:.2f}, Mean diff: {mean_diff_rbm:.4f}")
    print(f"DBN - Average active pixels: {gen_avg_active_dbn:.2f}, Mean diff: {mean_diff_dbn:.4f}")
    
    # Try different hyperparameters for RBM
    print("\nTest de différents hyperparamètres pour RBM:")
    
    hidden_units_list = [100, 500, 1000]
    epochs_list = [50, 100]
    
    rbm_results = []
    
    for hidden_units in hidden_units_list:
        for epochs in epochs_list:
            print(f"\nTest RBM: {hidden_units} unités cachées, {epochs} époques")
            
            # Initialize and train RBM
            test_rbm = RestrictedBoltzmannMachine(
                n_visible=X.shape[1],
                n_hidden=hidden_units,
                learning_rate=learning_rate
            )
            
            test_rbm.train(X, n_epochs=epochs, batch_size=batch_size, k=k_steps, verbose=0)
            
            # Generate images
            test_gen = test_rbm.generate_image_RBM(n_images=5, n_iter=1000, size_img=size_img)
            
            # Calculate metrics
            test_mean_diff = np.mean(np.abs(np.mean(test_gen, axis=0) - pixel_means))
            test_avg_active = np.mean(np.sum(test_gen, axis=1))
            
            rbm_results.append({
                'hidden_units': hidden_units,
                'epochs': epochs,
                'mean_diff': test_mean_diff,
                'avg_active': test_avg_active
            })
            
            print(f"Average active pixels: {test_avg_active:.2f}")
            print(f"Mean pixel difference: {test_mean_diff:.4f}")
    
    # Print summary of RBM results
    print("\nRésumé des résultats RBM:")
    for result in sorted(rbm_results, key=lambda x: x['mean_diff']):
        print(f"Unités cachées: {result['hidden_units']}, "
              f"Époques: {result['epochs']}, "
              f"Diff moyenne: {result['mean_diff']:.4f}, "
              f"Pixels actifs: {result['avg_active']:.2f}")
    
    # Try different hidden layer configurations for DBN
    print("\nTest de différentes configurations de couches cachées pour DBN:")
    
    dbn_configs = [
        [200],
        [500],
        [1000],
        [500, 200],
        [1000, 500]
    ]
    
    dbn_results = []
    
    for hidden_sizes in dbn_configs:
        print(f"\nTest DBN: {' -> '.join(map(str, hidden_sizes))}")
        
        # Initialize and train DBN
        test_dbn = DeepBeliefNetwork(
            hidden_sizes=hidden_sizes,
            n_visible=X.shape[1],
            learning_rate=learning_rate
        )
        
        test_dbn.train(X, num_epochs=n_epochs_dbn, batch_size=batch_size, k=k_steps, verbose=0)
        
        # Generate images
        test_gen = test_dbn.generer_img_DBN(n_images=5, n_steps=1000, size_img=size_img)
        
        # Calculate metrics
        test_mean_diff = np.mean(np.abs(np.mean(test_gen, axis=0) - pixel_means))
        test_avg_active = np.mean(np.sum(test_gen, axis=1))
        
        dbn_results.append({
            'hidden_sizes': hidden_sizes,
            'mean_diff': test_mean_diff,
            'avg_active': test_avg_active
        })
        
        print(f"Average active pixels: {test_avg_active:.2f}")
        print(f"Mean pixel difference: {test_mean_diff:.4f}")
    
    # Print summary of DBN results
    print("\nRésumé des résultats DBN:")
    for result in sorted(dbn_results, key=lambda x: x['mean_diff']):
        print(f"Couches cachées: {' -> '.join(map(str, result['hidden_sizes']))}, "
              f"Diff moyenne: {result['mean_diff']:.4f}, "
              f"Pixels actifs: {result['avg_active']:.2f}")
    
    # Final analysis and conclusion
    print("\n=== Analyse finale et conclusion ===")
    
    # Best RBM configuration
    best_rbm = min(rbm_results, key=lambda x: x['mean_diff'])
    print(f"Meilleure configuration RBM: {best_rbm['hidden_units']} unités cachées, {best_rbm['epochs']} époques")
    
    # Best DBN configuration
    best_dbn = min(dbn_results, key=lambda x: x['mean_diff'])
    print(f"Meilleure configuration DBN: {' -> '.join(map(str, best_dbn['hidden_sizes']))}")
    
    # Compare best models
    if best_rbm['mean_diff'] < best_dbn['mean_diff']:
        print("\nLe RBM a donné de meilleurs résultats en termes de qualité d'image.")
    else:
        print("\nLe DBN a donné de meilleurs résultats en termes de qualité d'image.")
    
    
    return rbm, dbn, X, size_img, rbm_results, dbn_results

if __name__ == "__main__":
    rbm, dbn, X, size_img, rbm_results, dbn_results = binary_alphadigit_study()
