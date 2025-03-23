#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Deep Learning Project: RBM, DBN, and DNN Implementation
Run script to conduct analysis and experiments
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import time

# Import experiment functions
from main import compare_pretrained_vs_random_initialization, test_visualization
from test_rbm_dbn import analyze_binary_alphadigits

def main():
    """
    Main function to run experiments based on command line arguments
    """
    parser = argparse.ArgumentParser(description='Run Deep Learning experiments with RBM, DBN, and DNN')
    
    parser.add_argument('--alphadigits', action='store_true', 
                        help='Run Binary AlphaDigits analysis with RBM and DBN')
    
    parser.add_argument('--mnist', action='store_true',
                        help='Run MNIST analysis comparing pretrained vs random initialization')
    
    parser.add_argument('--viz', action='store_true',
                        help='Run visualization tests for generated samples')
    
    parser.add_argument('--all', action='store_true',
                        help='Run all experiments')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # If no arguments provided, run all experiments
    if not (args.alphadigits or args.mnist or args.viz):
        args.all = True
    
    # Run appropriate experiments
    if args.all or args.alphadigits:
        print("\n" + "="*80)
        print("Running Binary AlphaDigits analysis with RBM and DBN")
        print("="*80)
        try:
            start_time = time.time()
            rbm, X, size_img, results = analyze_binary_alphadigits()
            elapsed_time = time.time() - start_time
            print(f"Binary AlphaDigits analysis completed in {elapsed_time:.2f} seconds")
        except Exception as e:
            print(f"Error running Binary AlphaDigits analysis: {e}")
    
    if args.all or args.viz:
        print("\n" + "="*80)
        print("Running visualization tests for generated samples")
        print("="*80)
        try:
            start_time = time.time()
            rbm, dbn, images_rbm, images_dbn = test_visualization()
            elapsed_time = time.time() - start_time
            print(f"Visualization tests completed in {elapsed_time:.2f} seconds")
        except Exception as e:
            print(f"Error running visualization tests: {e}")
    
    if args.all or args.mnist:
        print("\n" + "="*80)
        print("Running MNIST analysis comparing pretrained vs random initialization")
        print("="*80)
        try:
            start_time = time.time()
            results = compare_pretrained_vs_random_initialization()
            elapsed_time = time.time() - start_time
            print(f"MNIST analysis completed in {elapsed_time:.2f} seconds")
            
            # Print summary of results
            print("\nSummary of MNIST Results:")
            print(f"Best pretrained configuration: {results['best_config']['pretrained']['config']} with accuracy {results['best_config']['pretrained']['accuracy']:.4f}")
            print(f"Best random initialization configuration: {results['best_config']['random']['config']} with accuracy {results['best_config']['random']['accuracy']:.4f}")
        except Exception as e:
            print(f"Error running MNIST analysis: {e}")
    
    print("\nAll requested experiments completed!")

if __name__ == "__main__":
    main()