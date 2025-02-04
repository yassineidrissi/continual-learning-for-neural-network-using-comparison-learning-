from rank_generalisation2 import *
import time
import numpy as np
import pickle
from PIL import Image
import matplotlib.pyplot as plt

# Function to handle training or testing mode
def main(training_mode=True):
    seq_res = 3072  # CIFAR-10 input size (32x32x3)
    num_classes = 10  # CIFAR-10 has 10 classes

    if training_mode:
        print("Training Mode")
        # Initialize weights
        W1, W2 = initialize_weights(num_features=seq_res, num_classes=num_classes)

        # Train on all CIFAR-10 batches
        for i in range(1, 6):
            print(f"Loading and training on data_batch_{i}")
            data, labels = load_cifar10_batch(f'data_batch_{i}')
            data, labels = preprocess_cifar10(data, labels)
            W1, W2, losses = train_model(data, labels, W1, W2, learning_rate=0.01, epochs=5)

        # Save trained weights
        save_weights(W1, W2, filename='cifar10_weights.pkl')
    else:
        print("Testing Mode")
        # Load weights
        W1, W2 = load_weights(filename='cifar10_weights.pkl')

        # Test on the test batch
        test_data, test_labels = load_cifar10_batch('test_batch')
        test_data, test_labels = preprocess_cifar10(test_data, test_labels)

        # Perform testing
        test_output = iterative_connection_with_weights(seq_res, test_data.shape[1], num_permutation, sigma_range, test_data, num_link, H, W1, W2)
        print("Testing completed")


if __name__ == "__main__":
    # Set to True for training, False for testing
    training_mode = True
    main(training_mode=training_mode)
