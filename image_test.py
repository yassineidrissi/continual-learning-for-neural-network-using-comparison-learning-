from rank_generalisation2 import *
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Function to handle training or testing mode
def main(training_mode=True):
    # Load Image
    im = np.array(Image.open('A.png').convert('L'))  # Load grayscale image
    print("Dimensions de l'image :", im.shape)

    # Prepare input sentence
    sentence = im.reshape(im.shape[0] * im.shape[1])
    print("Sentence :", sentence)

    # Parameters
    seq_res = 256
    seq_len = len(sentence)
    num_permutation = im.shape[0] * 8
    num_link = im.shape[0] * 2
    sigma_range = seq_res // 4
    print("num_link", num_link)

    # Visualize Input Image
    plt.figure()
    plt.imshow(im[::-1], cmap='gray', origin='lower', aspect='auto', interpolation='none')
    plt.show()

    # Initialize weight matrices
    W1, W2 = initialize_weights(num_features=seq_res, num_classes=2)

    # Generate H matrix for connections
    H = h_ldpc(seq_len, num_permutation, num_link)

    if training_mode:
        # Training Mode
        print("Training Mode")

        # Prepare Dataset
        num_samples = 100  # Number of samples in the dataset
        X = np.random.rand(num_samples, seq_res)  # Random input data
        y_true = np.eye(2)[np.random.randint(0, 2, size=num_samples)]  # Random one-hot labels

        # Train the model
        epochs = 20
        learning_rate = 0.01
        W1, W2, losses = train_model(X, y_true, W1, W2, learning_rate, epochs)

        # Plot Training Loss
        plt.plot(range(1, epochs + 1), losses)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()
    else:
        # Testing Mode
        print("Testing Mode")

        # Process through the updated model
        output = iterative_connection_with_weights(seq_res, seq_len, num_permutation, sigma_range, sentence, num_link, H, W1, W2)

        # Visualize Outputs
        plt.figure()
        # plt.imshow(output.reshape(im.shape[0], im.shape[1]), cmap='gray', origin='lower', aspect='auto', interpolation='none')
        plt.title("Output Image")
        plt.show()

if __name__ == "__main__":
    # Set training_mode to True for training, False for testing
    main(training_mode=True)
