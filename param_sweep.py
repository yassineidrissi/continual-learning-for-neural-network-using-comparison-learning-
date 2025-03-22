#!/usr/bin/env python3
"""
param_sweep.py

This script demonstrates how to systematically vary parameters such as:
- num_link
- hidden layer size (for V2)
- number of iterations
- matrix type
- noise level
- image size

...and measure the reconstruction error and execution time for each setting.
"""

import time
import numpy as np
from PIL import Image
# Example imports for your code:
import rank_generalisation2 as rg1          # V1 (single-layer)
import rank_generalisation_2layer as rg2    # V2 (two-layer)

###############################################################################
# Helper functions (placeholders - adapt to your actual project functions)    #
###############################################################################

def load_and_preprocess_image(image_path, size=(64, 64), noise_level=0):
    """
    Loads an image, converts to grayscale, resizes, flattens, 
    and optionally adds noise.
    """
    im = array(Image.open(image_path).convert('L')) #you can pass multiple arguments in single line

    #hist(reshh_ldpce(im,(512*512,1)),100)
    #(216,): [0, 255]

    # full image decoding is very slow
    # img_dim=512
    # sentence = reshape(im,(img_dim*img_dim))

    # seq_res=256
    # seq_len=len(sentence)
    # num_permutation=512*8
    # num_link=512
    # sigma_range=64

    # partial image decoding
    img_dim= im.shape[0] #256*2
    print("Dimensions de l'image :", im.shape)

    sentence = rg1.reshape(im,(im.shape[0]*im.shape[1])).flatten()
    return sentence, im.size  # (width, height) or (64, 64)


def run_experiment_v1(image_path, 
                      num_link=32, 
                      iterations=20, 
                      matrix_type='LDPC', 
                      noise_level=0, 
                      image_size=(64, 64)):
    """
    Runs the single-layer version (V1) with given parameters.
    Returns (error, elapsed_time).
    """
    # 1) Load image
    image_vector = rg1.array(Image.open(image_path).convert('L'))
    rechape = rg1.reshape(image_vector, (image_vector.shape[0] * image_vector.shape[1])).flatten()
    # image_vector = rechape
    # 2) Build or choose your matrix H based on matrix_type
    #    Example: H = rg1.h_ldpc(seq_len, num_permutation, num_link)
    #    (You need to define seq_len, num_permutation, etc.)
    seq_len = len(rechape)#rg1.reshape(image_vector, (image_vector.shape[0] * image_vector.shape[1])).flatten())
    num_permutation = image_vector.shape[0]  # Example value
    # We'll do a placeholder:
    # if matrix_type == 'LDPC':
    #     H = rg1.h_ldpc(seq_len, num_permutation, num_link)
    # elif matrix_type == 'Gallager':
    #     H = rg1.h_gallager(seq_len, num_permutation, some_k)
    # elif matrix_type == 'Fixed':
    #     H = rg1.h_fixed_connection(...)

    # 3) Run iterative reconstruction
    start_time = time.time()
    H = rg1.h_ldpc(seq_len, num_permutation, num_link)

    output = rg1.iterative_connection(
        seq_res=256,          # Example if 8-bit
        seq_len=seq_len,
        num_permutation=num_permutation,
        sigma_range=(seq_len//4),        # Example param
        sentence=rechape,
        num_link=num_link,
        H=H,
        # max_iter=iterations
    )
    # For demonstration, we simulate an error:
    reconstructed = image_vector  # (Pretend we reconstructed perfectly)
    end_time = time.time()

    elapsed = end_time - start_time
    # print("Time", elapsed)
    # error = rg1.mean(pow(output[1]-output[0],2))
    error = rg1.mean(pow(output[0]-output[1],2))
    # print("Error re. true seq", error)
    # Placeholder error for demonstration:
    # error = np.random.rand() * 10  # Fake random error

    return error, elapsed


def run_experiment_v2(image_path, num_link=32, hidden_size=512, iterations=5, 
                        matrix_type='LDPC', noise_level=0, image_size=(64, 64)):
    """
    Runs the two-layer (V2) reconstruction experiment.
    
    Parameters:
      image_path   : path to the image file.
      num_link     : number of links per neuron.
      hidden_size  : size of the hidden layer (used as num_permutation).
      iterations   : number of iterations for the reconstruction.
      matrix_type  : type of connection matrix (if you have multiple types).
      noise_level  : percentage of noise to add to the image.
      image_size   : tuple indicating the size to which the image will be resized.
      
    Returns:
      (error, elapsed_time)
    """
    # Load and preprocess the image
    # image_vector, _ = load_and_preprocess_image(image_path, size=image_size, noise_level=noise_level)
    image_vector = rg1.array(Image.open(image_path).convert('L'))
    rechape = rg1.reshape(image_vector, (image_vector.shape[0] * image_vector.shape[1])).flatten()
    # image_vector = rechape
    # seq_len = len(image_vector)
    seq_len = len(rechape)#rg2.reshape(image_vector, (image_vector.shape[0] * image_vector.shape[1])).flatten())
    sigma_range = seq_len//4  # You can set or parameterize this as needed.
    num_permutation = image_vector.shape[0]  # Example value
    # Generate the two connection matrices H1 and H2.
    # Here we assume that rg2.h_ldpc returns a matrix with dimensions based on the given parameters.
    # For H1: from input (seq_len) to hidden (hidden_size)
    H1 = rg2.h_ldpc(seq_len, num_permutation, num_link)
    # For H2: from hidden back to output (here we assume it should have dimensions (hidden_size, seq_len))
    H2 = rg2.h_ldpc(seq_len, num_permutation, num_link)
    
    # Start the timer
    start_time = time.time()
    
    # Call the two-layer iterative reconstruction function.
    # Note: The function signature is:
    #   iterative_connection_two_layer(seq_res, seq_len, num_permutation, sigma_range,
    #                                    sentence, num_link, H1, H2, startseq=[], iter_max=5)
    # We pass hidden_size as the num_permutation parameter.
    output = rg2.iterative_connection_two_layer(
        seq_res=256,         # typically 256 for 8-bit images
        seq_len=seq_len,
        num_permutation=num_permutation,
        sigma_range=(seq_len//4),
        sentence=rechape,
        num_link=num_link,
        H1=H1,
        H2=H2,
        iter_max=iterations
    )
    
    end_time = time.time()
    
    # Assuming output[0] is the original sentence and output[1] is the reconstructed one,
    # compute the Mean Squared Error (MSE)
    # error = np.mean((output[1] - output[0]) ** 2)
    error = rg2.mean(pow(output[0]-output[1],2))
    elapsed = end_time - start_time
    return error, elapsed


###############################################################################
# Main param sweep logic                                                     #
###############################################################################

def main():
    image_path = 'srcs/256.png'  # Replace with your actual image path

    # Example parameter sets you might vary:
    # (Feel free to comment out the ones you don't need.)
    num_link_values = [8, 16, 32, 64]
    hidden_sizes = [256, 512, 1024, 2048]
    iteration_values = [5, 10, 20, 30]
    matrix_types = ['LDPC', 'Gallager', 'Fixed']
    noise_levels = [0, 5, 10, 20]  # in %
    image_sizes = [(3,3), (32, 32), (64, 64), (128, 128), (256, 256), (512, 512)]

    # Example: Sweep over num_link for both V1 and V2
    print("=== Sweep over num_link ===")
    print("Format: version, num_link, error, time")
    image_size = rg1.array(Image.open(image_path).convert('L')).shape
    print("image_size: " , image_size)

    for nl in num_link_values:
        # V1
        err_v1, time_v1 = run_experiment_v1(
            image_path=image_path,
            num_link=nl,
            iterations=5,        # fix 20 iterations
            matrix_type='LDPC',   # fix matrix type for now
            noise_level=0,        # no noise
            image_size=image_size  # fix image size
        )
        print(f"V1, num_link={nl}, error={err_v1:.4f}, time={time_v1:.4f}")

        # V2
        # Also choose a hidden layer size for V2, say 512
        err_v2, time_v2 = run_experiment_v2(
            image_path=image_path,
            num_link=nl,
            hidden_size=512,
            iterations=5,
            matrix_type='LDPC',
            noise_level=0,
            image_size= image_size
        )
        print(f"V2, num_link={nl}, error={err_v2:.4f}, time={time_v2:.4f}")

    # You can similarly create loops for hidden_sizes, iteration_values, etc.
    # For instance:
    print("\n=== Sweep over hidden_sizes (V2 only) ===")
    print("Format: hidden_size, error, time")
    for hs in hidden_sizes:
        err, t = run_experiment_v2(
            image_path=image_path,
            num_link=32,      # fix num_link
            hidden_size=hs,
            iterations=5,
            matrix_type='LDPC',
            noise_level=0,
            image_size=(64, 64)
        )
        print(f"hidden_size={hs}, error={err:.4f}, time={t:.4f}")

    # And so on for iteration_values, matrix_types, noise_levels, image_sizes, etc.
    # Just create loops and call the relevant function.

if __name__ == "__main__":
    main()
