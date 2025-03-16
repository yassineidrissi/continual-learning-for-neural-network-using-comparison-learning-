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
    img = Image.open(image_path).convert('L').resize(size)
    img_array = np.array(img, dtype=np.float32)  # shape: (height, width)

    # Optionally add noise:
    if noise_level > 0:
        # Example: add Gaussian noise
        noise_std = (noise_level / 100.0) * 255
        noise = np.random.normal(0, noise_std, img_array.shape)
        img_array += noise
        img_array = np.clip(img_array, 0, 255)

    # Flatten to 1D
    flattened = img_array.flatten()
    return flattened, img.size  # (width, height) or (64, 64)


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
    image_vector, _ = load_and_preprocess_image(image_path, size=image_size, noise_level=noise_level)

    # 2) Build or choose your matrix H based on matrix_type
    #    Example: H = rg1.h_ldpc(seq_len, num_permutation, num_link)
    #    (You need to define seq_len, num_permutation, etc.)
    seq_len = image_vector.shape[0]
    num_permutation = image_vector.shape[0]*8  # Example value
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
        sigma_range=3,        # Example param
        sentence=image_vector,
        num_link=num_link,
        H=H,
        # max_iter=iterations
    )
    # For demonstration, we simulate an error:
    reconstructed = image_vector  # (Pretend we reconstructed perfectly)
    end_time = time.time()

    elapsed = end_time - start_time
    # print("Time", elapsed)
    error = rg1.mean(pow(output[1]-output[0],2))
    # print("Error re. true seq", error)
    # Placeholder error for demonstration:
    # error = np.random.rand() * 10  # Fake random error

    return error, elapsed


def run_experiment_v2(image_path, 
                      num_link=32, 
                      hidden_size=512, 
                      iterations=20, 
                      matrix_type='LDPC', 
                      noise_level=0, 
                      image_size=(64, 64)):
    """
    Runs the two-layer version (V2) with given parameters.
    Returns (error, elapsed_time).
    """
    # 1) Load image
    image_vector, _ = load_and_preprocess_image(image_path, size=image_size, noise_level=noise_level)

    seq_len = image_vector.shape[0]
    # 2) Build or choose your matrices H1, H2
    #    Example:
    H1 = rg2.h_ldpc(seq_len, hidden_size, num_link)
    H2 = rg2.h_ldpc(hidden_size, seq_len, num_link)
    #    or depends on matrix_type, etc.

    start_time = time.time()
    output = rg2.iterative_connection_two_layer(
        seq_res=256,
        seq_len=seq_len,
        num_permutation=seq_len,
        # hidden_size=hidden_size,
        sigma_range=seq_len/4,
        sentence=image_vector,
        num_link=num_link,
        H1=H1,
        H2=H2,
        # max_iter=iterations
    )
    # For demonstration, we simulate an error:
    reconstructed = image_vector
    end_time = time.time()

    elapsed = end_time - start_time
    # error = np.mean((output[1] - output[0])**2)  # Example MSE
    error = np.random.rand() * 5  # Fake random error

    return error, elapsed


###############################################################################
# Main param sweep logic                                                     #
###############################################################################

def main():
    image_path = '64.png'  # Replace with your actual image path

    # Example parameter sets you might vary:
    # (Feel free to comment out the ones you don't need.)
    num_link_values = [8, 16, 32, 64]
    hidden_sizes = [256, 512, 1024, 2048]
    iteration_values = [5, 10, 20, 30]
    matrix_types = ['LDPC', 'Gallager', 'Fixed']
    noise_levels = [0, 5, 10, 20]  # in %
    image_sizes = [(32, 32), (64, 64), (128, 128)]

    # Example: Sweep over num_link for both V1 and V2
    print("=== Sweep over num_link ===")
    print("Format: version, num_link, error, time")

    for nl in num_link_values:
        # V1
        err_v1, time_v1 = run_experiment_v1(
            image_path=image_path,
            num_link=nl,
            iterations=20,        # fix 20 iterations
            matrix_type='LDPC',   # fix matrix type for now
            noise_level=0,        # no noise
            image_size=(64, 64)   # fix image size
        )
        print(f"V1, num_link={nl}, error={err_v1:.4f}, time={time_v1:.4f}")

        # V2
        # Also choose a hidden layer size for V2, say 512
        err_v2, time_v2 = run_experiment_v2(
            image_path=image_path,
            num_link=nl,
            hidden_size=512,
            iterations=20,
            matrix_type='LDPC',
            noise_level=0,
            image_size=(64, 64)
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
            iterations=20,
            matrix_type='LDPC',
            noise_level=0,
            image_size=(64, 64)
        )
        print(f"hidden_size={hs}, error={err:.4f}, time={t:.4f}")

    # And so on for iteration_values, matrix_types, noise_levels, image_sizes, etc.
    # Just create loops and call the relevant function.

if __name__ == "__main__":
    main()
