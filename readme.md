# Rank-Based Autoencoder (Single-Layer and Two-Layer)

This repository contains code for an image-reconstruction project using a rank-based approach and random dictionary (LDPC-like) connections. It includes both:

1. **V1 (Single-Layer)**: A single-layer autoassociative memory.  
2. **V2 (Two-Layer)**: An extended two-layer (autoencoder-style) architecture for improved reconstruction.

---

## Table of Contents
- [Project Idea](#project-idea)
- [Directory Structure](#directory-structure)
- [Installation and Requirements](#installation-and-requirements)
- [Usage](#usage)
- [Code Details](#code-details)
- [References / Further Reading](#references--further-reading)

---

## Project Idea

The main goal of this project is to explore an **iterative, rank-based reconstruction** method for images. The core concept:

- **Random Dictionary / LDPC-Like Connections**: Each neuron connects to a subset of image pixels (or the previous layer’s neurons) using a sparse matrix.
- **Rank-Based Updating**: Neurons iteratively “enforce” the rank order of pixel intensities, converging to a reconstructed image even if the initial input is incomplete or noisy.
- **Single-Layer (V1)** vs. **Two-Layer (V2)**:
  - V1 uses a single connection matrix `H`.
  - V2 introduces two matrices (`H1`, `H2`) for a more robust, autoencoder-like approach, which can improve reconstruction quality but at higher computational cost.

---

## Directory Structure

Below is an example of how the files might be organized:


*(Adjust as needed to match your actual files.)*

---

## Installation and Requirements

1. **Python Version**: Python 3.9+ recommended.
2. **Dependencies**:
   - [NumPy](https://numpy.org/)
   - [Matplotlib](https://matplotlib.org/)
   - [Pillow (PIL)](https://pillow.readthedocs.io/en/stable/)

Install via:

    pip install numpy matplotlib pillow

---


### Installation and Requirements

1. **Python Version**:  
   Python 3.9 or higher is recommended.

2. **Dependencies**:  
   This project requires the following libraries:
   - [NumPy](https://numpy.org/)
   - [Matplotlib](https://matplotlib.org/)
   - [Pillow (PIL)](https://pillow.readthedocs.io/en/stable/)

   To install these dependencies, run:

   ```bash
   pip install numpy matplotlib pillow


### How to Use This
1. **Rename** your `rank_generalisation_v2.py` file to `rank_generalisation_2layer.py`.
2. **Create** a new `README.md` file in your repository’s root folder.
3. **Copy and paste** the above markdown into `README.md`.
4. **Adjust** any filenames, paths, or descriptions to match your actual code structure.

This way, anyone who clones your project will have a clear idea of what each file does, how to install dependencies, and how to run your scripts.


### How to Use This
1. **Rename** your `rank_generalisation_v2.py` file to `rank_generalisation_2layer.py`.
2. **Create** a new `README.md` file in your repository’s root folder.
3. **Copy and paste** the above markdown into `README.md`.
4. **Adjust** any filenames, paths, or descriptions to match your actual code structure.

This way, anyone who clones your project will have a clear idea of what each file does, how to install dependencies, and how to run your scripts.

# Rank-Based Autoencoder (Single-Layer and Two-Layer)

This repository contains code for an image-reconstruction project using a rank-based approach and random dictionary (LDPC-like) connections. It includes both:

1. **V1 (Single-Layer)**: A single-layer autoassociative memory.  
2. **V2 (Two-Layer)**: An extended two-layer (autoencoder-style) architecture for improved reconstruction.

---

## Table of Contents
- [Project Idea](#project-idea)
- [Directory Structure](#directory-structure)
- [Installation and Requirements](#installation-and-requirements)
- [Usage](#usage)
- [Code Details](#code-details)
- [References / Further Reading](#references--further-reading)

---

## Project Idea

The main goal of this project is to explore an **iterative, rank-based reconstruction** method for images. The core concept:

- **Random Dictionary / LDPC-Like Connections**: Each neuron connects to a subset of image pixels (or the previous layer’s neurons) using a sparse matrix.
- **Rank-Based Updating**: Neurons iteratively “enforce” the rank order of pixel intensities, converging to a reconstructed image even if the initial input is incomplete or noisy.
- **Single-Layer (V1)** vs. **Two-Layer (V2)**:
  - V1 uses a single connection matrix `H`.
  - V2 introduces two matrices (`H1`, `H2`) for a more robust, autoencoder-like approach, which can improve reconstruction quality but at higher computational cost.

---
