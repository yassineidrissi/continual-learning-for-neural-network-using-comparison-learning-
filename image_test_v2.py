from rank_generalisation2 import *
import time
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import numpy as np
from numpy import array, reshape, arange

# Load image and convert to grayscale
im = array(Image.open('srcs/256.png').convert('L'))
img_dim = im.shape[0]
print("Dimensions de l'image :", im.shape)

# Reshape the image into a 1D sentence vector
sentence = reshape(im, (im.shape[0]*im.shape[1])).flatten()
print("Sentence :", sentence)

# Parameters
seq_res = 256
seq_len = len(sentence)
num_permutation = im.shape[0] * 8
num_link = im.shape[0]
sigma_range = seq_res // 4

# (Optional) Display the original image
plt.figure()
plt.imshow(reshape(sentence[::-1], (img_dim, img_dim)),
           cmap='gray', origin='lower', aspect='auto', interpolation='none')
plt.title("Image Originale")
plt.axis('off')
plt.show()

t0 = time.time()
H = h_ldpc(seq_len, num_permutation, num_link)
# iterative_connection is assumed to now return (original, final_recon, snapshots)
output = iterative_connection(seq_res, seq_len, num_permutation, sigma_range, sentence, num_link, H)
t1 = time.time()

print("Time:", t1 - t0)
print("Error re. true seq:", np.mean(np.power(output[1] - output[0], 2)))

# Unpack the outputs (original, final reconstruction, snapshots at each iteration)
original, final_recon, snapshots = output

# Determine layout for snapshots:
n_snapshots = len(snapshots)
if n_snapshots <= 3:
    nrows = 1
    ncols = n_snapshots
else:
    nrows = 2
    ncols = 3
    if n_snapshots > 6:
        snapshots = snapshots[:6]  # show only first 6 iterations
        n_snapshots = 6

# Create one combined figure for all snapshots
fig, axs = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
# Ensure axs is always iterable
if n_snapshots == 1:
    axs = [axs]
else:
    axs = np.array(axs).flatten()

for i, snapshot in enumerate(snapshots):
    axs[i].imshow(reshape(snapshot[::-1], (img_dim, img_dim)),
                  cmap='gray', origin='lower', aspect='auto', interpolation='none')
    axs[i].set_title(f"Reconstruction - Étape {i+1}")
    axs[i].axis('off')

# Hide any extra subplots if present
for j in range(n_snapshots, nrows*ncols):
    axs[j].axis('off')

plt.tight_layout()
plt.savefig("combined_snapshots.png")
plt.show()
