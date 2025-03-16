from rank_generalisation_2layer import *
import time
#LOAD IMAGE
from PIL import Image
# import peg
import pickle

im = array(Image.open('Lenna.png').convert('L'))
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
img_dim = im.shape[0]  # 256*2 (for a 512x512 image, this would be 512)
print("Dimensions de l'image :", im.shape)

sentence = reshape(im, (im.shape[0] * im.shape[1])).flatten()
print("Sentence :", sentence)

seq_res = 256
seq_len = len(sentence)
num_permutation = im.shape[0] * 8
num_link = im.shape[0]  # 256*2
sigma_range = seq_res // 4

figure()
imshow(reshape(sentence[::-1], (img_dim, img_dim)), cmap='gray', origin='lower', aspect='auto', interpolation='none')

t0 = time.time()

# Define first-layer and second-layer connection matrices
H1 = h_ldpc(seq_len, num_permutation, num_link)
H2 = h_ldpc(seq_len, num_permutation, num_link)
#H = h_gallager(seq_len, num_permutation, num_link)  # alternative single-layer matrix
#H1 = h_gallager(seq_len, num_permutation, num_link)
#H2 = h_gallager(seq_len, num_permutation, num_link)
# (Alternatively, use Gallager or other methods for H2 to introduce different constraints)

# for PEG matrices (single-layer usage)
store_matrix = False
if store_matrix:
    # compute the peg structure which contains the matrix H
    pp= peg.peg(seq_len, num_permutation, [num_link] * seq_len)
    pp.progressive_edge_growth()
    file=open("H_peg_%d_%d_%d.p" % (seq_len, num_permutation, num_link), "wb")
    pickle.dump(pp.H, file)
    file.close()

# file=open("H_peg_%d_%d_%d.p" % (seq_len, num_permutation, num_link), "rb")
# H=pickle.load(file)
# file.close()

# Use two-layer iterative decoding (original single-layer call is commented out below for comparison)
# output = iterative_connection(seq_res, seq_len, num_permutation, sigma_range, sentence, num_link, H)
# output = iterative_rank(seq_res, seq_len, num_permutation, sigma_range, sentence)
output_two = iterative_connection_two_layer(seq_res, seq_len, num_permutation, sigma_range, sentence, num_link, H1, H2)

t1 = time.time()
print("Time (two-layer)", t1 - t0)
print("Error re. true seq (two-layer)", mean(pow(output_two[1] - output_two[0], 2)))

figure()
imshow(reshape(output_two[1][::-1], (img_dim, img_dim)), cmap='gray', origin='lower', aspect='auto', interpolation='none')

print("output_two [0]", output_two[0])
print("output_two [1]", output_two[1])
