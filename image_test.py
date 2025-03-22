from rank_generalisation2 import *
import time
#LOAD IMAGE
from PIL import Image
# import peg
import pickle


im = array(Image.open('srcs/256.png').convert('L')) #you can pass multiple arguments in single line

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

sentence = reshape(im,(im.shape[0]*im.shape[1])).flatten()
print("Sentence :", sentence)

seq_res= 256
seq_len= len(sentence)
num_permutation=im.shape[0]*8
num_link= im.shape[0]#*2 #im.shape[1] #256*2
sigma_range=seq_res//4

figure()
imshow(reshape(sentence[::-1],(img_dim,img_dim)), cmap='gray', origin='lower', aspect='auto',interpolation='none')


t0=time.time()

H = h_ldpc(seq_len, num_permutation, num_link)
#H = h_gallager(seq_len, num_permutation, num_link)

# for PEG matrices
store_matrix = False
if store_matrix:
    #compute the peg structure which contains the matrix H
    pp= peg.peg(seq_len, num_permutation, [num_link]*seq_len)
    pp.progressive_edge_growth()
    file=open( "H_peg_%d_%d_%d.p" %(seq_len,num_permutation,num_link),"wb")
    pickle.dump(pp.H, file)
    file.close()

# file=open( "H_peg_%d_%d_%d.p" %(seq_len,num_permutation,num_link),"rb")
# H=pickle.load()
# file.close()
    
output=iterative_connection(seq_res,seq_len,num_permutation,sigma_range,sentence,num_link,H)

#output=iterative_rank(seq_res,seq_len,num_permutation,sigma_range,sentence)

t1=time.time()
print("Time", t1-t0)
print("Error re. true seq", mean(pow(output[1]-output[0],2)))

figure()
imshow(reshape(output[1][::-1],(img_dim,img_dim)), cmap='gray', origin='lower', aspect='auto',interpolation='none')

print("output [0]" ,output[0])
print("output [1]" ,output[1])
