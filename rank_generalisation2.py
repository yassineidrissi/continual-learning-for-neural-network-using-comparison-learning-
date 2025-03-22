# this file contains the various creation functions of H for the generalization of the connections between the sequence and the neurons. there is also the neural network which only encodes a certain number of items per neuron, defined by the matrix H constructed with the previous functions.

from matplotlib import *
from matplotlib.pylab import *
from PIL import Image

def gaussian(x, mu, sig):
    return exp(-pow(x - mu, 2.) / (2 * pow(sig, 2.)))

# function that contructs a random matrix of connections H in the case of LDPC codes. 
# num_links is the number of synapses per neuron
# seq_len is the length of the sequence and num_permutation is the number of neurons for the NN
def h_ldpc(seq_len, num_permutation, num_link):
        #define parity matrix H
        h = zeros((seq_len))
        h[0:num_link]=1
        H = zeros((num_permutation,seq_len))
        for idx in range(num_permutation):
                H[idx,:]=permutation(h)

        return H
        
# function that constructs a matrix of connections as defined by Gallager for the LDPC codes
# n is the length of the sequence
# k is the number of synapses per neuron
# l is the number of neurons = permutations
# j is the number of neurons connected to each item
def h_gallager(n,num_perm,k):
        l = num_perm
        j = l*k//n # l = j*(n//k)
        if j*n != l*k:
            print("Error: wrong parameters for Gallager LDPC matrix")
            print(l*k,"edges in",j*n,"edges out")
            return

        H = zeros((l,n))

        for i in range(1,n//k+1):
                H[i-1,(i-1)*k:i*k] = 1

        idx = [i for i in range(n)]
        for line in range(1,j):
                        idx = permutation(idx)
                        for i in range(n):
                                H[l//j*line:l//j*(line+1),i] = H[:l//j,idx[i]]
                        
        return H

# function that constructs a matrix of connections H in order to allows neurons to learns a chunk of the sequence
# position is the list of index of the middle of the chunks for each neuron
# len_view is the length of the chunks
# seq_len is the length of the sequence
# num_permutation is the number of neurons
def h_fixed_connection(position, len_view, seq_len, num_permutation):
        H = zeros((num_permutation,seq_len))
        for idx in range(len(position)):
                l_min = position[idx]-len_view
                l_max = position[idx]+len_view
                if l_min < 0:
                        l_min = 0
                if l_max > seq_len:
                        l_max = seq_len
                        
                H[idx,l_min:l_max]=1
        
        return H

# this is the neural network which takes into account the matrix H to configure the connections between the sequence and the neurons
def iterative_connection(seq_res,seq_len,num_permutation,sigma_range,sentence,num_link,H,startseq=[],iter_max=5):
        print("V1 the parametre value is ", seq_res, "seq_len", seq_len, "num_permutation", num_permutation, " sigma_range ", sigma_range, " iter_max ", iter_max)
        # min max alphabet index value
        amin_val=0
        amax_val=seq_res
        #2.32 Permutation key [0,amax_val]->[amin_val,amax_val] : [0,99]+1->[1,100]
        # Permutation key idx location => new index location
        permutation_key=zeros((num_permutation,amax_val),dtype=int32)
        edges = [] # indices of seq items connected to neuron idx
        for idx in range(num_permutation): 
                permutation_key[idx,:]=permutation(amax_val) # permutation cree un tableau avec tous les nombres 
                #jusqu'a amaxval dans des ordres différents (alphabet)
                edges.append(np.where(H[idx,:]==1)[0])

        # the last permutation key is the original order (for analysis)
        permutation_key[-1,:] =arange(amax_val) # remets les elements dans l'ordre

        # choose ONE sentence only in the database
        #sentence=randint(amax_val, size=seq_len) # defini une sequence

        inv_permutation=argsort(permutation_key, axis=1) # array of indices of the same shape as arg that would sort the array. 
        #It means indices of value arranged in ascending order

        #inits (-1 ranks to check for access errors)
        ranks_output = -ones((num_permutation,num_link),dtype=int32)

        for idx in arange(num_permutation): 
                liste = sentence[edges[idx]] # each neuron learns only the ranks of the items connected to it
                liste = liste.astype(int)
                # print("liste dtype:", liste.dtype)
                input_perm=permutation_key[idx,liste] #for each permutation order, outputs the indices of item connected with the constraint
                rank_order_idx=argsort(input_perm) # the most active first rank, rank order of the sequence mapped in the permutation
                ranks_output[idx,0:len(rank_order_idx)]= rank_order_idx
                

        # 7.1 decoding, start with random seq
        s_trial=zeros((num_permutation,seq_len),dtype=int32)

        s_hat=randint(amax_val,size=seq_len) # estimation of the sequence

        err = 1
        iter = 1
        while (err>1e-6) and (iter<=iter_max):
                s_hat0 = s_hat #each time the estimated sequence is changed
                # 7.2a CN operation Check Nodes
                # Mettre les elements dans le bon ordre
                # s_trial will contain s_hat sorted s.t. it satisfies the ranks

                for pk in arange(num_permutation): # pour i de 1 a num_permutation
                    # cw 11.5.23 was a mistake to sort the entire sequence,
                    # since each neuron may sort only its connected items
                    # >>> works better & faster on image example
                    s_hat_perm_sort = sort(permutation_key[pk,s_hat[edges[pk]]]) 
                    # remet la sequence dans le bon ordre
                    # # only the items connected to the neuron are modified
                    solution = edges[pk][ranks_output[pk,:]]
                    s_trial[pk,solution] = s_hat_perm_sort
                    
                # 7.2b VN operation Variable Nodes 
                mat_sum=zeros((amax_val,seq_len))

                gg = gaussian(arange(0,2*sigma_range+1),mu=sigma_range,sig=sigma_range)
                
                # 1. retrieve for each permutation alphabet ...
                for perm_key in arange(num_permutation):
                        for idx in edges[perm_key]:
                                sval=s_trial[perm_key,idx]
                                # 3. find then its neighbouring indices
                                born_max = min(amax_val,sval+sigma_range+1) # +1 for symmetry
                                born_min = max(amin_val,sval-sigma_range)
                                # 4. retrieve values in the original alphabet
                                inv_val=inv_permutation[perm_key,int(born_min):int(born_max)]
                                # 5. follow a gaussian density for estimation with respect to their distance to sval
                                mat_sum[inv_val,idx] += gg[born_min-sval+sigma_range:2*sigma_range+1+born_max-sval-sigma_range-1]
                                
                # VN decision
        
                #print(s_hat)
                s_hat = argmax(mat_sum,axis=0)
                err = mean(pow(s_hat-s_hat0,2))
                #figure()
                #imshow(reshape(s_hat[::-1],(512,512)), cmap='gray', origin='lower', aspect='auto',interpolation='none')

                #print ("iter %d : cur err %f" %( iter, err))
                iter += 1
                
        #c = imshow(mat_sum, origin='lower', aspect='auto',interpolation='none')
        #show()
        return sentence,s_hat
