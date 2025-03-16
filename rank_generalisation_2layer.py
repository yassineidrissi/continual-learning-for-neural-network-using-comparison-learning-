from matplotlib import *
from matplotlib.pylab import *
from PIL import Image

def gaussian(x, mu, sig):
    return exp(-pow(x - mu, 2.) / (2 * pow(sig, 2.)))

# function that constructs a random matrix of connections H in the case of LDPC codes. 
# num_link is the number of synapses per neuron
# seq_len is the length of the sequence and num_permutation is the number of neurons for the NN
def h_ldpc(seq_len, num_permutation, num_link):
    # define parity matrix H
    h = zeros((seq_len))
    h[0:num_link] = 1
    H = zeros((num_permutation, seq_len))
    for idx in range(num_permutation):
        H[idx, :] = permutation(h)
    return H

# function that constructs a matrix of connections as defined by Gallager for the LDPC codes
# n is the length of the sequence
# k is the number of synapses per neuron
# l is the number of neurons = permutations
# j is the number of neurons connected to each item
def h_gallager(n, num_perm, k):
    l = num_perm
    j = l * k // n  # l = j*(n//k)
    if j * n != l * k:
        print("Error: wrong parameters for Gallager LDPC matrix")
        print(l * k, "edges in", j * n, "edges out")
        return
    H = zeros((l, n))
    for i in range(1, n // k + 1):
        H[i - 1, (i - 1) * k : i * k] = 1
    idx = [i for i in range(n)]
    for line in range(1, j):
        idx = permutation(idx)
        for i in range(n):
            H[l // j * line : l // j * (line + 1), i] = H[: l // j, idx[i]]
    return H

# function that constructs a matrix of connections H to allow neurons to learn a chunk of the sequence
# position is the list of indices of the middle of the chunks for each neuron
# len_view is the half-length of the chunks (on each side of the middle)
# seq_len is the length of the sequence
# num_permutation is the number of neurons
def h_fixed_connection(position, len_view, seq_len, num_permutation):
    H = zeros((num_permutation, seq_len))
    for idx in range(len(position)):
        l_min = position[idx] - len_view
        l_max = position[idx] + len_view
        if l_min < 0:
            l_min = 0
        if l_max > seq_len:
            l_max = seq_len
        H[idx, l_min:l_max] = 1
    return H

# this is the neural network which takes into account the matrix H to configure 
# the connections between the sequence and the neurons (single-layer method)
def iterative_connection(seq_res, seq_len, num_permutation, sigma_range, sentence, num_link, H, startseq=[], iter_max=5):
    # min max alphabet index value
    amin_val = 0
    amax_val = seq_res
    # 2.32 Permutation key [0,amax_val] -> [amin_val,amax_val] : [0,99]+1 -> [1,100]
    # Permutation key idx location => new index location
    permutation_key = zeros((num_permutation, amax_val), dtype=int32)
    edges = []  # indices of seq items connected to neuron idx
    for idx in range(num_permutation):
        permutation_key[idx, :] = permutation(amax_val)  # permutation cree un tableau avec tous les nombres 
        # jusqu'a amax_val dans des ordres différents (alphabet)
        edges.append(np.where(H[idx, :] == 1)[0])
    # the last permutation key is the original order (for analysis)
    permutation_key[-1, :] = arange(amax_val)
    # choose ONE sentence only in the database
    # sentence = randint(amax_val, size=seq_len) # defini une sequence aléatoire si besoin
    inv_permutation = argsort(permutation_key, axis=1)  # array of indices that would sort each permutation row (inverse permutation)
    # inits (-1 ranks to check for access errors)
    ranks_output = -ones((num_permutation, num_link), dtype=int32)
    for idx in arange(num_permutation):
        liste = sentence[edges[idx]]  # each neuron learns only the ranks of the items connected to it
        input_perm = permutation_key[idx, liste]  # for each permutation, get indices of connected items in that permuted order
        rank_order_idx = argsort(input_perm)  # rank order of the connected items (smallest first in permuted alphabet)
        ranks_output[idx, 0:len(rank_order_idx)] = rank_order_idx
    # 7.1 decoding, start with random seq
    s_trial = zeros((num_permutation, seq_len), dtype=int32)
    s_hat = randint(amax_val, size=seq_len)  # initial estimation of the sequence
    err = 1
    iter = 1
    while (err > 1e-6) and (iter <= iter_max):
        s_hat0 = s_hat  # store current estimate to compute error
        # 7.2a CN operation (Check Nodes)
        # Sort the elements in each neuron's connected subset according to that neuron's permutation ranks
        for pk in arange(num_permutation):
            # each neuron sorts the values of its connected items in its permuted alphabet space
            s_hat_perm_sort = sort(permutation_key[pk, s_hat[edges[pk]]])
            # put the sorted values back into the sequence at the correct positions for this neuron
            solution = edges[pk][ranks_output[pk, :]]
            s_trial[pk, solution] = s_hat_perm_sort
        # 7.2b VN operation (Variable Nodes)
        mat_sum = zeros((amax_val, seq_len))
        gg = gaussian(arange(0, 2 * sigma_range + 1), mu=sigma_range, sig=sigma_range)
        # 1. retrieve for each permutation alphabet ...
        for perm_key in arange(num_permutation):
            for idx in edges[perm_key]:
                sval = s_trial[perm_key, idx]
                # 3. find then its neighbouring indices
                born_max = min(amax_val, sval + sigma_range + 1)  # +1 for symmetry
                born_min = max(amin_val, sval - sigma_range)
                # 4. retrieve values in the original alphabet
                inv_val = inv_permutation[perm_key, int(born_min):int(born_max)]
                # 5. follow a gaussian density for estimation with respect to their distance to sval
                mat_sum[inv_val, idx] += gg[born_min - sval + sigma_range : 2 * sigma_range + 1 + born_max - sval - sigma_range - 1]
        # VN decision
        s_hat = argmax(mat_sum, axis=0)
        err = mean(pow(s_hat - s_hat0, 2))
        # print("iter %d : cur err %f" % (iter, err))
        iter += 1
    # return original sequence and the reconstructed sequence
    return sentence, s_hat

# this is the extended neural network that uses two layers of connection matrices (H1 and H2) to refine the decoding process
# The first layer uses H1 and the second layer uses H2 sequentially in each iteration.
# The original single-layer method (iterative_connection) is preserved above; this two-layer method is implemented for comparison.
def iterative_connection_two_layer(seq_res, seq_len, num_permutation, sigma_range, sentence, num_link, H1, H2, startseq=[], iter_max=5):
    # min max alphabet index value
    amin_val = 0
    amax_val = seq_res
    # Permutation keys and connection edges for first layer (H1)
    permutation_key1 = zeros((num_permutation, amax_val), dtype=int32)
    edges1 = []
    for idx in range(num_permutation):
        permutation_key1[idx, :] = permutation(amax_val)
        edges1.append(np.where(H1[idx, :] == 1)[0])
    permutation_key1[-1, :] = arange(amax_val)
    inv_permutation1 = argsort(permutation_key1, axis=1)
    # Permutation keys and connection edges for second layer (H2)
    permutation_key2 = zeros((num_permutation, amax_val), dtype=int32)
    edges2 = []
    for idx in range(num_permutation):
        permutation_key2[idx, :] = permutation(amax_val)
        edges2.append(np.where(H2[idx, :] == 1)[0])
    permutation_key2[-1, :] = arange(amax_val)
    inv_permutation2 = argsort(permutation_key2, axis=1)
    # Compute rank order (ground truth) for each neuron in both layers
    ranks_output1 = -ones((num_permutation, num_link), dtype=int32)
    ranks_output2 = -ones((num_permutation, num_link), dtype=int32)
    for idx in arange(num_permutation):
        # First layer: rank order of actual values for neuron idx
        liste1 = sentence[edges1[idx]]
        input_perm1 = permutation_key1[idx, liste1]
        rank_order_idx1 = argsort(input_perm1)
        ranks_output1[idx, 0:len(rank_order_idx1)] = rank_order_idx1
        # Second layer: rank order of actual values for neuron idx
        liste2 = sentence[edges2[idx]]
        input_perm2 = permutation_key2[idx, liste2]
        rank_order_idx2 = argsort(input_perm2)
        ranks_output2[idx, 0:len(rank_order_idx2)] = rank_order_idx2
    # 7.1 decoding (initialization)
    s_trial1 = zeros((num_permutation, seq_len), dtype=int32)
    s_trial2 = zeros((num_permutation, seq_len), dtype=int32)
    s_hat = randint(amax_val, size=seq_len)  # initial sequence estimation
    err = 1
    iter = 1
    while (err > 1e-6) and (iter <= iter_max):
        s_hat0 = s_hat  # store current estimate
        # 7.2a1 CN operation (Layer 1 Check Nodes)
        # Apply first-layer constraints: sort each neuron's connected items as per H1
        for pk in arange(num_permutation):
            s_hat_perm_sort1 = sort(permutation_key1[pk, s_hat[edges1[pk]]])
            solution1 = edges1[pk][ranks_output1[pk, :]]
            s_trial1[pk, solution1] = s_hat_perm_sort1
        # 7.2a2 CN operation (Layer 2 Check Nodes)
        # Apply second-layer constraints: sort each neuron's connected items as per H2
        for pk in arange(num_permutation):
            s_hat_perm_sort2 = sort(permutation_key2[pk, s_hat[edges2[pk]]])
            solution2 = edges2[pk][ranks_output2[pk, :]]
            s_trial2[pk, solution2] = s_hat_perm_sort2
        # 7.2b VN operation (Variable Nodes) combining both layers
        mat_sum = zeros((amax_val, seq_len))
        gg = gaussian(arange(0, 2 * sigma_range + 1), mu=sigma_range, sig=sigma_range)
        # Combine contributions from first-layer neurons
        for perm_key in arange(num_permutation):
            for idx in edges1[perm_key]:
                sval = s_trial1[perm_key, idx]
                # 3. find then its neighbouring indices (layer 1)
                born_max = min(amax_val, sval + sigma_range + 1)
                born_min = max(amin_val, sval - sigma_range)
                # 4. retrieve values in the original alphabet (layer 1)
                inv_val = inv_permutation1[perm_key, int(born_min):int(born_max)]
                # 5. follow gaussian density for estimation (layer 1)
                mat_sum[inv_val, idx] += gg[born_min - sval + sigma_range : 2 * sigma_range + 1 + born_max - sval - sigma_range - 1]
        # Combine contributions from second-layer neurons
        for perm_key in arange(num_permutation):
            for idx in edges2[perm_key]:
                sval = s_trial2[perm_key, idx]
                # 3. find then its neighbouring indices (layer 2)
                born_max = min(amax_val, sval + sigma_range + 1)
                born_min = max(amin_val, sval - sigma_range)
                # 4. retrieve values in the original alphabet (layer 2)
                inv_val = inv_permutation2[perm_key, int(born_min):int(born_max)]
                # 5. follow gaussian density for estimation (layer 2)
                mat_sum[inv_val, idx] += gg[born_min - sval + sigma_range : 2 * sigma_range + 1 + born_max - sval - sigma_range - 1]
        # VN decision (update sequence estimation)
        s_hat = argmax(mat_sum, axis=0)
        err = mean(pow(s_hat - s_hat0, 2))
        # print("iter %d : cur err %f" % (iter, err))
        iter += 1
    # return original sequence and the reconstructed sequence
    return sentence, s_hat
