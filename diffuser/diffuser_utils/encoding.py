import numpy as np

def create_attention_matrix(seq_len, d, n=10000):
    p = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            p[k, 2*i] = np.sin(k/denominator)
            p[k, 2*i+1] = np.cos(k/denominator)
    return p

def decode_attention_matrix(matrix, size):
    attention_matrix = create_attention_matrix(size, size)
    decoded_matrix = ((matrix * 2) - attention_matrix) * 3.14
    
    return decoded_matrix
