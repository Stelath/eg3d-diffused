import numpy as np

def create_attention_matrix(r, c, n=10000):
    ch = int(c/2)
    p = np.zeros((r, c))
    for k in range(r):
        for i in np.arange(int(c/4)):
            denominator = np.power(n, 2*i/c)
            p[k, 2*i+ch] = np.sin(k/denominator)
            p[k, 2*i+1+ch] = np.cos(k/denominator)
            p[k, ch-2*i] = p[k, 2*i+ch]
            p[k, ch-2*i+1] = p[k, 2*i+1+ch]
            
    return p

def decode_attention_matrix(matrix, size):
    attention_matrix = create_attention_matrix(size, size)
    decoded_matrix = ((matrix * 2) - attention_matrix) * 3.14
    
    return decoded_matrix
