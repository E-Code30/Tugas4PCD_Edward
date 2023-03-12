import numpy as np

def dft(x):
    n = len(x)
    x = np.asarray(x, dtype=np.complex_)
    m = np.zeros((n,n), dtype=np.complex_)
    for i in range(n):
        for j in range(n):
            m[i][j] = np.exp(-2j * np.pi * i * j / n)
    return np.dot(m, x)

x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
print("Input:", x)
y = dft(x)
print("Output:", y)
