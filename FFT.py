import numpy as np

def fft(x):
    n = len(x)
    if n == 1:
        return x
    else:
        even = fft(x[0::2])
        odd = fft(x[1::2])

        combined = np.zeros(n, dtype=np.complex_)
        for i in range(n // 2):
            twiddle = np.exp(-2j * np.pi * i / n)
            combined[i] = even[i] + twiddle * odd[i]
            combined[i + n // 2] = even[i] - twiddle * odd[i]

        return combined

x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
print("Input:", x)
y = fft(x)
print("Output:", y)
