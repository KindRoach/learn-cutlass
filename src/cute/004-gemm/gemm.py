import numpy as np

m, n, k = 4, 6, 8
# m, n, k = 64, 96, 128
A = np.arange(m * k).reshape((m, k)) % 255
B = np.arange(k * n).reshape((k, n)) % 127

C = A @ B

print("Matrix A:", A)
print("\nMatrix B:", B)
print("\nMatrix C (A @ B):", C)
