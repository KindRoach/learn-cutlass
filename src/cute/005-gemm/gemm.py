import numpy as np

m, n, k = 8, 12, 16
# m, n, k = 64, 96, 128
A = np.arange(m * k).reshape((m, k)) % 255
B_row = np.arange(k * n).reshape((k, n)) % 127
B_col = np.arange(k * n).reshape((n, k)).T % 127

print("Matrix A:", A)
print("\nMatrix B:", B_row)
print("\nMatrix C = (A @ B):", A @ B_row)
print("\nMatrix C_col = (A @ B_col):", A @ B_col)
