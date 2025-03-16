import matplotlib.pyplot as plt
import numpy as np

# Define image dimensions (width of the square image)
dims = np.array([3, 32, 256, 512])

# Execution times in seconds for V1 and V2
v1_time = np.array([0.00788, 1.10713, 69.22369, 473.64856])
v2_time = np.array([0.008186, 1.39389, 149.56846, 668.69722])

# Reconstruction errors for V1 and V2
v1_error = np.array([1924.44444, 5099.34473, 78.22989, 12.95787])
v2_error = np.array([5752.66667, 771.77637, 1.13992, 0.0])

plt.figure(figsize=(12, 5))

# Plot execution times
plt.subplot(1, 2, 1)
plt.plot(dims, v1_time, 'o-', label='V1 Temps')
plt.plot(dims, v2_time, 's-', label='V2 Temps')
plt.xlabel('Dimension de l\'image (n x n)')
plt.ylabel('Temps (secondes)')
plt.title('Temps d\'exécution vs Dimension')
plt.legend()
plt.xscale('log')
plt.yscale('log')

# Plot reconstruction errors
plt.subplot(1, 2, 2)
plt.plot(dims, v1_error, 'o-', label='V1 Erreur')
plt.plot(dims, v2_error, 's-', label='V2 Erreur')
plt.xlabel('Dimension de l\'image (n x n)')
plt.ylabel('Erreur moyenne')
plt.title('Erreur de reconstruction vs Dimension')
plt.legend()
plt.xscale('log')
plt.yscale('log')

plt.tight_layout()
plt.show()
