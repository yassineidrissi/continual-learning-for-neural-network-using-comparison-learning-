import matplotlib.pyplot as plt
import numpy as np

dims = np.array([3, 32, 256, 512])

dims = np.array([3, 32, 64, 512])

# Execution times (V1 from your provided LaTeX table)
v1_time = np.array([0.0054, 2.3497, 4.9660, 465.5889])

# Reconstruction errors (V1 from your provided LaTeX table)
v1_error = np.array([3452.44, 6510.34, 76.49, 12.62])

# Execution times (V2 from your provided LaTeX table)
v2_time = np.array([0.0328, 6.5030, 12.7035, 625.0304])

# Reconstruction errors (V2 from your provided LaTeX table)
v2_error = np.array([6283.7, 726.40, 0.1, 0.1])

plt.figure(figsize=(12, 5))

# Plot execution times
plt.subplot(1, 2, 1)
plt.plot(dims, v1_time, 'o-', label='V1 Temps')
plt.plot(dims, v2_time, 's-', label='V2 Temps')
plt.xlabel("Dimension de l'image (n x n)")
plt.ylabel("Temps (secondes)")
plt.title("Temps d'exécution vs Dimension")
plt.legend()
plt.xscale('log')
plt.yscale('log')

# Plot reconstruction errors
plt.subplot(1, 2, 2)
plt.plot(dims, v1_error, 'o-', label='V1 Erreur')
plt.plot(dims, v2_error, 's-', label='V2 Erreur')
plt.xlabel("Dimension de l'image (n x n)")
plt.ylabel("Erreur moyenne")
plt.title("Erreur de reconstruction vs Dimension")
plt.legend()
plt.xscale('log')
plt.yscale('log')

plt.tight_layout()
plt.show()
