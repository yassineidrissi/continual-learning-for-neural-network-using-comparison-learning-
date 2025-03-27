#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import time

def run_experiment_v2(iter_max, image_size=(256,256), num_link=32):
    # Simulated behavior for V2: error decreases for lower iter_max then saturates/increases.
    if iter_max <= 10:
                error = 30 + (iter_max - 20) * 5
    elif iter_max <= 20:
        error = 10 + (iter_max - 10) * 2
    else:
                error = 100 / iter_max
    time_taken = 0.1 * iter_max  # time increases linearly with iter_max
    return error, time_taken

# List of iter_max values to test.
iter_max_values = [5, 10, 20, 30]
errors = [159.00, 149.72, 143.27, 140.42]  # updated values
times = [80.57, 127.17, 134.06, 140.67]  #

# for im in iter_max_values:
#     err, t = run_experiment_v2(im)
    # errors.append(err)
    # times.append(t)

# Plot Error vs. iter_max
plt.figure()
plt.plot(iter_max_values, errors, 'o-', label='V2 Erreur')
plt.xlabel('iter_max')
plt.ylabel('Erreur de reconstruction')
plt.title("Erreur vs Nombre d'itérations (V2)")
plt.legend()
plt.grid(True)
plt.savefig('graph_iter_max_error.png')

# Plot Time vs. iter_max
plt.figure()
plt.plot(iter_max_values, times, 'o-', label='V2 Temps (s)')
plt.xlabel('iter_max')
plt.ylabel('Temps (s)')
plt.title("Temps d'exécution vs Nombre d'itérations (V2)")
plt.legend()
plt.grid(True)
plt.savefig('graph_iter_max_time.png')
