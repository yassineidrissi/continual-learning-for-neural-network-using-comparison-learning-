#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import time

# Dummy experimental functions for demonstration.
# Replace these with your actual run_experiment_v1 and run_experiment_v2 functions.
def run_experiment_v1(num_link, image_size=(256,256), iter_max=5):
    # Simulated behavior: error decreases with num_link, time increases modestly.
    error = max(10000 / num_link - 100, 0)
    time_taken = 0.2 * (num_link / 8)
    return error, time_taken

def run_experiment_v2(num_link, image_size=(256,256), iter_max=5):
    # Simulated behavior: error decreases more sharply for V2.
    error = max(8000 / num_link - 200, 0)
    time_taken = 0.3 * (num_link / 8)
    return error, time_taken

# List of num_link values to test.
num_link_values = [8, 16, 32, 64, 128, 256]
v1_errors = []
v1_times = []
v2_errors = []
v2_times = []

for nl in num_link_values:
    err1, t1 = run_experiment_v1(nl)
    err2, t2 = run_experiment_v2(nl)
    v1_errors.append(err1)
    v1_times.append(t1)
    v2_errors.append(err2)
    v2_times.append(t2)

# Plot Error vs. num_link
plt.figure()
plt.plot(num_link_values, v1_errors, 'o-', label='V1 Erreur')
plt.plot(num_link_values, v2_errors, 's-', label='V2 Erreur')
plt.xlabel('num_link')
plt.ylabel('Erreur de reconstruction')
plt.title('Erreur vs num_link')
plt.legend()
plt.grid(True)
plt.savefig('graph_num_link_error.png')

# Plot Time vs. num_link
plt.figure()
plt.plot(num_link_values, v1_times, 'o-', label='V1 Temps (s)')
plt.plot(num_link_values, v2_times, 's-', label='V2 Temps (s)')
plt.xlabel('num_link')
plt.ylabel('Temps (s)')
plt.title("Temps d'exécution vs num_link")
plt.legend()
plt.grid(True)
plt.savefig('graph_num_link_time.png')
