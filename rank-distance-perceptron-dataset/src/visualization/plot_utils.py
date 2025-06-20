import random
import math
import matplotlib.pyplot as plt
import numpy as np

def plot_feature_distribution(data, feature_index):
    feature_data = [point[feature_index] for point in data]
    plt.hist(feature_data, bins=30, alpha=0.7, color='blue')
    plt.title(f'Feature {feature_index} Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def plot_feature_relationship(data, feature_index1, feature_index2):
    feature_data1 = [point[feature_index1] for point in data]
    feature_data2 = [point[feature_index2] for point in data]
    plt.scatter(feature_data1, feature_data2, alpha=0.5, color='red')
    plt.title(f'Relationship between Feature {feature_index1} and Feature {feature_index2}')
    plt.xlabel(f'Feature {feature_index1}')
    plt.ylabel(f'Feature {feature_index2}')
    plt.grid(True)
    plt.show()

def plot_pairwise_relationships(data, num_features):
    fig, axs = plt.subplots(num_features, num_features, figsize=(15, 15))
    for i in range(num_features):
        for j in range(num_features):
            if i == j:
                axs[i, j].hist([point[i] for point in data], bins=30, alpha=0.7, color='blue')
                axs[i, j].set_title(f'Feature {i} Distribution')
            else:
                axs[i, j].scatter([point[i] for point in data], [point[j] for point in data], alpha=0.5, color='red')
                axs[i, j].set_title(f'Feature {i} vs Feature {j}')
            axs[i, j].grid(True)
    plt.tight_layout()
    plt.show()