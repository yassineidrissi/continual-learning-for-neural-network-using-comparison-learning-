import random
import math

def generate_data(num_samples, num_features):
    dataset = []
    labels = []
    
    for _ in range(num_samples):
        features = [random.uniform(0, 1) for _ in range(num_features)]
        label = 1 if sum(features) > num_features / 2 else 0
        dataset.append(features)
        labels.append(label)
    
    return dataset, labels

def generate_synthetic_dataset(num_samples=1000, num_features=32):
    if num_features < 32:
        raise ValueError("Number of features must be at least 32.")
    
    dataset, labels = generate_data(num_samples, num_features)
    return dataset, labels

if __name__ == "__main__":
    data, labels = generate_synthetic_dataset()
    print(f"Generated dataset with {len(data)} samples and {len(data[0])} features.")