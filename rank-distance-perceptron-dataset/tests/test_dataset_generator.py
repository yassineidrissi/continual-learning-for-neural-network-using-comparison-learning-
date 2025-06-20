import random
import math

def generate_dataset(num_samples=1000, num_features=32):
    dataset = []
    labels = []
    
    for _ in range(num_samples):
        features = [random.uniform(0, 1) for _ in range(num_features)]
        label = 1 if sum(features) > num_features / 2 else 0
        dataset.append(features)
        labels.append(label)
    
    return dataset, labels

def generate_noisy_dataset(num_samples=1000, num_features=32, noise_level=0.1):
    dataset, labels = generate_dataset(num_samples, num_features)
    
    for i in range(num_samples):
        if random.random() < noise_level:
            noise_index = random.randint(0, num_features - 1)
            dataset[i][noise_index] = random.uniform(0, 1)
    
    return dataset, labels

def main():
    dataset, labels = generate_dataset()
    noisy_dataset, noisy_labels = generate_noisy_dataset()
    
    print("Generated dataset:")
    print(dataset[:5])
    print("Generated labels:")
    print(labels[:5])
    print("Generated noisy dataset:")
    print(noisy_dataset[:5])
    print("Generated noisy labels:")
    print(noisy_labels[:5])

if __name__ == "__main__":
    main()