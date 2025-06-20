import random
import math

def test_validate_dataset_size():
    dataset = generate_synthetic_dataset(num_samples=1000, num_features=32)
    assert len(dataset) == 1000, "Dataset size should be 1000"

def test_validate_feature_count():
    dataset = generate_synthetic_dataset(num_samples=1000, num_features=32)
    assert len(dataset[0]) == 32, "Each data point should have 32 features"

def test_validate_feature_distribution():
    dataset = generate_synthetic_dataset(num_samples=1000, num_features=32)
    for feature_index in range(32):
        feature_values = [data_point[feature_index] for data_point in dataset]
        assert all(isinstance(value, (int, float)) for value in feature_values), f"Feature {feature_index} should contain numeric values"

def generate_synthetic_dataset(num_samples, num_features):
    dataset = []
    for _ in range(num_samples):
        data_point = [random.uniform(0, 1) for _ in range(num_features)]
        dataset.append(data_point)
    return dataset