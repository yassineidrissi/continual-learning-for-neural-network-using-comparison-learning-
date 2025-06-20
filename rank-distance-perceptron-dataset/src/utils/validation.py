import random
import math

def validate_dataset(dataset):
    if not dataset:
        return False, "Dataset is empty."

    num_features = len(dataset[0])
    if num_features < 32:
        return False, "Dataset must have at least 32 features."

    for data_point in dataset:
        if len(data_point) != num_features:
            return False, "All data points must have the same number of features."

    return True, "Dataset is valid."

def check_feature_distribution(dataset):
    num_features = len(dataset[0])
    feature_means = [0] * num_features
    feature_stds = [0] * num_features

    for data_point in dataset:
        for i in range(num_features):
            feature_means[i] += data_point[i]

    feature_means = [mean / len(dataset) for mean in feature_means]

    for data_point in dataset:
        for i in range(num_features):
            feature_stds[i] += (data_point[i] - feature_means[i]) ** 2

    feature_stds = [math.sqrt(std / len(dataset)) for std in feature_stds]

    return feature_means, feature_stds

def validate_feature_distribution(dataset, expected_means, expected_stds, tolerance=0.1):
    actual_means, actual_stds = check_feature_distribution(dataset)

    for i in range(len(expected_means)):
        if abs(actual_means[i] - expected_means[i]) > tolerance:
            return False, f"Mean of feature {i} is out of tolerance."
        if abs(actual_stds[i] - expected_stds[i]) > tolerance:
            return False, f"Standard deviation of feature {i} is out of tolerance."

    return True, "Feature distribution is valid."