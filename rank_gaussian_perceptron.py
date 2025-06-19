#!/usr/bin/env python3
"""
Learned Rank-Distance Perceptron
by AI Researcher • 2025

Architecture:
Input [x1, x2, ..., xn] → Rank Layer → Distance-Gaussian Activation → Perceptron → Classes

Each neuron learns preferred rank patterns [ra1, ra2, ..., ran]
Activation based on distance between input ranks and preferred ranks
"""

import random
import math

def argsort_ranks(x):
    """
    Compute ranks of input x using argsort.
    Returns [r1, r2, ..., rn] where ri is the rank of xi
    Rank 0 = highest value, rank 1 = second highest, etc.
    """
    # Get indices sorted by descending values
    sorted_indices = sorted(range(len(x)), key=lambda i: -x[i])
    
    # Create rank array
    ranks = [0] * len(x)
    for rank, idx in enumerate(sorted_indices):
        ranks[idx] = rank
    
    return ranks

def rank_distance(r_input, r_preferred):
    """
    Compute L2 distance between rank vectors:
    d(r, ra) = sum(|ri - rai|^2)
    """
    return sum((ri - rai)**2 for ri, rai in zip(r_input, r_preferred))

def gaussian_activation(distance, sigma=1.0):
    """
    Gaussian activation: a = exp(-d / (2*sigma^2))
    """
    return math.exp(-distance / (2 * sigma**2))

def rational_activation(distance):
    """
    Alternative activation: a = 1 / (1 + d)
    """
    return 1.0 / (1.0 + distance)

class RankLayer:
    """
    Rank processing layer that learns preferred rank patterns
    """
    def __init__(self, input_size, num_neurons, activation_type='gaussian', sigma=1.0):
        self.input_size = input_size
        self.num_neurons = num_neurons
        self.activation_type = activation_type
        self.sigma = sigma
        
        # Each neuron has preferred ranks [ra1, ra2, ..., ran]
        # Initialize with random rank preferences
        self.preferred_ranks = []
        for _ in range(num_neurons):
            # Random permutation of ranks 0 to input_size-1
            ranks = list(range(input_size))
            random.shuffle(ranks)
            self.preferred_ranks.append(ranks)
    
    def forward(self, x):
        """
        Process input through rank layer:
        1. Compute input ranks: [r1, ..., rn] = argsort(x)
        2. For each neuron, compute distance to preferred ranks
        3. Apply activation function
        """
        # Step 1: Compute input ranks
        input_ranks = argsort_ranks(x)
        
        # Step 2-3: For each neuron, compute activation
        activations = []
        for i in range(self.num_neurons):
            # Distance between input ranks and neuron's preferred ranks
            distance = rank_distance(input_ranks, self.preferred_ranks[i])
            
            # Apply activation function
            if self.activation_type == 'gaussian':
                activation = gaussian_activation(distance, self.sigma)
            else:  # rational
                activation = rational_activation(distance)
            
            activations.append(activation)
        
        return activations, input_ranks
    
    def update_preferred_ranks(self, neuron_idx, new_ranks, lr=0.1):
        """
        Update preferred ranks for a specific neuron
        (simplified update - in practice might use gradient-based methods)
        """
        # Simple weighted update towards new ranks
        for i in range(self.input_size):
            current = self.preferred_ranks[neuron_idx][i]
            target = new_ranks[i]
            self.preferred_ranks[neuron_idx][i] = current + lr * (target - current)

class RankDistancePerceptron:
    """
    Complete network: Rank Layer → Perceptron
    """
    def __init__(self, input_size, num_classes, rank_neurons=None, activation_type='gaussian', sigma=1.0, lr=0.1):
        self.input_size = input_size
        self.num_classes = num_classes
        self.lr = lr
        
        # Default: one rank neuron per input dimension
        if rank_neurons is None:
            rank_neurons = input_size
        
        # Rank processing layer
        self.rank_layer = RankLayer(input_size, rank_neurons, activation_type, sigma)
        
        # Perceptron weights: from rank activations to classes
        self.weights = [
            [random.uniform(-1, 1) for _ in range(rank_neurons)]
            for _ in range(num_classes)
        ]
        self.bias = [random.uniform(-1, 1) for _ in range(num_classes)]
    
    def activation(self, x):
        """Step activation function"""
        return 1 if x >= 0 else 0
    
    def forward(self, x):
        """
        Forward pass:
        x → rank_layer → perceptron → classes
        """
        # Get rank activations
        rank_activations, input_ranks = self.rank_layer.forward(x)
        
        # Perceptron computation
        outputs = []
        for j in range(self.num_classes):
            net = sum(self.weights[j][i] * rank_activations[i] 
                     for i in range(len(rank_activations)))
            net += self.bias[j]
            outputs.append(net)
        
        return outputs, rank_activations, input_ranks
    
    def predict(self, x):
        """Binary predictions for each class"""
        outputs, _, _ = self.forward(x)
        return [self.activation(out) for out in outputs]
    
    def predict_class(self, x):
        """Single class prediction (highest output)"""
        outputs, _, _ = self.forward(x)
        return outputs.index(max(outputs))
    
    def train(self, X, Y, epochs=50):
        """
        Train the network
        X: list of input vectors
        Y: list of class labels (integers)
        """
        for epoch in range(1, epochs + 1):
            total_error = 0
            correct = 0
            
            for x, y_true in zip(X, Y):
                outputs, rank_activations, input_ranks = self.forward(x)
                predictions = [self.activation(out) for out in outputs]
                
                # One-hot target
                target = [1 if j == y_true else 0 for j in range(self.num_classes)]
                
                # Compute errors
                errors = [predictions[j] - target[j] for j in range(self.num_classes)]
                
                # Update perceptron weights
                if any(e != 0 for e in errors):
                    total_error += 1
                    for j in range(self.num_classes):
                        for i in range(len(rank_activations)):
                            self.weights[j][i] -= self.lr * errors[j] * rank_activations[i]
                        self.bias[j] -= self.lr * errors[j]
                
                # Check accuracy
                if self.predict_class(x) == y_true:
                    correct += 1
            
            accuracy = correct / len(X) * 100
            print(f"Epoch {epoch}/{epochs} — Errors: {total_error}, Accuracy: {accuracy:.1f}%")
            
            if total_error == 0:
                print("Perfect classification achieved!")
                break
    
    def show_learned_ranks(self):
        """Display learned rank preferences"""
        print("\nLearned Rank Preferences:")
        for i, ranks in enumerate(self.rank_layer.preferred_ranks):
            print(f"  Rank Neuron {i}: {[f'{r:.1f}' for r in ranks]}")

def create_synthetic_data():
    """Create test data with different rank patterns"""
    X = [
        # Class 0: high values at beginning
        [0.9, 0.8, 0.3, 0.2, 0.1, 0.0],
        [0.85, 0.75, 0.35, 0.25, 0.15, 0.05],
        [0.95, 0.85, 0.25, 0.15, 0.05, 0.02],
        
        # Class 1: high values in middle
        [0.2, 0.1, 0.9, 0.8, 0.3, 0.0],
        [0.25, 0.15, 0.85, 0.75, 0.35, 0.05],
        [0.15, 0.05, 0.95, 0.85, 0.25, 0.02],
        
        # Class 2: high values at end
        [0.1, 0.0, 0.2, 0.3, 0.8, 0.9],
        [0.15, 0.05, 0.25, 0.35, 0.75, 0.85],
        [0.05, 0.02, 0.15, 0.25, 0.85, 0.95],
    ]
    Y = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    return X, Y

if __name__ == "__main__":
    print("=== Learned Rank-Distance Perceptron ===\n")
    
    # Create synthetic dataset
    X_train, Y_train = create_synthetic_data()
    
    # Create and train network
    model = RankDistancePerceptron(
        input_size=6,
        num_classes=3,
        rank_neurons=8,  # More rank neurons than inputs
        activation_type='gaussian',
        sigma=2.0,
        lr=0.2
    )
    
    print("Training...")
    model.train(X_train, Y_train, epochs=30)
    
    # Show learned rank preferences
    model.show_learned_ranks()
    
    # Test predictions
    print("\nTest Predictions:")
    for i, (x, y_true) in enumerate(zip(X_train, Y_train)):
        pred_class = model.predict_class(x)
        ranks = argsort_ranks(x)
        print(f"Sample {i}: True={y_true}, Pred={pred_class}, Ranks={ranks}")
    
    print("\nTesting new samples:")
    # Test with new data
    X_test = [
        [0.88, 0.77, 0.32, 0.21, 0.11, 0.03],  # Should be class 0
        [0.22, 0.12, 0.88, 0.78, 0.32, 0.04],  # Should be class 1
        [0.12, 0.03, 0.22, 0.32, 0.78, 0.88],  # Should be class 2
    ]
    
    for i, x in enumerate(X_test):
        pred_class = model.predict_class(x)
        ranks = argsort_ranks(x)
        print(f"Test {i}: Pred={pred_class}, Ranks={ranks}")