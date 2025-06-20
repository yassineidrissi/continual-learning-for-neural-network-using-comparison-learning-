## Dataset Structure

- **Features**: The dataset includes 32 features, each representing a different dimension of the data. The features are generated using random values and may follow specific distributions based on the generation logic.
  
- **Labels**: Each data point is associated with a label that indicates its class or category. The labels are generated based on the relationships defined in the dataset generation logic.

## Usage

To use this dataset, you can load it into your preferred data analysis or machine learning framework. The dataset is structured in a way that allows for easy integration with various tools and libraries.

## Generation

The dataset is generated using the `src/dataset_generator.py` script, which utilizes the `random` and `math` libraries to create synthetic data points. You can modify the parameters in the script to generate different variations of the dataset.

## Visualization

For visualizing the dataset, you can refer to the functions provided in `src/visualization/plot_utils.py`. These functions allow you to explore the feature distributions and relationships between different features.

## Validation

To ensure the quality of the dataset, validation functions are provided in `src/utils/validation.py`. These functions check that the dataset meets the required specifications for size and feature distribution.

## Conclusion

This dataset serves as a valuable resource for testing and evaluating the Rank-Distance Perceptron. Feel free to explore and modify the dataset generation process to suit your needs.