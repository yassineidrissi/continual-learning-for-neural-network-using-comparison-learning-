# Rank-Distance Perceptron Dataset

This project is designed to generate and validate a synthetic dataset for testing the Rank-Distance Perceptron algorithm. The dataset consists of multiple features and is structured to facilitate various experiments and analyses.

## Project Structure

- **src/**: Contains the main source code for dataset generation, validation, and visualization.
  - **dataset_generator.py**: Implements the logic for generating a synthetic dataset with 32 or more features.
  - **utils/**: Contains utility functions for validation.
    - **validation.py**: Provides functions to validate the generated dataset.
  - **visualization/**: Contains functions for visualizing the dataset.
    - **plot_utils.py**: Implements plotting functions for feature distributions and relationships.

- **tests/**: Contains unit tests for the project.
  - **test_dataset_generator.py**: Tests for the dataset generation functions.
  - **test_validation.py**: Tests for the validation functions.

- **data/**: Contains information about the dataset.
  - **README.md**: Documentation about the dataset structure and usage.

- **notebooks/**: Contains Jupyter notebooks for dataset exploration.
  - **dataset_exploration.ipynb**: A notebook for visualizing and analyzing the dataset.

- **requirements.txt**: Lists the dependencies required for the project.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd rank-distance-perceptron-dataset
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To generate the dataset, run the `dataset_generator.py` script. You can customize the parameters to control the number of samples and features.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.