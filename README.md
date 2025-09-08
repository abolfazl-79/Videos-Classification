# Video Classification using ConvLSTM on UCF50 Dataset

This repository contains an implementation of a video classification project using a ConvLSTM-based neural network. The goal is to classify videos into 10 selected categories out of the 50 available in the UCF50 action recognition dataset. The project processes videos by extracting frames, training a model on batches of categories, and evaluating performance with metrics like accuracy and confusion matrices.

## Project Structure

- **Video_classification_main.ipynb**: The main Jupyter notebook that orchestrates the workflow. It loads data, trains the ConvLSTM model in a loop over batches of 10 categories, and evaluates results.
- **data_preprocessing.py**: Handles dataset splitting (train/test), loading video frames, and retrieving video paths for specified categories.
- **utils.py**: Includes functions for frame extraction from videos, building the ConvLSTM model, and managing category names.
- **evaluation.py**: Provides tools for plotting training history, generating predictions, computing confusion matrices, and calculating test accuracy.

## Features

- Processes videos from the UCF50 dataset (action recognition videos).
- Extracts frames from videos for sequence-based input to the model.
- Uses a ConvLSTM architecture for spatiotemporal feature learning.
- Supports training on multiple batches of 10 categories (configurable via `START_INDEX` and `END_INDEX`).
- Visualizes training curves and confusion matrices.
- Computes and prints test accuracy.

## Requirements

- Python 3.8+
- TensorFlow/Keras (for model building and training)
- OpenCV (cv2) for video frame extraction
- NumPy for data manipulation
- Matplotlib and Seaborn for plotting
- Scikit-learn for metrics like confusion matrices

Install dependencies using:
- pip install tensorflow opencv-python numpy matplotlib seaborn scikit-learn


## Dataset

This project uses the [UCF50 Action Recognition Dataset](https://www.crcv.ucf.edu/data/UCF50.php), which contains 50 action categories with realistic videos from YouTube.

- Download the dataset and extract it to a directory (e.g., `/content/UCF50` in the notebook).
- The code assumes the dataset is structured with category folders containing `.avi` video files.
- Note: The dataset is large (~6GB), so ensure sufficient storage and compute resources (e.g., Google Colab with GPU).

## Usage

1. **Clone the repository**:
  -git clone https://github.com/your-username/video-classification-ucf50.git
  -cd video-classification-ucf50

2. **Set up the environment**:
- Install dependencies as listed above.
- Download and place the UCF50 dataset in the appropriate path (update `DATASET_PATH` in the code if needed).

3. **Run the notebook**:
- Open `Video_classification_main.ipynb` in Jupyter Notebook or Google Colab.
- Upload or place the supporting scripts (`data_preprocessing.py`, `utils.py`, `evaluation.py`) in the working directory.
- Configure parameters like `START_INDEX`, `END_INDEX`, and dataset path.
- Execute the cells to preprocess data, train the model, and evaluate.

Example workflow in the notebook:
- Retrieves paths for 10 categories.
- Splits into train/test sets.
- Loads frames and labels.
- Builds and trains the ConvLSTM model (10 epochs, batch size 20).
- Plots accuracy/loss and confusion matrix.
- Computes test accuracy.

Sample output includes printed accuracy (e.g., "Accuracy on test data: 0.71") and visualizations.

## Model Details

- **Architecture**: ConvLSTM2D layers for handling video sequences, followed by dense layers for classification.
- **Input**: Sequences of video frames (resized and normalized).
- **Output**: Class probabilities for 10 categories.
- **Optimizer**: Adam
- **Loss**: Sparse Categorical Cross-Entropy
- **Metrics**: Accuracy
- **Training**: Includes validation split (20%) and early stopping potential (not implemented by default).

## Results

- The model is trained on batches of 10 categories at a time.
- Example test accuracy: ~71% (varies by category batch and hyperparameters).
- Confusion matrices and training plots are generated for each batch.

Feel free to experiment with hyperparameters like epochs, batch size, or frame extraction rate for better performance.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for improvements, bug fixes, or additional features.


## Acknowledgments

- Inspired by action recognition research and the UCF50 dataset.
- Built using TensorFlow/Keras for deep learning.

If you encounter issues or have questions, open an issue on GitHub!
