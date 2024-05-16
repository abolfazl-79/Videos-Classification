from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

#
from utils import FRAMES_NUM, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNEL, LABEL_NUM

def plotting_acc_loss(history):

  """
  Plot training and validation loss and accuracy from model training history.

  Args:
      history (tf.keras.callbacks.History): History object returned from model.fit().

  Returns:
      None
  """
  plt.figure(figsize=(12, 6))

  # Plot training and validation loss
  plt.subplot(1, 2, 1)
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Model Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend(['Train', 'Validation'], loc='upper right')

  # Plot training and validation accuracy
  plt.subplot(1, 2, 2)
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('Model Accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend(['Train', 'Validation'], loc='lower right')

  plt.tight_layout()
  plt.show()


def get_prediction_label(model, testing_frames):
    """
    Get predicted class labels for testing frames using a trained model.

    Args:
        model (tf.keras.Model): Trained model for making predictions.
        testing_frames (numpy.ndarray): Array of testing frames with shape (N, FRAMES_NUM, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNEL).

    Returns:
        numpy.ndarray: Predicted class labels for each testing sample.
    """
    # Use the model to make predictions on the testing frames
    prediction = model(testing_frames)
    
    # Find the index of the maximum value along axis=1 (which represents the predicted class)
    predicted_labels = np.argmax(prediction, axis=1)
    
    return predicted_labels


def get_confusion_matrix(model, testing_frames, testing_labels):
    """
    Generate the confusion matrix based on model predictions for testing frames.

    Args:
        model (tf.keras.Model): Trained model for making predictions.
        testing_frames (numpy.ndarray): Array of testing frames with shape (N, FRAMES_NUM, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNEL).
        testing_labels (numpy.ndarray): True labels corresponding to the testing frames.

    Returns:
        numpy.ndarray: Confusion matrix of shape (LABEL_NUM, LABEL_NUM).
            Rows represent predicted labels, columns represent true labels.
    """
    # Get the number of testing samples
    N = testing_frames.shape[0]
    
    # Initialize the confusion matrix with zeros
    confusion_matrix = np.zeros(shape=(LABEL_NUM, LABEL_NUM))
    
    # Get predicted labels using the model
    prediction_labels = get_prediction_label(model, testing_frames)
    
    # Populate the confusion matrix based on predicted and true labels
    for i in range(N):
        predicted_label = prediction_labels[i]
        true_label = testing_labels[i]
        confusion_matrix[predicted_label][true_label] += 1
    
    return confusion_matrix


def get_acc_test(model, testing_frames, testing_labels):
    """
    Calculate the accuracy of a model on testing data with precision up to 0.02.

    Args:
        model (tf.keras.Model): Trained model for making predictions.
        testing_frames (numpy.ndarray): Array of testing frames with shape (N, FRAMES_NUM, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNEL).
        testing_labels (numpy.ndarray): True labels corresponding to the testing frames.

    Returns:
        str: Accuracy message formatted as 'Accuracy on test data: {accuracy}'.
    """
    # Get the number of testing samples
    N = testing_frames.shape[0]
    
    # Get predicted labels using the model
    prediction_labels = get_prediction_label(model, testing_frames)
    
    # Initialize a counter for missed labels
    missed_labels = 0
    
    # Calculate the number of missed labels by comparing predicted and true labels
    for i in range(N):
        if prediction_labels[i] != testing_labels[i]:
            missed_labels += 1
    
    # Calculate accuracy as the percentage of correct predictions
    accuracy = 1.0 - (missed_labels / N)
    
    # Round the accuracy to two decimal places
    rounded_accuracy = round(accuracy, 2)
    
    # Format the accuracy message
    acc_message = 'Accuracy on test data: {0}'.format(rounded_accuracy)
    
    return acc_message



def plot_confusion_matrix(confusion_matrix, categories_names):
    """
    Plot the confusion matrix for a multi-class classification model.

    Args:
        confusion_matrix (numpy.ndarray): The confusion matrix computed from model predictions.
        categories_names (list): List of category names (labels) corresponding to the confusion matrix.

    Returns:
        None
    """
    # Create a new figure with a specific size for the plot
    plt.figure(figsize=(20, 10))
    
    # Create a ConfusionMatrixDisplay object with the provided confusion matrix and category labels
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=categories_names)
    
    # Plot the confusion matrix using the ConfusionMatrixDisplay object
    cm_display.plot()
    
    # Display the plot
    plt.show()
