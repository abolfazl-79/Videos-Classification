import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
import tensorflow as tf

FRAMES_NUM = 20
FRAME_HEIGHT = 100
FRAME_WIDTH = 100
FRAME_CHANNEL = 3
LABEL_NUM = 10



def get_frames(video_file):
    """
    Extract and resize random frames from a video file.

    Args:
        video_file (str): Path to the video file.

    Returns:
        numpy array: Array containing resized frames of shape (FRAMES_NUM, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNEL).
    """

    # Open the video file for reading
    video = cv2.VideoCapture(video_file)

    # Get the total number of frames in the video
    frame_ids = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize an array to store frames (shape: FRAMES_NUM x FRAME_HEIGHT x FRAME_WIDTH x FRAME_CHANNEL)
    frames = np.zeros(shape=(FRAMES_NUM, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNEL), dtype=np.uint8)

    # Select 20 random frame IDs from the video
    random_frame_ids = random.sample(range(frame_ids - 2), 20)
    
    # Sort the frame IDs in ascending order
    random_frame_ids.sort()

    # Initialize a frame counter
    frame_counter = 0

    # Iterate over each random frame ID selected
    for frame_id in random_frame_ids:
        # Set the current frame position to the selected frame ID
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        
        # Read the frame
        _, frame = video.read()

        # Check if the frame is valid (not None)
        if frame is not None:
            # Resize the frame to the specified dimensions (FRAME_WIDTH x FRAME_HEIGHT)
            resized_frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            # Store the resized frame in the frames array
            frames[frame_counter] = resized_frame

            # Increment the frame counter
            frame_counter += 1

    # Release the video capture object
    video.release()

    # Convert frames array to uint8 data type and return
    return frames.astype(np.uint8)



def build_model():

  """
    Build and return a ConvLSTM-based model for video classification.

    The model architecture consists of:
    - Four ConvLSTM2D layers with decreasing number of filters and recurrent dropout.
    - MaxPooling3D layers to reduce spatial dimensions.
    - TimeDistributed Dropout layers for regularization.
    - Flatten layer to prepare for classification.
    - Dense layer with softmax activation for classification.

    Returns:
        tf.keras.Sequential: Compiled ConvLSTM-based model.
    """
  
  model = tf.keras.Sequential([

    ## First ConvLSTM layer with 4 filters ##
    tf.keras.layers.ConvLSTM2D(4, kernel_size=(3, 3), activation='tanh', data_format='channels_last', recurrent_dropout=.2, input_shape=(20, 100, 100, 3), return_sequences=True),
    tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(.5)),

    ## Second ConvLSTM layer with 8 filters ##
    tf.keras.layers.ConvLSTM2D(8, kernel_size=(3, 3), activation='tanh', data_format='channels_last', recurrent_dropout=.2, return_sequences=True),
    tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(.5)),

    ## Third ConvLSTM layer with 14 filters ##
    tf.keras.layers.ConvLSTM2D(14, kernel_size=(3, 3), activation='tanh', data_format='channels_last', recurrent_dropout=.2, return_sequences=True),
    tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(.5)),

    ## Fourth ConvLSTM layer with 16 filters ##
    tf.keras.layers.ConvLSTM2D(16, kernel_size=(3, 3), activation='tanh', data_format='channels_last', recurrent_dropout=.2, return_sequences=True),
    tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2), padding='same', data_format='channels_last'),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(.5)),

    ## Flatten layer to prepare for classification ##
    tf.keras.layers.Flatten(),

    ## Dense layer with softmax activation for classification (10 output classes) ##
    tf.keras.layers.Dense(10, activation='softmax')
  ])
  return model




def get_categories_name(ith_ten_categories, categories_path):
    """
    Extract category names from the given list of video paths for a specific set of 10 categories.

    Args:
        ith_ten_categories (int): Index of the set of 10 categories to retrieve.
        categories_path (list): List of video paths grouped by categories.

    Returns:
        list: List of category names corresponding to the specified set of 10 categories.
    """
    # Calculate the start and end indices for retrieving category names
    start = ith_ten_categories * 10
    end = (ith_ten_categories + 1) * 10
    
    # Get the names of the categories from the dataset directory
    category_names = os.listdir(categories_path)
    
    # Select the category names corresponding to the specified set of 10 categories
    selected_category_names = category_names[start:end]

    return selected_category_names







