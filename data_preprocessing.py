
import os
import numpy as np
import random

# Set up constants for video frames
# FRAMES_NUM: Number of frames in the video
# FRAME_HEIGHT: Height of each frame in pixels
# FRAME_WIDTH: Width of each frame in pixels
# FRAME_CHANNEL: Number of color channels in each frame (e.g., 3 for RGB)
# These constants are imported from utils.py
from utils import FRAMES_NUM, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNEL
from utils import get_frames


def splitting_dataset(classes_videos_path, ratio):
    """
    Split the video dataset into training and testing sets based on a given ratio.

    Args:
        classes_videos_path (list): A list of lists where each sublist contains paths
                                    to videos belonging to the corresponding class.
        ratio (float): The ratio of videos to be assigned to the training set.

    Returns:
        tuple: A tuple containing two lists:
            - training_videos: List of (video_path, class_index) tuples for training.
            - testing_videos: List of (video_path, class_index) tuples for testing.
    """
    # Initialize empty lists to store training and testing videos
    training_videos = []
    testing_videos = []
    
    # Iterate over each class
    for i in range(len(classes_videos_path)):
        class_video_path = classes_videos_path[i]
        # Iterate over each video path for the ith class
        for j in range(len(class_video_path)):
            # Randomly determine whether to assign the video to training or testing set
            p = np.random.uniform(0.0, 1.0)
            if p <= ratio:
                # Append the video to the training set with its class index
                training_videos.append((class_video_path[j], i))
            else:
                # Append the video to the testing set with its class index
                testing_videos.append((class_video_path[j], i))
    
    # Return a tuple containing the lists of training and testing videos
    return (training_videos, testing_videos)






def load_data(filepaths):
    """
    Load frames and labels of videos based on provided filepaths.

    Args:
        filepaths (list): A list of (filepath, class_index) tuples representing videos.

    Returns:
        tuple: A tuple containing:
            - frames (numpy.ndarray): Array of video frames with shape (num_videos, FRAMES_NUM, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNEL).
            - labels (numpy.ndarray): Array of labels corresponding to the videos with shape (num_videos,).
    """
    # Initialize empty numpy arrays to store frames and labels
    frames = np.empty(shape=(len(filepaths), FRAMES_NUM, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNEL), dtype=np.uint8)
    labels = np.empty(shape=(len(filepaths),), dtype=np.uint8)
    
    # Iterate over each video in the list of filepaths
    for i in range(len(filepaths)):
        # Extract the filepath and class index for the current video
        video_filepath = filepaths[i][0]
        class_index = filepaths[i][1]
        
        # Load frames for the current video using the get_frames function
        frames[i] = get_frames(video_filepath)
        
        # Assign the label (class index) to the corresponding position in the labels array
        labels[i] = class_index
    
    # Return the frames and labels arrays as a tuple
    return frames, labels



    

def get_dataset_filepaths(classes_videos_path):
    """
    Split the provided video paths into training and testing sets and then shuffle them.

    Args:
        classes_videos_path (numpy.ndarray): An array containing video paths grouped by classes.

    Returns:
        tuple: A tuple containing two numpy arrays:
            - training_filepaths: Shuffled array of video paths for training.
            - testing_filepaths: Shuffled array of video paths for testing.
    """
    # Split video paths into training and testing filepaths using a specified ratio
    training_filepaths, testing_filepaths = splitting_dataset(classes_videos_path, 0.8)

    # Shuffle the training and testing filepaths
    random.shuffle(training_filepaths)
    random.shuffle(testing_filepaths)

    # Convert shuffled lists to numpy arrays
    training_filepaths = np.array(training_filepaths)
    testing_filepaths = np.array(testing_filepaths)

    # Return the shuffled training and testing filepaths as a tuple
    return (training_filepaths, testing_filepaths)


    
def get_ten_top_categories(ith_ten_categories, DATASET_PATH):
    """
    Retrieve paths of the top 10 categories from the dataset starting at a specified index.

    Args:
        ith_ten_categories (int): Index of the set of 10 categories to retrieve.
        DATASET_PATH (str): Path to the dataset directory containing category subdirectories.

    Returns:
        list: A list containing numpy arrays of video paths for each category in the selected set of 10 categories.
    """
    # Calculate the start and end indices for retrieving the top 10 categories
    start = ith_ten_categories * 10
    end = (ith_ten_categories + 1) * 10
        
    # Get names of the top 10 categories from the dataset directory
    ten_top_categories = os.listdir(DATASET_PATH)[start:end]
    
    # Initialize a list to store the paths of videos for each category in the top 10
    ten_top_categories_path = []

    # Iterate over each category in the top 10 categories
    for category in ten_top_categories:
        # Get list of video names in the current category directory
        category_videos_path = os.listdir(os.path.join(DATASET_PATH, category))
        # Initialize a list to store full paths of videos in the current category
        single_category_videos_path = []
        # Construct full paths for each video in the current category
        for video in category_videos_path:
            single_category_videos_path.append(os.path.join(DATASET_PATH, category, video))
          
        # Append numpy array of video paths for the current category to the result list
        ten_top_categories_path.append(np.array(single_category_videos_path))
        
    return ten_top_categories_path

                
            
        
        