import numpy as np
import matplotlib.pyplot as plt
import idx2numpy
import gzip
import shutil
import os
import time

# Print current working directory
print("Current Directory:", os.getcwd())

# List all files in the directory
print("Files in the current directory:", os.listdir())


files = [
    "train-images-idx3-ubyte (1).gz",
    "train-labels-idx1-ubyte (1).gz",
    "t10k-images-idx3-ubyte (2).gz",
    "t10k-labels-idx1-ubyte (3).gz",
]

for file in files:
    if os.path.exists(file):  # Check if the file exists before extracting
        with gzip.open(file, "rb") as f_in:
            with open(file.replace(".gz", ""), "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"Extracted: {file}")
    else:
        print(f"File not found: {file}")


# Define correct file paths
train_images_path = "train-images-idx3-ubyte (1)"
train_labels_path = "train-labels-idx1-ubyte (1)"
test_images_path = "t10k-images-idx3-ubyte (2)"
test_labels_path = "t10k-labels-idx1-ubyte (3)"

# Ensure the files exist before loading
for file in [train_images_path, train_labels_path, test_images_path, test_labels_path]:
    if not os.path.exists(file):
        print(f"Error: File not found - {file}")
        exit()

# Load data
train_images = idx2numpy.convert_from_file(train_images_path)
train_labels = idx2numpy.convert_from_file(train_labels_path)
test_images = idx2numpy.convert_from_file(test_images_path)
test_labels = idx2numpy.convert_from_file(test_labels_path)

print("Training images shape:", train_images.shape)  # (60000, 28, 28)
print("Training labels shape:", train_labels.shape)  # (60000,)
print("Testing images shape:", test_images.shape)  # (10000, 28, 28)
print("Testing labels shape:", test_labels.shape)  # (10000,)

# Function to display a sample digit
def show_digit(img, label):
    plt.imshow(img, cmap="gray")
    plt.title(f"Label: {label}")
    plt.axis("off")
    plt.show()

show_digit(train_images[7], train_labels[7])

#Function that computes squared Euclidean distance between two vectors.
def squared_dist(x,y):
    return np.sum(np.square(x-y))
# Compute distance between a seven and a one in our training set.
print("Distance from 7 to 1: ",squared_dist(train_images[7],train_images[1]))
# Compute distance between a seven and a two in our training set.
print("Distance from 7 to 2: ",squared_dist(train_images[7],train_images[2]))
# Compute distance between two seven's in our training set.
print("Distance from 7 to 7: ",squared_dist(train_images[7],train_images[7]))

#4. Computing nearest neighbors
# Now that we have a distance function defined,we can now turn to nearest neighbor classification.
# Function that takes a vector x and returns the index of its nearest neighbor in train_data.
def find_NN(x):
# Compute distances from x to every row in train_data
    distances = [squared_dist(x,train_images[i,]) for i in range(len(train_images))]
# Get the index of the smallest distance
    return np.argmin(distances)
# Function that takes a vector x and returns the class of its nearest neighbor in train_data.
def NN_classifier(x):
 # Get the index of the the nearest neighbor
    index = find_NN(x)
 # Return its class
    return train_images[index]
#A success case:
print("A success case:")
print("NN classification: ", NN_classifier(test_images[0,]))
print("True label: ", test_labels[0])
print("The test image:")
show_digit(test_images[0], test_labels[0])
print("The Corresponding Nearest Neighbor Image:")
nearest_index = find_NN(test_images[0])
show_digit(train_images[nearest_index],test_labels[0])  # Display the nearest training image

#A failure case:
print("A failure case:")
print("NN classification: ", NN_classifier(test_images[39]))
print("True label: ", test_labels[39])
print("The test image:")
show_digit(test_images[39], test_labels[39])
print("The corresponding nearest neighbor image:")
nearest_index = find_NN(test_images[39])
show_digit(train_images[nearest_index], test_labels[39])

# 5. Processing the full test set using NN Classifier
t_before = time.time()
test_predictions = [NN_classifier(test_images[i]) for i in range(len(test_labels))]
t_after = time.time()

# Compute the error:
error = np.mean(test_predictions != test_labels)
print("Error of nearest neighbor classifier:", error)
print("Classification time (seconds):", t_after - t_before)

# 6. Faster nearest neighbor methods using BallTree
from sklearn.neighbors import BallTree
t_before = time.time()
train_images_reshaped = train_images.reshape(train_images.shape[0], -1)  # (60000, 784)
ball_tree = BallTree(train_images_reshaped)
t_after = time.time()
print("Time to build BallTree (seconds):", t_after - t_before)

# Classify test set using BallTree
t_before = time.time()
test_images_reshaped = test_images.reshape(test_images.shape[0], -1)  # (10000, 784)
test_neighbors = np.squeeze(ball_tree.query(test_images_reshaped, k=1, return_distance=False))
ball_tree_predictions = train_labels[test_neighbors]
t_after = time.time()
print("Time to classify test set using BallTree (seconds):", t_after - t_before)

# Verify predictions
print("Ball tree produces same predictions as NN classifier?", np.array_equal(test_predictions, ball_tree_predictions))

# 7. Faster nearest neighbor methods using KDTree
from sklearn.neighbors import KDTree
t_before = time.time()
kd_tree = KDTree(train_images_reshaped)
t_after = time.time()
print("Time to build KDTree (seconds):", t_after - t_before)

# Classify test set using KDTree
t_before = time.time()
test_neighbors = np.squeeze(kd_tree.query(test_images_reshaped, k=1, return_distance=False))
kd_tree_predictions = train_labels[test_neighbors]
t_after = time.time()
print("Time to classify test set using KDTree (seconds):", t_after - t_before)

# Verify predictions
print("KD tree produces same predictions as NN classifier?", np.array_equal(test_predictions, kd_tree_predictions))

