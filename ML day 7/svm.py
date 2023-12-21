import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape the data to be suitable for the SVM classifier
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

# Split the data into training and testing sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42)

# Create the SVM classifier
clf = svm.LinearSVC()

# Train the classifier
clf.fit(x_train, y_train)

# Evaluate the classifier on the test set
y_pred = clf.predict(x_test)
accuracy = np.sum(y_pred == y_test) / len(y_test)

# Print the accuracy
print("Accuracy:", accuracy)