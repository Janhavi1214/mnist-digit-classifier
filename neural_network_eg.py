import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess the data: flatten the images and normalize the pixel values
train_images = train_images.reshape((train_images.shape[0], 28 * 28)).astype('float32') / 255
test_images = test_images.reshape((test_images.shape[0], 28 * 28)).astype('float32') / 255

# Build the model
model = Sequential([
    Dense(128, activation='relu', input_shape=(28 * 28,)),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")

import numpy as np

digit = test_images[0].reshape(1, 784)  # pick first test image, reshape for model
prediction = model.predict(digit)
print(prediction)              # probabilities for digits 0–9
print(np.argmax(prediction))   # digit with highest probability
print("Actual label:", test_labels[0])

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Get predictions for all test images
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Generate confusion matrix
cm = confusion_matrix(test_labels, predicted_labels)

# Plot
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - MNIST Digit Classifier')
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()# Visualize a sample prediction
def plot_sample_prediction(index):
    img = test_images[index].reshape(28, 28)
    digit = test_images[index].reshape(1, 784)
    
    prediction = model.predict(digit)
    predicted_label = np.argmax(prediction)
    actual_label = test_labels[index]
    
    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap='gray')
    plt.title(f"Predicted: {predicted_label}  |  Actual: {actual_label}",
              fontsize=13,
              color='green' if predicted_label == actual_label else 'red')
    plt.axis('off')
    plt.savefig('sample_prediction.png', dpi=150, bbox_inches='tight')
    plt.show()

# Call with any test index
plot_sample_prediction(0)
