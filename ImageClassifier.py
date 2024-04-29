import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import random
import numpy as np

# Load the CIFAR-10 dataset
(train_images,train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize the pixel value between 0 and 1
train_images, test_images = train_images/255.0, test_images / 255.0

# Class Name
class_names = ['airplane','automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# plot the image
plt.figure(figsize = (10, 10))
for i in range(25):
  plt.subplot(5,5, i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(train_images[i])
  plt.xlabel(class_names[train_labels[i][0]])
plt.show()

# Create the Convolutional Neural Network
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation= 'relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(10))

# Model Summary
model.summary()

# Compile and Train the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics = ['accuracy'])

history = model.fit(train_images, train_labels, epochs =10, validation_data = (test_images, test_labels))

# Plot the accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
test_loss, test_acc = model.evaluate(test_images , test_labels, verbose =2 )


# Select the random image from test dataset
idx = random.randint(0, len(test_labels))
plt.imshow(test_images[idx])
plt.show()

# Predict the label of the image
# y_pred consist of probability distribution of each category
y_pred = model.predict(test_images[idx].reshape(1, 32, 32, 3)) 
predicted_label_index = np.argmax(y_pred) #  return the index of max value i.e. probability
predicted_label = class_names[predicted_label_index]
print(predicted_label)