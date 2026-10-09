import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import os
data_dir = r'C:\Users\obiko\Downloads\archive (2)\F1_new_upd_car_data'
img_size = (150, 150)
batch_size = 16
epochs = 15
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2,
                             rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
                             shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_gen = datagen.flow_from_directory(data_dir, target_size=img_size, batch_size=batch_size,
                                        class_mode='sparse', subset='training')
val_gen = datagen.flow_from_directory(data_dir, target_size=img_size, batch_size=batch_size,
                                      class_mode='sparse', subset='validation', shuffle=False)
num_classes = len(train_gen.class_indices)
classes = list(train_gen.class_indices.keys())
print(f"Detected classes: {classes}")
# Task C: Computer Vision - My Local 4-Layer Convolutional Neural Network
# I designed this architecture to classify F1 car liveries by extracting 
# spatial features through increasing filter depths.
model = keras.Sequential([
    layers.Input(shape=(150, 150, 3)),
    # I start with 32 filters to capture basic edges and shapes.
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    # I double the filters at each stage (64, 128, 256) to learn more complex patterns.
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(256, (3,3), activation='relu'),
    # Each layer is followed by MaxPooling to reduce the spatial dimensions.
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    # I specifically used a Dropout layer at 50% to prevent overfitting and improve generalization.
    layers.Dropout(0.5),
    # The final layer uses Softmax implicit in the loss for team classification.
    layers.Dense(num_classes)
])
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model.fit(train_gen, epochs=epochs, validation_data=val_gen)
model.save('Enoch_F1_trained.h5')
print("Model saved as Enoch_F1_trained.h5")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(history.history['accuracy'], label='Train Accuracy')
ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
ax1.set_title('Accuracy vs Epochs')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)
ax2.plot(history.history['loss'], label='Train Loss')
ax2.plot(history.history['val_loss'], label='Val Loss')
ax2.set_title('Loss vs Epochs')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)
plt.tight_layout()
plt.savefig('training_curves.png')
print("Training curves saved as training_curves.png")
val_gen.reset()
preds = model.predict(val_gen)
pred_classes = np.argmax(preds, axis=1)
true_classes = val_gen.classes
print(classification_report(true_classes, pred_classes, target_names=classes))
cm = confusion_matrix(true_classes, pred_classes)
plt.figure(figsize=(10,8))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved as confusion_matrix.png")
per_class_acc = cm.diagonal() / cm.sum(axis=1)
print("\nPer-class accuracy:")
for cls, acc in zip(classes, per_class_acc):
    print(f"  {cls}: {acc*100:.1f}%")