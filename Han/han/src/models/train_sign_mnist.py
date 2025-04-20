import os
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization  # Added import
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping  # Added import
import tensorflow as tf

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load Data
base_path = os.path.dirname(__file__)
train_path = os.path.normpath(os.path.join(
    base_path, "..", "data", "sign_mnist_train.csv"))
test_path = os.path.normpath(os.path.join(
    base_path, "..", "data", "sign_mnist_test.csv"))

if not os.path.exists(train_path) or not os.path.exists(test_path):
    raise FileNotFoundError("Training or testing data file not found.")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Prepare features and labels
X_train = train_df.drop("label", axis=1).values
y_train = train_df["label"].values
X_test = test_df.drop("label", axis=1).values
y_test = test_df["label"].values

# Remap labels to be sequential
unique_labels = np.unique(y_train)
label_map = {label: idx for idx, label in enumerate(unique_labels)}
y_train = np.array([label_map[label] for label in y_train])
y_test = np.array([label_map[label] for label in y_test])

# Normalize and reshape
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

# One-hot encode labels
num_classes = len(unique_labels)
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    BatchNormalization(),  # Added Batch Normalization
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),  # Added Batch Normalization
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),  # Added Batch Normalization
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# Callbacks
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    os.path.join(base_path, "..", "models", "sign_mnist_model.h5"),
    monitor='val_accuracy', save_best_only=True, mode='max', verbose=1
)
# Added learning rate scheduler
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
early_stop = EarlyStopping(monitor='val_loss', patience=3,
                           restore_best_weights=True)  # Added early stopping

# Train
epochs = 10
batch_size = 32
history = model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.2,
    callbacks=[checkpoint, lr_scheduler, early_stop],  # Added callbacks
    verbose=1
)

# Save model
# Changed to save in han/models instead of han/src/models/models
model_dir = os.path.join(base_path, "..", "models")
os.makedirs(model_dir, exist_ok=True)
model.save(os.path.join(model_dir, "sign_mnist_model.h5"))
print("Model saved successfully.")

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
