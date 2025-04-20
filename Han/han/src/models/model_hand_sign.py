import pandas as pd
import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2  # type: ignore
from tensorflow.keras.models import Model   # type: ignore
from tensorflow.keras.layers import Dense  # type: ignore
from tensorflow.keras.layers import GlobalAveragePooling2D  # type: ignore
from tensorflow.keras.layers import Input  # type: ignore
from tensorflow.keras.utils import to_categorical  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore

# Load data
csv_path = r"c:\Users\venkatesh\Han\han\src\data\sign_mnist_test.csv"

csv_path = os.path.abspath(csv_path)
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV file not found at {csv_path}")
df = pd.read_csv(csv_path)
print(f"Total samples: {len(df)}")

# Prepare data
X, y = [], []
for _, row in df.iterrows():
    img_dir = os.path.dirname(__file__)
    img_path = os.path.abspath(os.path.join(
        img_dir, "..", row["<correct_column_name>"]))
    label = row["label"]

    image = cv2.imread(img_path)
    if image is None:
        print(f"Warning: Couldn't read image at {img_path}")
        continue

    image = cv2.resize(image, (224, 224)) / 255.0
    X.append(image)
    y.append(label)

X = np.array(X)
y_labels = sorted(list(set(y)))
label_map = {label: idx for idx, label in enumerate(y_labels)}
y = np.array([label_map[label] for label in y])
y = to_categorical(y, num_classes=len(label_map))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load MobileNetV2 base
input_tensor = Input(shape=(224, 224, 3))
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_tensor=input_tensor
)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(len(label_map), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(
    optimizer=Adam(),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

# Train
model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=16,
    validation_data=(X_test, y_test)
)

# Save model
model_dir = os.path.dirname(__file__)
model_path = os.path.join(model_dir, "hand_sign_mobilenetv2.h5")
model.save(model_path)
print(f"Model saved to {model_path}")
