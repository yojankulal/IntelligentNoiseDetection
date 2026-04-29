import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

IMG_SIZE = 128
DATASET = "dataset"

classes = ["clean", "gaussian", "salt_pepper", "speckle", "motion_blur"]
label_map = {c: i for i, c in enumerate(classes)}

X = []
y = []

# ---- LOAD DATA ----
for label in classes:
    folder = os.path.join(DATASET, label)

    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        img = cv2.imread(path)
        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0

        X.append(img)
        y.append(label_map[label])

X = np.array(X)
y = to_categorical(y, num_classes=len(classes))

print("Dataset loaded:", X.shape)

# ---- CNN MODEL ----
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(classes), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ---- TRAIN ----
model.fit(X, y, epochs=10, batch_size=16, validation_split=0.2)

# ---- SAVE MODEL ----
model.save("cnn_noise_model.h5")

print("✅ CNN model trained and saved!")