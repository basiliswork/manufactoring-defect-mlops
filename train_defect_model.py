import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
import cv2

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# 1. Generate Synthetic Manufacturing Data
def generate_factory_data(num_samples=200, img_size=(128, 128, 3)):
    X = np.ones((num_samples, *img_size)) * 200  
    y = np.random.randint(0, 2, num_samples)
    
    for i in range(num_samples):
        noise = np.random.normal(0, 10, img_size)
        X[i] = np.clip(X[i] + noise, 0, 255)
        if y[i] == 1:
            x1, y1 = np.random.randint(20, 100, 2)
            x2, y2 = x1 + np.random.randint(10, 20), y1 + np.random.randint(10, 20)
            cv2.line(X[i], (x1, y1), (x2, y2), (50, 50, 50), 3)
            
    X = X / 255.0
    return X, y

X, y = generate_factory_data()
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 2. Build and Train the Model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False  

model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=16, validation_data=(X_test, y_test))

# 3. Export for MLOps using the Keras v3 format
MODEL_PATH = "defect_detection_model.keras"
model.save(MODEL_PATH)
print(f"\nArtifact saved to {MODEL_PATH}")