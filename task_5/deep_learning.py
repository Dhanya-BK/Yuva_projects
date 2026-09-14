import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 1. Load Lightweight Dataset (~30MB download)
(X_train_full, y_train_full), (X_test_full, y_test_full) = fashion_mnist.load_data()

# 2. Slice to a small dataset (3,000 train images, 1,000 test images)
X_train = X_train_full[:3000].reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = y_train_full[:3000]

X_test = X_test_full[:1000].reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_test = y_test_full[:1000]

# One-hot encode labels for 10 clothing classes
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

print(f"Training on: {X_train.shape[0]} images")
print(f"Testing on: {X_test.shape[0]} images")

# 3. Build Lightweight CNN Model
def build_small_cnn():
    model = models.Sequential()

    # Block 1
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    # Block 2
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.30))

    # Dense Classifier
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.40))
    model.add(layers.Dense(10, activation='softmax'))

    return model

model = build_small_cnn()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 4. Callbacks & Fast Training Setup
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
]

# 5. Execute Training
history = model.fit(
    X_train, y_train_cat,
    batch_size=32,
    epochs=15,
    validation_data=(X_test, y_test_cat),
    callbacks=callbacks
)

# 6. Final Evaluation
test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\nFinal Test Accuracy: {test_acc * 100:.2f}%")
print(f"Final Test Loss: {test_loss:.4f}")