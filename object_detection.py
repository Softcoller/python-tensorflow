import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Impor dataset contoh dari TensorFlow
mnist = keras.datasets.mnist

# Bagi dataset menjadi data latih dan data uji
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalisasi data (0-255 menjadi 0-1)
train_images, test_images = train_images / 255.0, test_images / 255.0

# Bangun Model
model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Ubah gambar menjadi vektor 1D (28x28)
    layers.Dense(128, activation='relu'),  # Layer tersembunyi dengan 128 unit dan fungsi aktivasi ReLU
    layers.Dropout(0.2),  # Dropout layer untuk menghindari overfitting
    layers.Dense(10, activation='softmax')  # Layer output dengan 10 unit (klasifikasi 0-9) dan softmax activation
])

# Kompilasi Model
model.compile(optimizer='adam',  # Optimizer
              loss='sparse_categorical_crossentropy',  # Fungsi loss
              metrics=['accuracy'])  # Metrik untuk dinilai

# Latih Model
model.fit(train_images, train_labels, epochs=5)  # Latih model selama 5 epoch

# Evaluasi Model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Loss: {test_loss}, Accuracy: {test_acc}')