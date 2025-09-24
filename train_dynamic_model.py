import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- PARÁMETROS CLAVE ---
SEQUENCE_LENGTH = 30  # El número de fotogramas que componen un gesto
NUM_FEATURES = 63     # 21 landmarks * 3 coordenadas (x, y, z)

# Cargar los datos (asume que los datos ya están guardados)
with open('training_data_dynamic.json', 'r') as f:
    data = json.load()

# --- PREPROCESAMIENTO ---
X = []
y = []

for sample in data:
    sequence = []
    for frame in sample['sequence']:
        # Aplanar cada fotograma
        flat_frame = [coord for lm in frame for coord in [lm['x'], lm.get('y', 0), lm.get('z', 0)]]
        sequence.append(flat_frame)
    
    # Asegurarse de que todas las secuencias tengan la misma longitud
    if len(sequence) == SEQUENCE_LENGTH:
        X.append(sequence)
        y.append(sample['label'])

X = np.array(X)
y = np.array(y)

# Codificar etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = tf.keras.utils.to_categorical(y_encoded)
np.save('classes_dynamic.npy', label_encoder.classes_)
print(f"Clases dinámicas encontradas: {label_encoder.classes_}")

X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# --- CONSTRUCCIÓN DEL MODELO LSTM ---
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, activation='relu', input_shape=(SEQUENCE_LENGTH, NUM_FEATURES)),
    tf.keras.layers.LSTM(128, return_sequences=True, activation='relu'),
    tf.keras.layers.LSTM(64, return_sequences=False, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(y_categorical.shape[1], activation='softmax')
])

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.summary()

# --- ENTRENAMIENTO ---
model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))
model.save('dynamic_signs_model.h5')
print("Modelo dinámico entrenado y guardado.")
