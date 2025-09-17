import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import time
import subprocess
import sys

# --- Configuración del Servidor Flask ---
app = Flask(__name__)
CORS(app)

def convert_model_to_web_format():
    print("\n--- Iniciando la Conversión del Modelo a Formato Web ---")
    # Este comando es para un entorno local. En Render, la conversión no es necesaria
    # ya que no podemos acceder al sistema de archivos del frontend directamente.
    # El modelo debe ser descargado manualmente desde los logs o una ubicación de almacenamiento.
    print("=> En un entorno de producción como Render, descarga el archivo 'sign_language_model.h5'")
    print("=> y conviértelo manualmente en tu máquina local.")
    
def train_the_model(training_data):
    if not training_data:
        print("Advertencia: No se recibieron datos de entrenamiento.")
        return

    time.sleep(1) 
    print("\n--- Iniciando el Proceso de Entrenamiento del Modelo ---")
    
    # 1. Cargar y Preprocesar los Datos
    df = pd.json_normalize(training_data)
    
    # --- ¡LA CORRECCIÓN CRUCIAL ESTÁ AQUÍ! ---
    landmarks_data = []
    for index, row in df.iterrows():
        # Aplanamos la lista de 21 diccionarios en una sola lista de 63 números
        flat_landmarks = [coord for lm in row['landmarks'] for coord in [lm['x'], lm['y'], lm['z']]]
        landmarks_data.append(flat_landmarks)
        
    landmarks_df = pd.DataFrame(landmarks_data)
    data = pd.concat([df['label'], landmarks_df], axis=1)

    X = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    classes = label_encoder.classes_
    np.save('classes.npy', classes)
    print(f"✅ Etiquetas encontradas: {classes}")

    X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # 2. Construcción del modelo
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(len(classes), activation='softmax')
    ])

    # 3. Compilación del modelo
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    
    # 4. Entrenamiento del modelo
    print("\n--- Entrenando... ---")
    model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_val, y_val), verbose=1)

    # 5. Guardar el modelo
    model.save('sign_language_model.h5')
    print("\n✅ Modelo entrenado y guardado como 'sign_language_model.h5'.")
    
    # (La conversión automática se comenta para Render)
    # convert_model_to_web_format()

@app.route('/receive_data', methods=['POST'])
def receive_data():
    try:
        data = request.get_json()
        with open('training_data.json', 'w') as f:
            json.dump(data, f, indent=2)
        print("=> Datos recibidos y guardados exitosamente.")
        training_thread = threading.Thread(target=train_the_model, args=(data,))
        training_thread.start()
        return jsonify({"status": "success", "message": "Training started."}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    print(">>> Servidor de Entrenamiento Iniciado <<<")
    # Render usa gunicorn, pero esta línea permite la ejecución local
    app.run(port=5001, debug=False)