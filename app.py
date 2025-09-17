# ==============================================================================
# SCRIPT DE ENTRENAMIENTO FINAL Y ROBUSTO PARA RENDER
# ==============================================================================
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

# --- Configuración del Servidor Flask ---
app = Flask(__name__)

# --- ¡LA CONFIGURACIÓN DE CORS MÁS SIMPLE Y ROBUSTA! ---
# Esto permite peticiones desde cualquier origen a cualquier ruta.
# Es perfecto y seguro para una herramienta de presentación como esta.
CORS(app)

def train_the_model(training_data):
    if not training_data: return
    time.sleep(1)
    print("\n--- Iniciando el Proceso de Entrenamiento del Modelo ---")
    
    # Preprocesamiento de Datos
    df = pd.json_normalize(training_data)
    landmarks_data = [ [coord for lm in row['landmarks'] for coord in [lm['x'], lm['y'], lm['z']]] for _, row in df.iterrows() ]
    landmarks_df = pd.DataFrame(landmarks_data)
    data = pd.concat([df['label'], landmarks_df], axis=1)
    X = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values

    # Lógica de Entrenamiento
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    classes = label_encoder.classes_
    print(f"✅ Etiquetas encontradas: {classes}")
    X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(len(classes), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    print("\n--- Entrenando... ---")
    model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_val, y_val), verbose=1)
    model.save('sign_language_model.h5')
    print("\n✅ Modelo entrenado y guardado como 'sign_language_model.h5'.")
    print("\n>>> El entrenamiento ha finalizado. El servidor sigue activo. <<<")


@app.route('/receive_data', methods=['POST', 'OPTIONS'])
def receive_data():
    if request.method == 'OPTIONS':
        return '', 204
    try:
        data = request.get_json()
        with open('training_data.json', 'w') as f: json.dump(data, f, indent=2)
        print("=> Datos recibidos y guardados.")
        training_thread = threading.Thread(target=train_the_model, args=(data,))
        training_thread.start()
        return jsonify({"status": "success", "message": "Training started."}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/')
def index():
    return "<h1>Servidor de Entrenamiento KaiSeñas Activo (v2)</h1>"

# Esta parte solo se usa para pruebas locales. Render usa el Start Command.
if __name__ == '__main__':
    print(">>> Servidor de Entrenamiento Local Iniciado <<<")
    app.run(port=5001, debug=False)