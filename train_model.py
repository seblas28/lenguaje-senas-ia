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
import os

# --- Configuración del Servidor Flask ---
app = Flask(__name__)
CORS(app)

def convert_model_to_web_format(category):
    print(f"\n--- Iniciando Conversión del Modelo [{category}] ---")
    
    model_filename = f"./{category}_model.h5"
    labels_filename = f"./labels_{category}.json"
    output_path = f"../lenguaje-senas/public/tfjs_{category}"

    if not os.path.exists(model_filename):
        print(f"❌ ERROR: No se encontró el archivo del modelo '{model_filename}'.")
        return

    command = [
        "tensorflowjs_converter",
        "--input_format=keras",
        model_filename,
        output_path
    ]
    
    try:
        subprocess.run(command, check=True, shell=True)
        # Movemos el archivo de etiquetas al directorio público de React
        os.rename(labels_filename, f"../lenguaje-senas/public/{labels_filename}")
        print("\n=> ¡Conversión y copia completada!")
        print(f"=> El modelo para '{category}' ya está en tu proyecto de React.")
    except Exception as e:
        print(f"\n❌ ERROR durante la conversión: {e}")

def train_the_model(payload):
    category = payload.get('category')
    training_data = payload.get('data')

    if not training_data or not category:
        return

    time.sleep(1) 
    print(f"\n--- Iniciando Entrenamiento para [{category}] ---")
    
    df = pd.json_normalize(training_data)
    
    landmarks_data = [ [coord for lm in row['landmarks'] for coord in [lm['x'], lm['y'], lm['z']]] for _, row in df.iterrows() ]
    landmarks_df = pd.DataFrame(landmarks_data)
    data = pd.concat([df['label'], landmarks_df], axis=1)

    X = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    classes = label_encoder.classes_
    
    labels_filename = f"labels_{category}.json"
    with open(labels_filename, 'w') as f:
        json.dump(classes.tolist(), f)
    print(f"✅ Etiquetas para '{category}' guardadas: {classes}")

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

    model_filename = f"{category}_model.h5"
    model.save(model_filename)
    print(f"\n✅ Modelo entrenado y guardado como '{model_filename}'.")
    
    convert_model_to_web_format(category)

@app.route('/receive_data', methods=['POST'])
def receive_data():
    try:
        payload = request.get_json()
        category = payload.get('category')
        data = payload.get('data')
        
        if not category or not data:
            return jsonify({"status": "error", "message": "Payload incompleto"}), 400

        with open(f'training_data_{category}.json', 'w') as f:
            json.dump(data, f, indent=2)
        print(f"=> Datos para '{category}' recibidos y guardados.")
        
        training_thread = threading.Thread(target=train_the_model, args=(payload,))
        training_thread.start()
        
        return jsonify({"status": "success", "message": "Training started."}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    print(">>> Servidor de Entrenamiento Local Iniciado <<<")
    print(">>> Esperando datos en http://localhost:5001/receive_data ...")
    app.run(port=5001, debug=False)

