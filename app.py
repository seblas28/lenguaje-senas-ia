# ==============================================================================
# SCRIPT DE ENTRENAMIENTO v4 - CON GUARDADO DE PRECISIÓN
# ==============================================================================
# Este script inicia un servidor local, recibe datos por categoría,
# entrena el modelo, guarda su precisión de validación, y automáticamente
# convierte y copia los archivos a tu proyecto de React.
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
import subprocess
import os

# --- Configuración del Servidor Flask ---
app = Flask(__name__)
CORS(app)

training_sumary_in_memory = {}

script_dir = os.path.dirname(os.path.abspath(__file__))
public_folder_path = os.path.abspath(os.path.join(script_dir, '..', 'lenguaje-senas','public'))
summary_file_path = os.path.join(public_folder_path, 'training_summary.json')

def update_training_summary(category, accuracy):
    global training_summary_in_memory
    
    # Actualiza el diccionario en memoria
    training_summary_in_memory[category] = accuracy
    print(f"✅ Resumen en memoria actualizado para '{category}' con una precisión de {accuracy:.2f}%.")
    
    # Guarda el diccionario actualizado en el archivo JSON
    with open(summary_file_path, 'w') as f:
        json.dump(training_summary_in_memory, f, indent=2)
    print(f"✅ Resumen de entrenamiento guardado en disco en: {summary_file_path}")


def convert_model_to_web_format(category):
    print(f"\n--- Iniciando Conversión del Modelo [{category}] ---")
    
    model_filename = f"./{category}_model.h5"
    output_path = os.path.join(public_folder_path, f"tfjs_{category}")
    public_labels_path = os.path.join(public_folder_path, f"labels_{category}.json")

    local_labels_filename = f"labels_{category}.json"

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
        # Usamos 'shell=True' en Windows para asegurar que el comando se encuentre
        subprocess.run(command, check=True, shell=True)
        if os.path.exists(local_labels_filename):
            os.rename(local_labels_filename, public_labels_path)
        print("\n=> ¡Conversión y copia completada!")
        print(f"=> El modelo para '{category}' ya está en tu proyecto de React.")
    except Exception as e:
        print(f"\n❌ ERROR durante la conversión: {e}")

def train_the_model(payload):
    category = payload.get('category')
    training_data = payload.get('data')

    if not training_data or not category: return
        
    time.sleep(1) 
    print(f"\n--- Iniciando Entrenamiento para [{category}] ---")

    if category == 'matematicas':
        try:
            with open('training_data_numeros.json', 'r') as f:
                numeros_data = json.load(f)
                print("✅ Datos de 'numeros' encontrados. Fusionando...")
                training_data.extend(numeros_data)
        except FileNotFoundError:
            print("\n⚠️ ADVERTENCIA: No se encontró 'training_data_numeros.json'.")
            print("   Asegúrate de entrenar la categoría 'numeros' primero.\n")
    
    df = pd.json_normalize(training_data)
    
    landmarks_data = []
    for index, row in df.iterrows():
        flat_landmarks = [coord for lm in row['landmarks'] for coord in [lm['x'], lm['y'], lm['z']]]
        landmarks_data.append(flat_landmarks)
        
    max_len = 126 
    padded_landmarks_data = [row + [0] * (max_len - len(row)) for row in landmarks_data]
    
    landmarks_df = pd.DataFrame(padded_landmarks_data)
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
        tf.keras.layers.InputLayer(input_shape=(max_len,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(len(classes), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    print("\n--- Entrenando... ---")
    history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_val, y_val), verbose=1)

    model_filename = f"{category}_model.h5"
    model.save(model_filename)
    print(f"\n✅ Modelo entrenado y guardado como '{model_filename}'.")
    
    # Obtenemos la mejor precisión de validación y la guardamos
    best_accuracy = max(history.history['val_accuracy']) * 100
    update_training_summary(category, best_accuracy)
    
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

