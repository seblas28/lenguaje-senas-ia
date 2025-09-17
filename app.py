# lenguaje-senas-ia/app.py
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, jsonify
from flask_cors import CORS # Importamos CORS
import threading
import time
import subprocess
import sys

# --- Configuración del Servidor Flask ---
app = Flask(__name__)

# --- ¡CONFIGURACIÓN DE CORS PARA PRODUCCIÓN! ---
# Reemplaza la URL de Vercel con la URL real de tu sitio web desplegado si es diferente.
# Esto le dice a nuestro servidor que SOLO acepte peticiones desde nuestra app de React.
origins = "https://kai-senas-web.vercel.app" 

CORS(app, resources={
    r"/receive_data": {"origins": origins}
})

def convert_model_to_web_format():
    print("\n--- Iniciando la Conversión del Modelo a Formato Web ---")
    command = [
        "tensorflowjs_converter",
        "--input_format=keras",
        "./sign_language_model.h5",
        # Esta ruta es para cuando el script se ejecuta localmente.
        # En Render, esto no funcionará, pero el modelo se guardará en el servidor.
        "../lenguaje-senas/public/tfjs_model"
    ]
    try:
        # Usamos shell=True en Windows para que encuentre el comando
        subprocess.run(command, check=True, shell=True)
        print("\n=> ¡Conversión completada con éxito!")
    except Exception as e:
        print(f"\n❌ ERROR durante la conversión: {e}")

def train_the_model(training_data):
    if not training_data: return
    time.sleep(1)
    print("\n--- Iniciando el Proceso de Entrenamiento del Modelo ---")
    df = pd.json_normalize(training_data)
    landmarks_data = []
    for index, row in df.iterrows():
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
    # convert_model_to_web_format() # Comentado para el despliegue en Render

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

# Ruta para verificar que el servidor está vivo y para "despertarlo"
@app.route('/')
def index():
    return "<h1>Servidor de Entrenamiento KaiSeñas Activo</h1><p>CORS habilitado para el endpoint /receive_data.</p>"

if __name__ == '__main__':
    print(">>> Servidor de Entrenamiento Local Iniciado <<<")
    app.run(port=5001, debug=False)


