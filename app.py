import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import subprocess
import sys
import time

# --- CONFIGURACIÓN DEL SERVIDOR FLASK ---
app = Flask(__name__)
CORS(app)

def convert_model_to_web_format():
    print("\n--- Iniciando la Conversión del Modelo a Formato Web ---")
    command = [
        "tensorflowjs_converter",
        "--input_format=keras",
        "./sign_language_model.h5",
        "../lenguaje-senas/public/tfjs_model"
    ]
    try:
        subprocess.run(command, check=True, shell=True)
        print("\n=> ¡Conversión completada con éxito!")
        print("=> El modelo actualizado ya está en la carpeta de tu proyecto de React.")
    except Exception as e:
        print(f"\n❌ ERROR durante la conversión: {e}")

def train_the_model(training_data):
    if not training_data:
        print("Advertencia: No se recibieron datos de entrenamiento.")
        return

    # Pequeña pausa para que los mensajes del servidor no se mezclen con los del entrenamiento
    time.sleep(1) 
    print("\n--- Iniciando el Proceso de Entrenamiento del Modelo ---")
    
    # 1. Cargar y Preprocesar los Datos
    df = pd.json_normalize(training_data)
    
    # --- ¡CORRECCIÓN CLAVE AQUÍ! Aplanamos los landmarks ---
    landmarks_data = []
    for index, row in df.iterrows():
        # 'row['landmarks']' es una lista de 21 diccionarios ({x, y, z})
        # La convertimos en una sola lista de 63 números [x1, y1, z1, x2, y2, z2, ...]
        flat_landmarks = [coord for lm in row['landmarks'] for coord in [lm['x'], lm['y'], lm['z']]]
        landmarks_data.append(flat_landmarks)
        
    landmarks_df = pd.DataFrame(landmarks_data)
    
    # Unimos las etiquetas con los datos aplanados
    data = pd.concat([df['label'], landmarks_df], axis=1)

    # Ahora sí, separamos X e y con el formato correcto
    X = data.iloc[:, 1:].values  # Todas las columnas excepto la primera (label)
    y = data.iloc[:, 0].values   # Solo la primera columna (label)

    # Codificar las etiquetas de texto a números (0, 1, 2, ...)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    classes = label_encoder.classes_
    np.save('classes.npy', classes)
    print(f"✅ Etiquetas encontradas: {classes}")

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # 2. Construcción del modelo de red neuronal
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)), # La forma de entrada es ahora correcta (63)
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(len(classes), activation='softmax')
    ])

    # 3. Compilación del modelo
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', # Usamos esta loss porque nuestras etiquetas son números enteros (0, 1, 2...)
                  metrics=['accuracy'])
    
    model.summary()
    
    # 4. Entrenamiento del modelo
    print("\n--- Entrenando... ---")
    model.fit(X_train, y_train,
              epochs=50,
              batch_size=16, # Añadir batch_size es una buena práctica
              validation_data=(X_val, y_val),
              verbose=1) # Usamos verbose=1 para ver la barra de progreso

    # 5. Guardar el modelo
    model.save('sign_language_model.h5')
    print("\n✅ Modelo entrenado y guardado como 'sign_language_model.h5'.")
    
    # 6. Conversión automática
    convert_model_to_web_format()

# --- Servidor Flask y Flujo Principal (sin cambios) ---
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
    print(">>> Esperando datos en http://127.0.0.1:5001/receive_data ...")
    app.run(port=5001, debug=False)

