# Entrenamiento local de Kai Señas

El entrenador funciona en tu equipo con una interfaz de escritorio y, opcionalmente, comandos. No inicia Flask, no recibe datos por HTTP y no escribe en el frontend. El resultado es `model.kai.json`, un paquete Kai v1 que ya admite Administración → Modelos de IA.

El flujo es: **capturar → descargar datasets → auditar → entrenar → revisar → subir → aprobar → publicar**. Entrenar no concede permisos ni publica nada.

## Interfaz de escritorio (Windows)

Una vez preparado `.venv` con las dependencias de abajo, abre **`Abrir-entrenador.vbs` con doble clic**. Selecciona el intérprete del entorno del proyecto automáticamente y abre la aplicación sin una consola. No instala dependencias ni cambia configuraciones de Windows.

1. Elige el módulo. La interfaz carga los archivos `dataset-MODULO-*.json` de `datasets` si existen; usa **Añadir archivos** para otras ubicaciones o nombres.
2. Mantén Etiquetas vacío para el conjunto predeterminado o escribe una cobertura parcial como `B, C, Nulo`.
3. Pulsa **Auditar capturas** y revisa conteos y problemas. Entrenar se habilita solo al superar esa auditoría. Los archivos se vuelven a comprobar al empezar el entrenamiento.
4. Elige un nombre de versión nuevo, ajusta épocas/lote/hilos si lo necesitas y pulsa **Entrenar modelo**.
5. La pestaña Entrenamiento muestra las épocas y el registro. TensorFlow funciona en un proceso separado; la ventana sigue disponible. Los registros completos se conservan en `artifacts/gui-logs`.
6. **Cancelar** solicita una parada cooperativa: espera un punto seguro, marca la ejecución y libera su bloqueo. Durante la carga inicial de TensorFlow puede tardar; no fuerza la terminación del proceso. Cerrar la ventana durante un trabajo pregunta si quieres esperar su cancelación.
7. Resultados muestra validación/test y matriz de confusión. **Abrir informe anterior** permite consultar un `training-report.json` existente. **Copiar ruta del paquete** se habilita cuando la carpeta tiene una ejecución completa y `model.kai.json`.
8. Evalúa la calidad y sube el paquete manualmente desde Administración; la aplicación local no inicia sesión ni aprueba ni publica modelos.

La ventana es redimensionable y requiere al menos 960 × 700 píxeles. Utiliza Tkinter incluido en Python para Windows; no agrega paquetes de interfaz a `requirements.txt`. La consola sigue disponible como alternativa si necesitas un diagnóstico. Si el equipo bloquea VBScript, puedes crear un acceso directo cuyo destino sea `"RUTA-DEL-PROYECTO\.venv\Scripts\pythonw.exe" "RUTA-DEL-PROYECTO\entrenador.pyw"`.

## 1. Preparar un entorno nuevo

Usa **Python 3.12 de 64 bits**. Este conjunto fija TensorFlow 2.20.0 y NumPy 2.2.6. El entrenamiento admite Python 3.10–3.13; Python 3.15 no es compatible con este conjunto. Puedes conservar tu Python actual e instalar 3.12 en paralelo.

Instala Python 3.12 desde [Python para Windows](https://www.python.org/downloads/windows/). Después abre PowerShell y ejecuta desde el repositorio:

```powershell
Set-Location -LiteralPath "D:\Proyectos\Proyecto-Kaisenas\lenguaje-senas-ia"
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Si el lanzador `py` no encuentra Python 3.12, usa la ruta completa al `python.exe` de esa instalación para crear `.venv`. No hace falta activar el entorno: los ejemplos invocan directamente su intérprete.

**No reutilices `ia_env`:** contiene rutas absolutas de otra instalación/computadora. No mezcles estas dependencias con TensorFlow/Keras 2.10, tensorflowjs, Flask o el entorno antiguo. La nueva exportación no necesita `tensorflowjs_converter`, pandas ni scikit-learn.

En Windows nativo se utiliza CPU con TensorFlow moderno; la aceleración CUDA requiere una configuración separada en WSL2. No cambies tu sistema a WSL para empezar: la red actual es pequeña. Referencias: [instalación oficial](https://www.tensorflow.org/install/pip), [archivos y versiones Python de TensorFlow 2.20](https://pypi.org/project/tensorflow/2.20.0/).

## 2. Reunir capturas útiles

En la web, entra con admin o superadmin a Administración → Captura de muestras. Captura y descarga datasets completos del módulo en **al menos tres sesiones independientes**, guardando un JSON diferente por sesión. Ejemplo:

```text
datasets/
  vocales-sesion-1.json
  vocales-sesion-2.json
  vocales-sesion-3.json
```

Cada etiqueta necesita al menos 20 muestras únicas en total y aparecer en tres archivos con capturas distintas. Cada conjunto final —entrenamiento, validación y test— necesita al menos tres muestras únicas de cada etiqueta. Es un mínimo operativo, no una garantía de precisión: recoge muchas más y varía personas, distancia, orientación, iluminación y fondo.

**Nulo es obligatorio.** Captura poses ajenas a las etiquetas positivas, movimientos de transición y otras posiciones de manos. No etiquetes una misma pose como positiva y Nulo. Renombrar o copiar un archivo no constituye otra sesión. Los archivos idénticos y las poses duplicadas no suman capturas nuevas.

El formato actual de descarga es `{ "category": "vocales", "data": [{ "label": "E", "landmarks": [...] }] }`. También se leen las listas de los `training_data_*.json` antiguos, pero deben superar las mismas comprobaciones. Las coordenadas deben estar normalizadas por la web: x/y relativas a la primera muñeca, z sin modificar. No vuelvas a normalizarlas ni reordenes las manos. Una mano ocupa 63 valores y se completa con ceros hasta 126; dos manos ocupan los 126.

## 3. Auditar antes de entrenar

```powershell
.\.venv\Scripts\python.exe app.py audit --category vocales --data ".\datasets\vocales-sesion-1.json" ".\datasets\vocales-sesion-2.json" ".\datasets\vocales-sesion-3.json"
```

La auditoría no necesita TensorFlow; también puede ejecutarse con otro Python moderno. Devuelve un JSON con conteos, duplicados, cobertura, grupos asignados y problemas. `ready_to_train: true` y código de salida 0 indican que supera las comprobaciones de datos. Código 2 indica que hay que corregir los problemas antes de entrenar.

Por defecto exige las etiquetas del módulo. Para probar una versión parcial, declara explícitamente su cobertura:

```powershell
.\.venv\Scripts\python.exe app.py audit --category abecedario --labels B C Nulo --data ".\datasets\abc-1.json" ".\datasets\abc-2.json" ".\datasets\abc-3.json"
```

Las muestras de otras etiquetas válidas del módulo se excluyen y se contabilizan; una etiqueta desconocida se rechaza. En abecedario, el conjunto predeterminado son las consonantes estáticas. **J y Z quedan bloqueadas** porque requieren movimiento y evaluación temporal, ausentes en este formato de poses. Puedes añadir datasets de vocales al entrenar abecedario y declarar sus etiquetas si necesitas una cobertura distinta.

### Matemáticas

La calculadora necesita números además de Suma, Resta, Igual y Nulo. Debes pasar explícitamente las capturas de matemáticas **y** de números en `--data`. No se mezclan archivos antiguos de forma automática. Reúne al menos tres sesiones por cada conjunto de etiquetas; pueden ser seis archivos (tres de operadores y tres de números). Los operadores necesitan muestras de dos manos, como indica la referencia de la web.

## 4. Entrenar una versión

```powershell
.\.venv\Scripts\python.exe app.py train --category vocales --version vocales-local-v1 --data ".\datasets\vocales-sesion-1.json" ".\datasets\vocales-sesion-2.json" ".\datasets\vocales-sesion-3.json"
```

Opciones: `--epochs 50`, `--batch-size 16`, `--patience 8`, `--threads 2`, `--seed 42`, `--output artifacts`. Para una versión parcial repite `--labels B C Nulo`, igual que en la auditoría. No se sobrescribe una versión existente: usa otro nombre para repetirla.

La red conserva el contrato de 126 coordenadas y Dense 128/ReLU → Dropout → Dense 64/ReLU → Dropout → salida Softmax. Se separa por archivos completos para impedir que frames de una captura aparezcan en entrenamiento y evaluación. Se intenta reservar aproximadamente 20 % de los archivos para validación y 20 % para test; aumenta esas proporciones si hace falta para cubrir todas las etiquetas. El informe registra los conjuntos exactos. Si no encuentra una separación válida, falla antes de entrenar.

Se selecciona el checkpoint de menor pérdida de validación, se restauran esos pesos y se evalúa el mismo modelo sobre validación y test. El test no selecciona el checkpoint. La precisión publicada en el informe corresponde a ese modelo, no al máximo de otra época.

Al finalizar verás:

```text
artifacts/vocales/vocales-local-v1/
  model.kai.json          <- único archivo a seleccionar para subir en la web
  model.keras             <- modelo local para inspección posterior
  best.weights.h5         <- checkpoint elegido
  labels_vocales.json
  dataset-report.json     <- composición, duplicados y separación
  training-report.json    <- métricas, matriz de confusión e historial
  environment.txt         <- versiones realmente instaladas
  status.json             <- complete / failed y huellas de los archivos
```

Solo utiliza una ejecución con `status.json` en estado `complete`. Si falla, se interrumpe o la cancelas desde la interfaz, permanece marcada como `failed`, `interrupted` o `cancelled` y no debe subirse. El archivo `artifacts/.training.lock` impide entrenamientos simultáneos en el mismo directorio de salida. Si queda tras un cierre abrupto, comprueba primero que no siga entrenando el proceso cuyo PID contiene, y solo entonces retira ese bloqueo. No se borra automáticamente un bloqueo que podría pertenecer a otro proceso.

## 5. Revisar y enviar a aprobación

1. Revisa `training-report.json`: precisión en test, matriz de confusión y recall por etiqueta. Comprueba especialmente E/O y Nulo.
2. Evalúa el resultado con capturas/personas que no participaron en los archivos usados. La separación por archivo evita una fuga concreta, pero no demuestra independencia de personas ni elimina toda similitud entre sesiones.
3. Desde Administración → Modelos de IA, selecciona el mismo módulo y un nombre de versión coherente con el entrenamiento. Describe los cambios y limitaciones.
4. Selecciona **solo `model.kai.json`**. No incluyas junto a él `.keras`, informes o datasets; el frontend reconoce el paquete individual.
5. Admin puede subir. Superadmin comprueba, aprueba y publica en pasos separados. La comprobación de compatibilidad no prueba precisión real.

El paquete usa la estructura Kai v1, pesos float32 little endian y etiquetas en el mismo orden de la salida. No contiene rutas del equipo, datasets ni código ejecutable. El límite es 2 MiB. El entrenamiento no llama a Supabase y no incluye credenciales.

## Diagnóstico y pruebas

```powershell
.\.venv\Scripts\python.exe app.py --help
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Las pruebas verifican validación, deduplicación, agrupación, rechazo de etiquetas/rutas, protección de versiones y formato binario. La prueba de integración de TensorFlow entrena con datos **sintéticos** y comprueba los archivos de salida. Las pruebas de los procesos de la interfaz también ejecutan auditoría, entrenamiento y cancelación reales. Para ejecutar la suite completa usa el Python compatible del entorno con TensorFlow instalado; estas pruebas no miden calidad para reconocer señas.

Los datasets antiguos auditados en esta revisión contienen 420/14 muestras totales/únicas en vocales, 60/2 en abecedario, 780/26 en números y 120/4 en matemáticas. Ninguno supera la auditoría nueva: necesitas recopilar capturas nuevas, no repetir o copiar las anteriores.
