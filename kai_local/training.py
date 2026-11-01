"""Training is imported only after the dataset has passed the audit."""
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from .bundle import make_bundle
from .data import DatasetError


class TrainingCancelled(Exception):
    """Cooperative cancellation requested by the local interface."""


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def train(category, version, labels, parts, audit, output, epochs=50, batch_size=16, patience=8, seed=42, threads=2, cancel_file=None):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,39}", version or ""):
        raise DatasetError("Versión inválida: usa hasta 40 letras, números, puntos, guiones o guiones bajos.")
    if not (1 <= epochs <= 1000 and 1 <= batch_size <= 1024 and 1 <= patience <= 100 and 1 <= threads <= 16):
        raise DatasetError("Parámetros fuera de rango: epochs 1–1000, batch 1–1024, patience 1–100, threads 1–16.")
    output = Path(output).resolve()
    target = output / category / version
    if target.exists():
        raise DatasetError(f"La versión ya existe y no se sobrescribirá: {target}")
    if sys.version_info[:2] < (3, 10) or sys.version_info[:2] > (3, 13):
        raise DatasetError("Este entorno de entrenamiento requiere Python 3.10–3.13. Se recomienda crear .venv con Python 3.12; no reutilices ia_env.")
    if audit["problems"] or not parts:
        raise DatasetError("El dataset no supera la auditoría.")
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".training.lock"
    try:
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise DatasetError(f"Existe un entrenamiento en curso o un bloqueo de una ejecución interrumpida: {lock}. Comprueba el proceso antes de retirar ese archivo.") from exc
    run_created = False
    phase = "preparing"
    def check_cancel():
        if cancel_file and Path(cancel_file).is_file():
            raise TrainingCancelled("Cancelación solicitada desde la interfaz local.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "version": version, "category": category}, handle)
        check_cancel()
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        try:
            import numpy as np
            import tensorflow as tf
        except ImportError as exc:
            raise DatasetError("TensorFlow/NumPy no están instalados en este intérprete. Ejecuta python -m pip install -r requirements.txt en el entorno nuevo.") from exc
        check_cancel()
        tf.config.threading.set_inter_op_parallelism_threads(threads)
        tf.config.threading.set_intra_op_parallelism_threads(threads)
        tf.keras.utils.set_random_seed(seed)
        target.mkdir(parents=True, exist_ok=False)
        run_created = True
        write_json(target / "status.json", {"state": "training", "started_at": datetime.now(timezone.utc).isoformat()})
        write_json(target / "dataset-report.json", audit)
        label_index = {label: index for index, label in enumerate(labels)}
        arrays = {name: (np.asarray([row.features for row in rows], dtype=np.float32), np.asarray([label_index[row.label] for row in rows], dtype=np.int32)) for name, rows in parts.items()}
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        options.threading.max_intra_op_parallelism = threads
        def dataset(name, shuffle=False):
            x, y = arrays[name]
            result = tf.data.Dataset.from_tensor_slices((x, y))
            if shuffle:
                result = result.shuffle(len(x), seed=seed, reshuffle_each_iteration=True)
            return result.batch(batch_size).with_options(options)
        network = [{"units": 128, "activation": "relu", "dropout": .2}, {"units": 64, "activation": "relu", "dropout": .2}, {"units": len(labels), "activation": "softmax", "dropout": 0}]
        model = tf.keras.Sequential([tf.keras.Input(shape=(126,))])
        for layer in network:
            model.add(tf.keras.layers.Dense(layer["units"], activation=layer["activation"]))
            if layer["dropout"]:
                model.add(tf.keras.layers.Dropout(layer["dropout"]))
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        checkpoint = target / "best.weights.h5"
        phase = "fitting"
        write_json(target / "status.json", {"state": "training", "phase": phase})
        class CancelCallback(tf.keras.callbacks.Callback):
            def on_train_batch_begin(self, batch, logs=None):
                check_cancel()
            def on_test_batch_begin(self, batch, logs=None):
                check_cancel()
        history = model.fit(dataset("train", True), validation_data=dataset("validation"), shuffle=False, epochs=epochs, callbacks=[
            CancelCallback(),
            tf.keras.callbacks.ModelCheckpoint(str(checkpoint), monitor="val_loss", save_best_only=True, save_weights_only=True),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience),
            tf.keras.callbacks.TerminateOnNaN(),
        ], verbose=2)
        # Evaluate and export the SAME best checkpoint, not the last epoch.
        check_cancel()
        phase = "evaluating"
        model.load_weights(str(checkpoint))
        metrics = {name: {key: float(value) for key, value in model.evaluate(dataset(name), callbacks=[CancelCallback()], return_dict=True, verbose=0).items()} for name in ("validation", "test")}
        probabilities = np.asarray(model(arrays["test"][0], training=False))
        if not np.isfinite(probabilities).all() or any(not np.isfinite(value) for values in metrics.values() for value in values.values()):
            raise DatasetError("El entrenamiento produjo resultados no finitos; no se exportó un paquete válido.")
        predicted = probabilities.argmax(axis=1)
        confusion = [[0 for _ in labels] for _ in labels]
        for truth, guess in zip(arrays["test"][1], predicted):
            confusion[int(truth)][int(guess)] += 1
        recall = {label: confusion[index][index] / sum(confusion[index]) for index, label in enumerate(labels)}
        check_cancel()
        phase = "exporting"
        model.save(str(target / "model.keras"))
        weights = [float(value) for weight in model.get_weights() for value in weight.reshape(-1)]
        bundle = make_bundle(labels, network, weights)
        # Check that encoding float32 preserved every tensor element and prediction.
        # make_bundle enforces the frontend's network, shape and size limits.
        import base64
        decoded = json.loads(bundle)
        flat = np.frombuffer(base64.b64decode(decoded["weightData"]), dtype="<f4")
        rebuilt, offset = [], 0
        for weight in model.get_weights():
            rebuilt.append(flat[offset:offset + weight.size].reshape(weight.shape))
            offset += weight.size
        model.set_weights(rebuilt)
        smoke = np.asarray(model(arrays["test"][0], training=False))
        if not np.allclose(probabilities, smoke, rtol=1e-6, atol=1e-7):
            raise DatasetError("La comprobación de exportación no conserva las predicciones.")
        (target / "model.kai.json").write_bytes(bundle)
        write_json(target / f"labels_{category}.json", labels)
        report = {"category": category, "version": version, "created_at": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(), "tensorflow": tf.__version__, "numpy": np.__version__, "seed": seed, "epochs_run": len(history.history["loss"]), "selected_by": "minimum validation loss; best checkpoint restored", "metrics": metrics, "test_confusion_matrix": confusion, "matrix_labels": labels, "test_recall_per_label": recall, "history": {key: [float(value) for value in values] for key, values in history.history.items()}, "dataset": audit, "limitations": ["La precisión sobre estos archivos no demuestra calidad con personas nuevas.", "Señas dinámicas como J/Z requieren evaluación temporal; esta red clasifica poses estáticas.", "No publica modelos ni modifica el frontend. Requiere revisión y publicación del superadmin."]}
        write_json(target / "training-report.json", report)
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=False)
        (target / "environment.txt").write_text(freeze.stdout if freeze.returncode == 0 else "No se pudo capturar pip freeze.\n", encoding="utf-8")
        hashes = {file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in target.iterdir() if file.is_file() and file.name != "status.json"}
        write_json(target / "status.json", {"state": "complete", "completed_at": datetime.now(timezone.utc).isoformat(), "files_sha256": hashes, "upload_file": "model.kai.json", "publication": "pending_manual_review"})
        return target, report
    except BaseException as exc:
        if run_created:
            (target / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
            write_json(target / "status.json", {"state": "cancelled" if isinstance(exc, TrainingCancelled) else "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", "phase": phase, "error_type": type(exc).__name__, "error": str(exc) or type(exc).__name__, "traceback_file": "traceback.txt", "note": "No subas los archivos de esta ejecución. Usa una versión nueva al reintentar."})
        raise
    finally:
        lock.unlink(missing_ok=True)
