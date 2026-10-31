"""Dataset validation uses only Python's standard library; no training imports."""
import hashlib
import json
import math
import random
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

NUMBERS = tuple(str(value) for value in range(1, 11))
ALPHABET = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
LABELS = {
    "vocales": tuple("AEIOU") + ("Nulo",),
    "abecedario": tuple(c for c in ALPHABET if c not in "AEIOUJZ") + ("Nulo",),
    "numeros": NUMBERS + ("Nulo",),
    "matematicas": NUMBERS + ("Suma", "Resta", "Igual", "Nulo"),
}
ALLOWED = {**LABELS, "abecedario": ALPHABET + ("Nulo",)}
MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_SAMPLES = 100_000


class DatasetError(ValueError):
    pass


@dataclass(frozen=True)
class Sample:
    label: str
    features: tuple
    group: str


def read_json(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
        raise DatasetError(f"Archivo ausente o mayor de 20 MiB: {path}")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeError, ValueError) as exc:
        raise DatasetError(f"JSON inválido: {path}") from exc
    return value, hashlib.sha256(raw).hexdigest()


def features_from_landmarks(landmarks):
    if not isinstance(landmarks, list) or len(landmarks) not in (21, 42):
        raise DatasetError("Cada muestra debe tener 21 o 42 landmarks.")
    values = []
    for point in landmarks:
        if not isinstance(point, dict):
            raise DatasetError("Landmark inválido; se esperan x, y, z.")
        for axis in ("x", "y", "z"):
            value = point.get(axis)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > 10:
                raise DatasetError("Las coordenadas deben ser finitas y estar entre -10 y 10.")
            values.append(float(value))
    # The capture page already subtracts the FIRST wrist's x/y, including both hands.
    # Do not normalize again, subtract z, reorder hands, or scale these coordinates.
    if abs(values[0]) > 1e-5 or abs(values[1]) > 1e-5:
        raise DatasetError("Las coordenadas no están normalizadas como en la web: la primera x/y debe ser cero.")
    values.extend([0.0] * (126 - len(values)))
    return tuple(values)


def choose_labels(category, requested=None):
    if category not in LABELS:
        raise DatasetError("Módulo no reconocido.")
    labels = list(LABELS[category] if requested is None else requested)
    if len(labels) < 2 or len(set(labels)) != len(labels) or "Nulo" not in labels or any(label not in ALLOWED[category] for label in labels):
        raise DatasetError("Las etiquetas deben ser únicas, pertenecer al módulo e incluir Nulo y al menos una seña.")
    if any(label in ("J", "Z") for label in labels):
        raise DatasetError("J y Z requieren movimiento y evaluación temporal. Esta red de poses estáticas todavía no puede entrenarlas correctamente.")
    return sorted(labels)


def load_samples(paths, category, labels):
    if not 1 <= len(paths) <= 100:
        raise DatasetError("Selecciona entre 1 y 100 archivos de capturas.")
    samples, fingerprints, files = [], {}, []
    duplicates = excluded = raw_count = duplicate_files = 0
    seen_files = set()
    for path in paths:
        payload, digest = read_json(path)
        if digest in seen_files:
            duplicate_files += 1
            continue
        seen_files.add(digest)
        if isinstance(payload, dict):
            origin = payload.get("category")
            accepted = {category}
            if category == "matematicas":
                accepted.add("numeros")
            if category == "abecedario":
                accepted.add("vocales")
            if origin not in accepted:
                raise DatasetError(f"El dataset {path} pertenece a otro módulo: {origin}.")
            rows = payload.get("data")
        else:
            rows = payload  # Legacy training_data_*.json: still checked and deduplicated.
        if not isinstance(rows, list) or not rows:
            raise DatasetError(f"El dataset {path} no contiene una lista de muestras.")
        raw_count += len(rows)
        if raw_count > MAX_SAMPLES:
            raise DatasetError("Máximo 100 000 muestras por entrenamiento.")
        files.append({"path": str(Path(path).resolve()), "sha256": digest, "rows": len(rows)})
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or row.get("label") not in ALLOWED[category]:
                raise DatasetError(f"Etiqueta inválida en {path}, muestra {index + 1}.")
            features = features_from_landmarks(row.get("landmarks"))
            # Float32 and 6 decimals avoid pretending tiny numerical jitter is new data.
            fingerprint = hashlib.sha256(struct.pack("<126f", *(round(value, 6) or 0.0 for value in features))).digest()
            if fingerprint in fingerprints:
                if fingerprints[fingerprint] != row["label"]:
                    raise DatasetError(f"La misma pose tiene etiquetas contradictorias: {fingerprints[fingerprint]} / {row['label']}.")
                duplicates += 1
                continue
            fingerprints[fingerprint] = row["label"]
            if row["label"] not in labels:
                excluded += 1
                continue
            samples.append(Sample(row["label"], features, digest))
    if not files:
        raise DatasetError("Selecciona al menos un dataset.")
    return samples, {"files": files, "raw_samples": raw_count, "duplicates_removed": duplicates, "duplicate_files_skipped": duplicate_files, "excluded_labels_samples": excluded}


def split_groups(samples, labels, seed=42):
    """Keep each original file wholly in train, validation OR test."""
    groups = sorted({sample.group for sample in samples})
    if len(groups) < 3:
        raise DatasetError("Necesitas al menos tres archivos de capturas independientes.")
    count = max(1, round(len(groups) * .2))
    if count * 2 >= len(groups):
        count = 1
    rng = random.Random(seed)
    grouped = {group: Counter() for group in groups}
    for sample in samples:
        grouped[sample.group][sample.label] += 1
    # Separate operator/number captures may need 2 files in EACH holdout, not 1.
    for holdout_size in range(count, len(groups) // 3 + 1):
        for _ in range(500):
            shuffled = groups.copy()
            rng.shuffle(shuffled)
            assignments = {group: "test" if index < holdout_size else "validation" if index < holdout_size * 2 else "train" for index, group in enumerate(shuffled)}
            totals = {name: Counter() for name in ("train", "validation", "test")}
            for group, counts in grouped.items():
                totals[assignments[group]].update(counts)
            if all(set(counts) == set(labels) and min(counts.values()) >= 3 for counts in totals.values()):
                parts = {name: [] for name in totals}
                for sample in samples:
                    parts[assignments[sample.group]].append(sample)
                return parts
    raise DatasetError("No se puede separar por archivos manteniendo al menos tres muestras únicas de cada etiqueta en los tres conjuntos. Captura más sesiones que incluyan cada etiqueta.")


def prepare(paths, category, requested=None, seed=42, min_samples=20):
    labels = choose_labels(category, requested)
    samples, report = load_samples(paths, category, labels)
    counts = Counter(sample.label for sample in samples)
    groups = defaultdict(set)
    hands = Counter()
    for sample in samples:
        groups[sample.label].add(sample.group)
        hands["two_hands" if any(sample.features[63:]) else "one_hand"] += 1
    problems = []
    for label in labels:
        if counts[label] < min_samples:
            problems.append(f"{label}: {counts[label]} muestras únicas; mínimo {min_samples}.")
        if len(groups[label]) < 3:
            problems.append(f"{label}: aparece en {len(groups[label])} archivos; se requieren al menos tres capturas independientes.")
        if label in ("Suma", "Resta", "Igual") and not any(sample.label == label and any(sample.features[63:]) for sample in samples):
            problems.append(f"{label}: necesita ejemplos de dos manos, como exige su referencia en la web.")
    parts = None
    if not problems:
        try:
            parts = split_groups(samples, labels, seed)
        except DatasetError as exc:
            problems.append(str(exc))
    report.update({"category": category, "labels": labels, "unique_samples": len(samples), "samples_per_label": dict(sorted(counts.items())), "files_per_label": {label: len(groups[label]) for label in labels}, "hand_coverage": dict(hands), "problems": problems, "ready_to_train": not problems, "seed": seed,
                   "evaluation_note": "Separación por archivo. No demuestra generalización entre personas si los archivos proceden de la misma persona."})
    if parts:
        report["split"] = {name: {"samples": len(part), "groups": sorted({sample.group for sample in part}), "per_label": dict(Counter(sample.label for sample in part))} for name, part in parts.items()}
    return labels, parts, report
