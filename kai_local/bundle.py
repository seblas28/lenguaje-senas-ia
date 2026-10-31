"""The same declarative Kai v1 contract validated by the web and Supabase."""
import base64
import json
import math
import struct

MAX_BYTES = 2 * 1024 * 1024


def make_bundle(labels, network, values):
    if len(labels) < 2 or "Nulo" not in labels or len(set(labels)) != len(labels):
        raise ValueError("Las etiquetas deben ser únicas e incluir Nulo.")
    previous, count = 126, 0
    if not 1 <= len(network) <= 6:
        raise ValueError("Arquitectura no compatible.")
    for index, layer in enumerate(network):
        last = index == len(network) - 1
        units = layer["units"]
        if type(units) is not int or not 1 <= units <= 256 or layer["activation"] != ("softmax" if last else "relu") or not 0 <= layer["dropout"] <= .8 or last and (units != len(labels) or layer["dropout"] != 0):
            raise ValueError("Arquitectura no compatible.")
        count += (previous + 1) * units
        previous = units
    weights = list(values)
    if len(weights) != count or any(not math.isfinite(float(value)) for value in weights):
        raise ValueError("Pesos incompletos o no finitos.")
    binary = struct.pack(f"<{count}f", *weights)
    if any(not math.isfinite(value) for value in struct.unpack(f"<{count}f", binary)):
        raise ValueError("Los pesos no caben en float32.")
    bundle = {"schemaVersion": 1, "inputFeatures": 126, "labels": labels, "network": network, "weightData": base64.b64encode(binary).decode("ascii")}
    encoded = json.dumps(bundle, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    # Leave room for JSONB's spaces in the server's 2 MiB check.
    if len(encoded) > MAX_BYTES - 4096:
        raise ValueError("El paquete supera el límite de 2 MiB.")
    return encoded
