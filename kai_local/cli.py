import argparse
import json
import sys
import traceback

from .data import LABELS, DatasetError, prepare


def main(argv=None):
    parser = argparse.ArgumentParser(description="Kai Señas: auditar datasets y entrenar localmente, sin servidor ni publicación automática.")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("audit", "train"):
        action = sub.add_parser(command, help="Revisar capturas sin TensorFlow" if command == "audit" else "Entrenar y exportar un paquete Kai v1")
        action.add_argument("--category", required=True, choices=sorted(LABELS))
        action.add_argument("--data", required=True, nargs="+", help="Rutas de datasets JSON descargados de la web; cada archivo es una captura independiente")
        action.add_argument("--labels", nargs="+", help="Etiquetas a entrenar; incluye Nulo. Si se omite, se espera el módulo completo (consonantes estáticas en abecedario, sin J/Z).")
        action.add_argument("--seed", type=int, default=42)
        if command == "train":
            action.add_argument("--version", required=True)
            action.add_argument("--output", default="artifacts")
            action.add_argument("--epochs", type=int, default=50)
            action.add_argument("--batch-size", type=int, default=16)
            action.add_argument("--patience", type=int, default=8)
            action.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        labels, parts, report = prepare(args.data, args.category, args.labels, args.seed)
        if args.command == "audit":
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report["ready_to_train"] else 2
        if report["problems"]:
            raise DatasetError("El dataset no está listo:\n- " + "\n- ".join(report["problems"]))
        from .training import train
        target, result = train(args.category, args.version, labels, parts, report, args.output, args.epochs, args.batch_size, args.patience, args.seed, args.threads)
        print(f"\nEntrenamiento completado: {target}")
        print(f"Exactitud de test: {result['metrics']['test']['accuracy']:.2%}. Revisa también la matriz y el rendimiento por etiqueta.")
        print("Sube únicamente model.kai.json desde Administración → Modelos de IA. Aún requiere aprobación y publicación del superadmin.")
        return 0
    except (DatasetError, OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        traceback.print_exc()
        print("Se recibió KeyboardInterrupt durante la ejecución. El origen no se puede determinar solo con esta excepción. No se publicó ningún modelo. Revisa status.json y traceback.txt si se creó el directorio de resultados.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
