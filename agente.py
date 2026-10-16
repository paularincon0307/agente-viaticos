import csv
import json
import sys
import unicodedata
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from google import genai

MODELO = "gemini-3.5-flash-lite"
BASE = Path(__file__).parent
GASTOS_CSV = BASE / "Conocimientos" / "gastos.csv"
MAX_RONDAS = 5

SOLICITUDES_CSV = BASE / "solicitudes.csv"
CAMPOS_SOLICITUD = ["radicado", "fecha_registro", "empleado", "ciudad",
                    "fecha_salida", "noches", "motivo", "estado"]

HERRAMIENTAS = [
    {
        "type": "function",
        "name": "sumar_gastos",
        "description": (
            "Cuenta y suma los gastos registrados en gastos.csv, con filtros opcionales "
            "por área y por estado. Úsala siempre que pregunten cuántos gastos hay o "
            "cuánto suman; nunca cuentes ni sumes a mano. Devuelve el número de filas "
            "y el total en COP."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "area": {"type": "string", "description": "Área, ej. Ventas. Omítelo para no filtrar."},
                "estado": {"type": "string", "description": "Aprobado, Pendiente o Rechazado. Omítelo para no filtrar."},
            },
        },
    },
    {
        "type": "function",
        "name": "registrar_solicitud",
        "description": (
            "Registra una solicitud de viaje en solicitudes.csv y devuelve su radicado. "
            "Úsala solo cuando el usuario pida registrar una solicitud y ya tengas los "
            "cinco datos; si falta alguno, pídelo antes de llamarla."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "empleado": {"type": "string", "description": "Nombre del empleado"},
                "ciudad": {"type": "string", "description": "Ciudad de destino"},
                "fecha_salida": {"type": "string", "description": "AAAA-MM-DD"},
                "noches": {"type": "integer", "description": "Número de noches"},
                "motivo": {"type": "string", "description": "Motivo del viaje"},
            },
            "required": ["empleado", "ciudad", "fecha_salida", "noches", "motivo"],
        },
    },
]


def normalizar(texto):
    """Minúsculas y sin tildes, para comparar sin distinguir ambos."""
    sin_tildes = unicodedata.normalize("NFD", str(texto))
    return "".join(c for c in sin_tildes if unicodedata.category(c) != "Mn").lower()


def sumar_gastos(area=None, estado=None):
    with open(GASTOS_CSV, encoding="utf-8", newline="") as f:
        filas = list(csv.DictReader(f))
    if area:
        filas = [x for x in filas if normalizar(x["area"]) == normalizar(area)]
    if estado:
        filas = [x for x in filas if normalizar(x["estado"]) == normalizar(estado)]
    return {"filas": len(filas), "total_cop": sum(int(x["valor_cop"]) for x in filas)}


def registrar_solicitud(empleado, ciudad, fecha_salida, noches, motivo):
    existe = SOLICITUDES_CSV.exists()
    consecutivo = 1
    if existe:
        with open(SOLICITUDES_CSV, encoding="utf-8", newline="") as f:
            consecutivo = sum(1 for _ in csv.DictReader(f)) + 1
    radicado = f"S-{consecutivo:03d}"
    with open(SOLICITUDES_CSV, "a", encoding="utf-8", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=CAMPOS_SOLICITUD)
        if not existe:
            escritor.writeheader()
        escritor.writerow({
            "radicado": radicado, "fecha_registro": date.today().isoformat(),
            "empleado": empleado, "ciudad": ciudad, "fecha_salida": fecha_salida,
            "noches": noches, "motivo": motivo,
            "estado": "Pendiente de aprobación del jefe inmediato",
        })
    return {"radicado": radicado}


FUNCIONES = {"sumar_gastos": sumar_gastos, "registrar_solicitud": registrar_solicitud}


def cargar_instrucciones():
    instrucciones = (BASE / "prompts" / "sistema.md").read_text(encoding="utf-8")
    documentos = [
        f'<documento nombre="{archivo.name}">\n{archivo.read_text(encoding="utf-8")}\n</documento>'
        for archivo in sorted((BASE / "Conocimientos").iterdir())
        if archivo.is_file() and not archivo.name.startswith(".")
        and archivo != GASTOS_CSV  # se consulta con la herramienta sumar_gastos
    ]
    return "\n\n".join([instrucciones, *documentos])


def ejecutar(llamada):
    try:
        resultado, error = FUNCIONES[llamada.name](**llamada.arguments), False
    except Exception as e:  # el modelo recibe el error y puede corregir los argumentos
        resultado, error = {"error": str(e)}, True
    print(f"[herramienta] {llamada.name}({llamada.arguments}) -> {resultado}")
    return {"type": "function_result", "call_id": llamada.id, "name": llamada.name,
            "result": json.dumps(resultado, ensure_ascii=False), "is_error": error}


def main():
    if len(sys.argv) < 2:
        sys.exit('Uso: python agente.py "tu pregunta"')
    pregunta = " ".join(sys.argv[1:])

    load_dotenv()  # carga GEMINI_API_KEY desde .env
    client = genai.Client()  # lee GEMINI_API_KEY del entorno
    instrucciones = cargar_instrucciones()

    entrada, previa = pregunta, None
    for _ in range(MAX_RONDAS):
        interaction = client.interactions.create(
            model=MODELO,
            input=entrada,
            system_instruction=instrucciones,
            tools=HERRAMIENTAS,
            previous_interaction_id=previa,
        )
        llamadas = [s for s in interaction.steps if s.type == "function_call"]
        if not llamadas:
            break
        entrada, previa = [ejecutar(l) for l in llamadas], interaction.id
    print(interaction.output_text)


if __name__ == "__main__":
    main()
