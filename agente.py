import csv
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai

MODELO = "gemini-3.5-flash-lite"
BASE = Path(__file__).parent
GASTOS_CSV = BASE / "Conocimientos" / "gastos.csv"
MAX_RONDAS = 5

HERRAMIENTAS = [{
    "type": "function",
    "name": "consultar_gastos",
    "description": (
        "Consulta los gastos registrados en gastos.csv. Filtra por los campos dados "
        "y devuelve la suma, el conteo y (si listar es true) las filas. Úsala para "
        "cualquier total, conteo o detalle de gastos; nunca sumes a mano."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "empleado": {"type": "string", "description": "Nombre del empleado, ej. Daniela Osorio"},
            "area": {"type": "string", "description": "Área, ej. Ventas"},
            "ciudad": {"type": "string", "description": "Ciudad, ej. Cartagena"},
            "categoria": {"type": "string", "description": "Categoría, ej. Alimentación, Hospedaje"},
            "estado": {"type": "string", "description": "Aprobado, Pendiente o Rechazado"},
            "fecha_desde": {"type": "string", "description": "AAAA-MM-DD, inclusive"},
            "fecha_hasta": {"type": "string", "description": "AAAA-MM-DD, inclusive"},
            "listar": {"type": "boolean", "description": "true para incluir las filas (máx. 50)"},
        },
    },
}]


def consultar_gastos(empleado=None, area=None, ciudad=None, categoria=None,
                     estado=None, fecha_desde=None, fecha_hasta=None, listar=False):
    with open(GASTOS_CSV, encoding="utf-8", newline="") as f:
        filas = list(csv.DictReader(f))
    igual = {"empleado": empleado, "area": area, "ciudad": ciudad,
             "categoria": categoria, "estado": estado}
    for campo, valor in igual.items():
        if valor:
            filas = [x for x in filas if valor.lower() in x[campo].lower()]
    if fecha_desde:
        filas = [x for x in filas if x["fecha"] >= fecha_desde]
    if fecha_hasta:
        filas = [x for x in filas if x["fecha"] <= fecha_hasta]
    resultado = {"conteo": len(filas), "suma_cop": sum(int(x["valor_cop"]) for x in filas)}
    if listar:
        resultado["filas"] = filas[:50]
    return resultado


def cargar_instrucciones():
    instrucciones = (BASE / "prompts" / "sistema.md").read_text(encoding="utf-8")
    documentos = [
        f'<documento nombre="{archivo.name}">\n{archivo.read_text(encoding="utf-8")}\n</documento>'
        for archivo in sorted((BASE / "Conocimientos").iterdir())
        if archivo.is_file() and not archivo.name.startswith(".")
        and archivo != GASTOS_CSV  # se consulta con la herramienta consultar_gastos
    ]
    return "\n\n".join([instrucciones, *documentos])


def ejecutar(llamada):
    try:
        resultado, error = consultar_gastos(**llamada.arguments), False
    except Exception as e:  # el modelo recibe el error y puede corregir los argumentos
        resultado, error = {"error": str(e)}, True
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
