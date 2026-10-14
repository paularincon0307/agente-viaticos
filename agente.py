import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai

MODELO = "gemini-3.5-flash-lite"
INSTRUCCIONES = "Responde como un cavernicola en español, en un máximo de 3 frases."


def cargar_instrucciones():
    base = Path(__file__).parent
    instrucciones = (base / "prompts" / "sistema.md").read_text(encoding="utf-8")
    documentos = [
        f'<documento nombre="{archivo.name}">\n{archivo.read_text(encoding="utf-8")}\n</documento>'
        for archivo in sorted((base / "Conocimientos").iterdir())
        if archivo.is_file() and not archivo.name.startswith(".")
    ]
    return "\n\n".join([instrucciones, *documentos])


def main():
    if len(sys.argv) < 2:
        sys.exit('Uso: python agente.py "tu pregunta"')
    pregunta = " ".join(sys.argv[1:])

    load_dotenv()  # carga GEMINI_API_KEY desde .env
    client = genai.Client()  # lee GEMINI_API_KEY del entorno

    interaction = client.interactions.create(
        model=MODELO,
        input=pregunta,
        system_instruction=cargar_instrucciones(),
    )
    print(interaction.output_text)


if __name__ == "__main__":
    main()
