import sys

from dotenv import load_dotenv
from google import genai

MODELO = "gemini-3.8-flash"
INSTRUCCIONES = "Responde siempre en español, en un máximo de 3 frases."


def main():
    if len(sys.argv) < 2:
        sys.exit('Uso: python agente.py "tu pregunta"')
    pregunta = " ".join(sys.argv[1:])

    load_dotenv()  # carga GEMINI_API_KEY desde .env
    client = genai.Client()  # lee GEMINI_API_KEY del entorno

    interaction = client.interactions.create(
        model=MODELO,
        input=pregunta,
        system_instruction=INSTRUCCIONES,
    )
    print(interaction.output_text)


if __name__ == "__main__":
    main()
