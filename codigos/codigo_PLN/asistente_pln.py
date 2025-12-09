import re
import time
import nltk
import paho.mqtt.client as mqtt
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# ===============================================================
# DESCARGAS NLTK
# ===============================================================
nltk.download("punkt")
nltk.download("vader_lexicon")

sia = SentimentIntensityAnalyzer()

# ===============================================================
# 1. DATASET DE ENTRENAMIENTO (tú lo puedes ampliar)
# ===============================================================
frases = [
    "enciende el foco", "prende la luz", "activar luz",
    "apaga el foco", "apagar luz", "quita la luz",

    "enciende la bomba", "prende la bomba de agua",
    "apaga la bomba", "desactiva el riego",

    "enciende el ventilador", "prende el ventilador",
    "apaga el ventilador", "ventilador off",

    "cambia a modo manual", "pon modo automático", "modo inteligente",
    "activar modo smart", "modo auto",

    "ventilador a 120", "pon el ventilador a 200", "sube velocidad del ventilador",

    "estoy en casa", "no estoy en casa",
    "va a llover", "no va a llover",

    "que calor", "hace calor",
    "tengo frío", "que frío",
    "que oscuro", "está muy oscuro"
]

intenciones = [
    0,0,0,      # encender foco
    1,1,1,      # apagar foco

    2,2,        # encender bomba
    3,3,        # apagar bomba

    4,4,        # encender ventilador
    5,5,        # apagar ventilador

    6,6,6,6,6,  # modos

    7,7,7,      # velocidades

    8,8,        # usuario en casa
    9,9,        # lluvia

    10,10,      # calor
    11,11,      # frío
    12,12       # oscuro
]

# ===============================================================
# 2. ENTRENAR MODELO DE REGRESIÓN LOGÍSTICA
# ===============================================================
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(frases)

clf = LogisticRegression()
clf.fit(X, intenciones)

# ===============================================================
# MQTT CONFIG
# ===============================================================
BROKER = "192.168.1.19"
PORT = 1883

TOPIC_FOCO = "casa/foco/cmd"
TOPIC_BOMBA = "casa/bomba/cmd"
TOPIC_VENT  = "casa/vent/cmd"
TOPIC_VEL   = "casa/vent/vel"
TOPIC_MODE  = "casa/mode/cmd"
TOPIC_HOME  = "casa/ml/home"
TOPIC_RAIN  = "casa/ml/rain"

client = mqtt.Client()
client.connect(BROKER, PORT, 60)

def publicar(topic, msg):
    print(f"[MQTT] {topic} → {msg}")
    client.publish(topic, msg)

# ===============================================================
# NLP: PROCESAR FRASE Y ACTUAR
# ===============================================================

# extraer velocidad
def extraer_numero(texto):
    m = re.search(r"\d+", texto)
    return int(m.group(0)) if m else None

def procesar(texto):
    t = texto.lower().strip()

    # 1️⃣ Detección de emoción automática antes del modelo
    sentimiento = sia.polarity_scores(t)
    if "calor" in t:
        return 10
    if "oscuro" in t:
        return 12
    if "frio" in t or "frío" in t:
        return 11

    # 2️⃣ Clasificación ML
    x = vectorizer.transform([t])
    intent = clf.predict(x)[0]
    return intent

# ===============================================================
#  ACCIONES SEGÚN INTENCIÓN
# ===============================================================
def ejecutar_intencion(intent, texto):
    # Foco
    if intent == 0: publicar(TOPIC_FOCO, "ON"); return "Foco encendido"
    if intent == 1: publicar(TOPIC_FOCO, "OFF"); return "Foco apagado"

    # Bomba
    if intent == 2: publicar(TOPIC_BOMBA, "ON"); return "Bomba encendida"
    if intent == 3: publicar(TOPIC_BOMBA, "OFF"); return "Bomba apagada"

    # Ventilador
    if intent == 4: publicar(TOPIC_VENT, "ON"); return "Ventilador encendido"
    if intent == 5: publicar(TOPIC_VENT, "OFF"); return "Ventilador apagado"

    # Modo
    if intent == 6:
        if "manual" in texto: publicar(TOPIC_MODE, "manual"); return "Modo manual"
        if "auto" in texto: publicar(TOPIC_MODE, "auto"); return "Modo auto"
        publicar(TOPIC_MODE, "smart"); return "Modo inteligente"

    # Velocidad
    if intent == 7:
        vel = extraer_numero(texto)
        if vel is None:
            return "No entendí la velocidad"
        vel = max(0, min(255, vel))
        publicar(TOPIC_VEL, vel)
        publicar(TOPIC_VENT, "ON")
        return f"Ventilador ajustado a {vel}"

    # ML manual (para pruebas)
    if intent == 8: publicar(TOPIC_HOME, "1"); return "ML: usuario en casa"
    if intent == 9: publicar(TOPIC_RAIN, "1"); return "ML: va a llover"

    # Emociones
    if intent == 10: publicar(TOPIC_VENT, "ON"); return "Hace calor → ventilador ON"
    if intent == 11: publicar(TOPIC_VENT, "OFF"); return "Hace frío → ventilador OFF"
    if intent == 12: publicar(TOPIC_FOCO, "ON"); return "Oscuridad → foco ON"

    return "No pude interpretar la intención"

# ===============================================================
# MAIN LOOP — TEXTO (SIN AUDIO)
# ===============================================================
def main():
    print("\n=== NLP DOMÓTICO (texto) ===")
    print("Escribe frases como:")
    print(" • enciende el foco")
    print(" • apaga la bomba")
    print(" • ventilador a 150")
    print(" • cambia a modo inteligente")
    print(" • que calor")
    print("Escribe 'salir' para terminar.\n")

    while True:
        frase = input("Tú: ").strip()
        if frase.lower() == "salir":
            break

        intent = procesar(frase)
        resp = ejecutar_intencion(intent, frase)
        print("✔", resp, "\n")

if __name__ == "__main__":
    main()
