# 🤖 Asistente de Lenguaje Natural para Sistema Domótico (`asistente_pln.py`)

Este archivo implementa un **asistente domótico basado en Procesamiento de Lenguaje Natural (PLN)** que permite controlar el sistema domótico mediante **comandos en lenguaje natural escritos**, utilizando **Machine Learning ligero**, análisis de sentimientos y comunicación **MQTT**.

El asistente actúa como una **capa inteligente externa** que interpreta frases del usuario y las traduce en acciones reales ejecutadas por la ESP32.

---

## 🧠 Tecnologías utilizadas

- **Python 3**
- **NLTK** (tokenización y análisis de sentimiento)
- **Scikit-learn**
  - `CountVectorizer`
  - `LogisticRegression`
- **MQTT (paho-mqtt)**
- **Expresiones regulares (regex)**

---

## 📌 Función general del archivo

- Recibe frases escritas por el usuario.
- Interpreta la **intención semántica** del texto.
- Detecta **emociones o contexto ambiental** (calor, frío, oscuridad).
- Publica comandos MQTT hacia la ESP32.
- Permite simular entradas de un **modelo inteligente (ML)** como:
  - Presencia en casa
  - Lluvia