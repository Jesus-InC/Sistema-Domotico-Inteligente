# 🏠 Sistema Domótico Inteligente con ESP32

El firmware del **sistema domótico inteligente basado en ESP32** integra control manual, automático y smart, comunicación MQTT, servidor web embebido, sensores ambientales y control de actuadores (foco, bomba y ventilador).

El sistema está diseñado con **FreeRTOS**, separando responsabilidades en tareas independientes para lograr un funcionamiento robusto y escalable.

---

## 📂 Estructura del proyecto

El proyecto está compuesto por los siguientes archivos principales cargados en la ESP32:


A continuación se describe la función de cada uno.

---

## 🔹 main.ino

Archivo principal del proyecto.

### Función:
- Punto de entrada del programa (`setup()` y `loop()`).
- Inicializa:
  - Estados del sistema (modo, actuadores, flags).
  - Hardware (GPIOs, PWM, sensores).
  - Conexión WiFi (modo STA o AP).
  - Servidor web.
  - Cliente MQTT.
- Crea las **tareas FreeRTOS**:
  - `mqttTask`
  - `sensorTask`
  - `logicTask`
- El `loop()` permanece vacío ya que toda la lógica corre en tareas.

📌 **Responsabilidad:** Orquestar la inicialización y arranque del sistema.

---

## 🔹 funciones.h

Archivo de **definiciones globales y prototipos**.

### Contiene:
- Inclusión de librerías (WiFi, MQTT, DHT, NVS, HTTP).
- Definición de:
  - Pines de hardware.
  - Tópicos MQTT.
  - Modos de operación (Manual, Auto, Smart).
  - Setpoints e histéresis.
  - Configuración de PWM.
- Declaración de:
  - Variables globales compartidas.
  - Flags de control y notificaciones.
  - Objetos globales (DHT, MQTT).
- Prototipos de todas las funciones del sistema.

📌 **Responsabilidad:** Centralizar configuraciones, constantes y contratos entre módulos.

---

## 🔹 funciones.cpp

Archivo que implementa la **lógica principal del sistema**.

### Función:
- Implementación de:
  - Lógica de control Manual, Automática y Smart.
  - Lectura de sensores (DHT y humedad de suelo).
  - Control de actuadores (foco, bomba, ventilador).
  - Control PWM del ventilador.
  - Gestión de decisiones con histéresis.
- Manejo de:
  - WiFi (con NVS).
  - MQTT (callback, publicación y suscripción).
  - Envío de notificaciones (Telegram).
- Implementación de las tareas FreeRTOS:
  - `mqttTask`
  - `sensorTask`
  - `logicTask`

📌 **Responsabilidad:** Ejecutar toda la lógica funcional e inteligente del sistema.

---

## 🔹 webserver.h

Archivo de definición del **servidor web embebido**.

### Contiene:
- Prototipos de funciones relacionadas al servidor web.
- Declaraciones necesarias para:
  - Inicializar el servidor.
  - Manejar rutas HTTP.
  - Interacción con la interfaz web.

📌 **Responsabilidad:** Definir la interfaz del módulo web.

---

## 🔹 webserver.cpp

Implementación del **servidor web HTTP** que corre en la ESP32.

### Función:
- Inicializa un servidor web local.
- Expone endpoints para:
  - Cambiar modos de operación.
  - Encender/apagar actuadores.
  - Ajustar velocidad del ventilador.
  - Recibir credenciales WiFi.
- Sirve la página web principal al cliente.

📌 **Responsabilidad:** Permitir el control local del sistema desde un navegador.

---

## 🔹 index_html.h

Archivo que contiene la **interfaz web embebida**.

### Función:
- Almacena el código HTML/CSS/JavaScript como una cadena en memoria.
- Implementa:
  - Botones de control.
  - Sliders (por ejemplo, velocidad o luminosidad).
  - Visualización del estado del sistema.
- Es servido directamente por el servidor web de la ESP32.

📌 **Responsabilidad:** Proveer una interfaz gráfica para el usuario sin depender de servidores externos.