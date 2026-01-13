# 📊 Módulos de Machine Learning y Análisis de Datos para Domótica Inteligente

Esta carpeta contiene los scripts encargados de **entrenar modelos de Machine Learning**, **analizar datos históricos** y **enviar predicciones inteligentes a la ESP32** mediante MQTT, complementando el sistema domótico embebido.

Los modelos permiten al sistema tomar decisiones automáticas basadas en **patrones de uso** y **condiciones ambientales**.


## 🧠 Archivos incluidos

- `predicciones_graficas.py`
- `enviar_predicciones_ESP32.py`


## 🤖 `predicciones_graficas.py`

### 📌 Descripción general

Este archivo se encarga de:
- Analizar un **dataset domótico histórico**
- Visualizar patrones y correlaciones
- Entrenar y evaluar **modelos de Machine Learning**
- Guardar los modelos y escaladores finales
- (Opcionalmente) enviar predicciones a la ESP32

Es el **núcleo de análisis y entrenamiento** del sistema inteligente.

---

### 📊 Análisis exploratorio de datos (EDA)

Se realizan múltiples análisis visuales sobre el dataset `dataset_domotica_final.csv`:

- Histogramas de distribución de variables
- Detección de outliers mediante boxplots
- Mapas de correlación (heatmap)
- Relación entre:
  - Temperatura interior vs exterior
  - Humedad del suelo vs activación de la bomba
  - Presencia del usuario según la hora
- Pairplots multivariables

📌 Esto permite comprender el comportamiento del hogar antes de entrenar los modelos.

---

### 🏠 Predicción: Usuario en casa

#### Variables utilizadas
- Hora
- Día de la semana
- Temperatura interior
- Humedad interior

#### Modelos entrenados
- **Regresión Logística**
- **Random Forest (Bosque Aleatorio)**

#### Evaluación
- Matriz de confusión
- Precisión (accuracy)
- Curva ROC y AUC
- Visualización en 2D mediante **PCA**

📌 El modelo Random Forest se selecciona como el más eficiente.

---

### 🌧️ Predicción: Lluvia inminente

#### Variables utilizadas
- Temperatura exterior
- Humedad exterior
- Presión atmosférica
- Humedad del suelo

#### Modelos entrenados
- **Regresión Logística**
- **Random Forest**

#### Evaluación
- Métricas de clasificación
- Curvas ROC
- Visualización con PCA

📌 El modelo Random Forest ofrece mejor desempeño y estabilidad.

---

### 💾 Guardado de modelos

Al finalizar el entrenamiento, se guardan los modelos y escaladores:

```text
modelo_usuario.pkl
scaler_usuario.pkl
modelo_lluvia.pkl
scaler_lluvia.pkl
```

Estos archivos son utilizados posteriormente para generar predicciones sin necesidad de reentrenar los modelos.


## 📡 Envío de predicciones por MQTT (opcional)

El script incluye una sección final donde:

- Se generan predicciones de prueba
- Se publican los resultados vía MQTT a la ESP32

### 📌 Tópicos utilizados
- `casa/ml/home`
- `casa/ml/rain`

---

## 📤 enviar_predicciones_ESP32.py

### 📌 Descripción general

Este archivo se encarga exclusivamente de:

- Cargar los modelos entrenados previamente
- Generar predicciones a partir de datos de entrada
- Enviar dichas predicciones a la ESP32 por MQTT

Es un script ligero de inferencia, ideal para simulación o integración con sistemas externos.

---

### 🔄 Flujo de funcionamiento

1. Carga de modelos y escaladores (`.pkl`)
2. Definición de datos de demostración
3. Escalado de variables
4. Predicción de:
   - Presencia del usuario en casa
   - Lluvia inminente
5. Publicación de resultados vía MQTT

---

### 📡 Comunicación MQTT

- Broker local configurado
- Publicación con `retain=True`
- La ESP32 recibe los valores incluso si se conecta después

### 📌 Tópicos utilizados
- `casa/ml/home` → `0 / 1`
- `casa/ml/rain` → `0 / 1`

Estos valores alimentan la lógica automática y el modo inteligente del firmware.

---

## 🧩 Integración con la ESP32

Las predicciones enviadas influyen directamente en:

- Activación automática del riego
- Decisiones de iluminación
- Ventilación inteligente
- Modo SMART del sistema domótico

La ESP32 actúa como **ejecutor**, mientras que Python funciona como **cerebro analítico**.

---

## 🎯 Objetivo del módulo

- Incorporar inteligencia predictiva al sistema domótico
- Reducir la intervención manual
- Anticipar eventos (lluvia, presencia)
- Servir como base para sistemas más avanzados:
  - IA en la nube
  - Aprendizaje continuo
  - Integración con asistentes virtuales
