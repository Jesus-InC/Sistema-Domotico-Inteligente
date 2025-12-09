import pickle
import numpy as np
import paho.mqtt.client as mqtt

# CARGAR MODELOS
with open("modelo_usuario.pkl", "rb") as f:
    rf_usuario = pickle.load(f)

with open("scaler_usuario.pkl", "rb") as f:
    sc_usuario = pickle.load(f)

with open("modelo_lluvia.pkl", "rb") as f:
    rf_lluvia = pickle.load(f)

with open("scaler_lluvia.pkl", "rb") as f:
    sc_lluvia = pickle.load(f)

# DATOS DE DEMOSTRACIÓN 
# Usuario NO está en casa
hora = 17
dia = 3
T_int = 18
H_int = 40

# Va a llover
T_ext = 10
H_ext = 88
P_ext = 1009
hum_suelo = 70

# PREDICCIÓN HOME
entrada_home = np.array([[hora, dia, T_int, H_int]])
entrada_home_scaled = sc_usuario.transform(entrada_home)
pred_home = int(rf_usuario.predict(entrada_home_scaled)[0])

print("Predicción ML – Usuario en casa:", pred_home)

# PREDICCIÓN LLUVIA
entrada_rain = np.array([[T_ext, H_ext, P_ext, hum_suelo]])
entrada_rain_scaled = sc_lluvia.transform(entrada_rain)
pred_rain = int(rf_lluvia.predict(entrada_rain_scaled)[0])

print("Predicción ML – Lluvia inminente:", pred_rain)

# ENVÍO POR MQTT
BROKER = "192.168.1.19"
PORT = 1883
TOPIC_HOME = "casa/ml/home"
TOPIC_RAIN = "casa/ml/rain"

client = mqtt.Client()
client.connect(BROKER, PORT, 60)

client.publish(TOPIC_HOME, pred_home, retain=True)
client.publish(TOPIC_RAIN, pred_rain, retain=True)

print("\n*** Valores publicados al broker MQTT ***")
print(f"{TOPIC_HOME} = {pred_home}")
print(f"{TOPIC_RAIN} = {pred_rain}")

client.disconnect()
