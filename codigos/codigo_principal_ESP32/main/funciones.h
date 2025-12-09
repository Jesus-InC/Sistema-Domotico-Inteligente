#ifndef FUNCIONES_H
#define FUNCIONES_H

#include <WiFi.h>
#include <PubSubClient.h>
#include <Arduino.h>
#include <DHT.h>
#include <Preferences.h>

// ===== TOPICS MQTT =====
#define TOPIC_MODE_CMD     "casa/mode/cmd"
#define TOPIC_MODE_STATUS  "casa/mode/status"

#define TOPIC_FOCO_CMD     "casa/foco/cmd"
#define TOPIC_FOCO_STATUS  "casa/foco/status"

#define TOPIC_BOMBA_CMD    "casa/bomba/cmd"
#define TOPIC_BOMBA_STATUS "casa/bomba/status"

#define TOPIC_VENT_CMD     "casa/vent/cmd"
#define TOPIC_VENT_STATUS  "casa/vent/status"
#define TOPIC_VENT_VEL     "casa/vent/vel"

#define TOPIC_ML_HOME      "casa/ml/home"
#define TOPIC_ML_RAIN      "casa/ml/rain"

// ===== MODOS =====
#define MODE_MANUAL 0
#define MODE_AUTO   1
#define MODE_SMART  2

extern volatile uint8_t modoActual;

// ========= WIFI =========
// SIN valores por defecto → primera vez estará vacío
#define WIFI_SSID_DEFAULT   ""
#define WIFI_PASSWORD_DEFAULT ""

// ========= MQTT =========
#define MQTT_BROKER     "192.168.1.19"
#define MQTT_PORT       1883
#define MQTT_CLIENT_ID  "ESP32_SMARTHOME"

// ========= HARDWARE =========
#define PIN_FOCO   17
#define PIN_BOMBA  19
#define PIN_VENT   18
#define DHT_PIN     4
#define SOIL_PIN   34

// ========= PWM =========
#define VENT_PWM_FREQ      5000
#define VENT_PWM_RES       8
#define VENT_PWM_CHANNEL   0
#define VENT_PWM_DEFAULT   200

// ========= SETPOINTS =========
#define TEMP_SETPOINT      25
#define TEMP_HISTERESIS     2
#define SOIL_SETPOINT      30
#define SOIL_HISTERESIS     5

// ========= VARIABLES =========
extern volatile bool focoEstado;
extern volatile bool bombaEstado;

extern bool ventEncendido;
extern volatile int ventVelocidad;
extern volatile int ventUltimaVelocidad;

extern float tempActual;
extern float humActual;
extern int sueloRaw;
extern int sueloPorc;

// ML
extern volatile uint8_t ml_home;
extern volatile uint8_t ml_rain;

extern String wifi_ssid_nvs;
extern String wifi_pass_nvs;

// ========= SUELO =========
#define SOIL_RAW_SECO    4000
#define SOIL_RAW_HUMEDO  2700

// ========= TIEMPOS =========
#define DHT_PERIOD_MS        2000
#define SOIL_PERIOD_MS       1000
#define SENSOR_LOG_PERIOD_MS 30000

// ========= CONFIG =========
#define BAUDRATE       115200
#define RECONNECT_MS   2000

#define STACK_SIZE     4096
#define PRIORITY_MQTT  1
#define PRIORITY_LOGIC 1
#define PRIORITY_SENS  1

// ========= PROTOTIPOS =========
void mqttTask(void *parameter);
void logicTask(void *parameter);
void sensorTask(void *parameter);

void mqttCallback(char* topic, byte* message, unsigned int length);
void aplicarPWM();

void conectar_wifi();
void cargarWiFiDesdeNVS();
void guardarWiFiEnNVS(const char* ssid, const char* pass);
void borrarWiFiEnNVS();

extern PubSubClient client;

#endif
