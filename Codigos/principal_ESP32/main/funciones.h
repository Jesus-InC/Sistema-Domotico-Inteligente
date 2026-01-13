#ifndef FUNCIONES_H
#define FUNCIONES_H

#include <WiFi.h>
#include <PubSubClient.h>
#include <Arduino.h>
#include <DHT.h>
#include <Preferences.h>
#include <HTTPClient.h>

// ======================================================
//                TELEGRAM (HARDCODE)
// ======================================================
// Ya con tus datos reales:
#define TG_BOT_TOKEN   "8565394779:AAFmbz6J74HF0vUDsxp50SfPa4aesRFflU4"
#define TG_CHAT_ID     "7520466083"

// Función global para enviar mensajes a Telegram
bool tg_send(const String &msg);

// ======================================================
//                TOPICS MQTT
// ======================================================
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


// ======================================================
//                     MODOS
// ======================================================
#define MODE_MANUAL 0
#define MODE_AUTO   1
#define MODE_SMART  2

extern volatile uint8_t modoActual;

// DHT global (definido en funciones.cpp)
extern DHT dht;


// ======================================================
//                WIFI (sin valores por defecto)
// ======================================================
#define WIFI_SSID_DEFAULT     ""
#define WIFI_PASSWORD_DEFAULT ""


// ======================================================
//                     MQTT
// ======================================================
#define MQTT_BROKER     "10.171.220.174"
#define MQTT_PORT       1883
#define MQTT_CLIENT_ID  "ESP32_SMARTHOME"


// ======================================================
//                     HARDWARE
// ======================================================
#define PIN_FOCO   17          // Puede quedar como indicador / LED
#define PIN_BOMBA  19
#define PIN_VENT   18
#define DHT_PIN     4
#define SOIL_PIN   34

// Pines específicos del DIMMER del foco
#define PIN_FOCO_TRIAC 25      // GPIO25 → MOC3021 (disparo TRIAC)
#define PIN_FOCO_ZC    27      // GPIO27 → H11AA1 (cruce por cero)


// ======================================================
//                     PWM VENTILADOR
// ======================================================
#define VENT_PWM_FREQ      5000
#define VENT_PWM_RES       8
#define VENT_PWM_CHANNEL   0
#define VENT_PWM_DEFAULT   200


// ======================================================
//                   DIMMER FOCO
// ======================================================
// 60 Hz → periodo ≈ 16.666 ms → 360° → ~46.3 µs por grado eléctrico
#define FOCO_TGE_US         46      // Periodo del timer en µs (1° eléctrico aprox.)
#define FOCO_LEVEL_DEFAULT  255     // Nivel por defecto al encender desde 0 (máxima luz)
#define FOCO_LEVEL_MIN_DEG  28      // Mínimo ángulo (máxima luz)
#define FOCO_LEVEL_MAX_DEG  179     // Máximo ángulo (mínima luz)


// ======================================================
//                   SETPOINTS
// ======================================================
#define TEMP_SETPOINT      25
#define TEMP_HISTERESIS     2
#define SOIL_SETPOINT      30
#define SOIL_HISTERESIS     5


// ======================================================
//                 VARIABLES GLOBALES
// ======================================================

// Actuadores / estado lógico
extern volatile bool focoEstado;       // ON/OFF lógico del foco (dimmer)
extern volatile bool bombaEstado;

extern bool        ventEncendido;
extern volatile int ventVelocidad;
extern volatile int ventUltimaVelocidad;

// DIMMER foco
extern volatile int focoNivel;          // 0–255 (0 = apagado)
extern volatile int focoGradosObjetivo; // 28–179, -1 = sin disparo
extern volatile int focoGE;             // contador de grado eléctrico

// Sensores
extern float tempActual;
extern float humActual;
extern int   sueloRaw;
extern int   sueloPorc;

// ML (entradas de tu modelo)
extern volatile uint8_t ml_home;   // 1 = hay gente en casa, 0 = no
extern volatile uint8_t ml_rain;   // 1 = lluvia, 0 = no lluvia

// Para detectar cambios en las salidas del modelo ML
extern uint8_t prev_ml_home;
extern uint8_t prev_ml_rain;

// WIFI almacenado en NVS
extern String wifi_ssid_nvs;
extern String wifi_pass_nvs;


// ======================================================
//             FLAGS PARA NOTIFICACIONES / LÓGICA
// ======================================================
// Convención:
// 0 = sin decisión / ninguno
// 1 = activado (ON)
// 2 = desactivado (OFF)

// ---- MODO AUTO ----
extern int  autoPrevVentDecision;    // 0=ninguno, 1=ON, 2=OFF
extern int  autoPrevBombaDecision;   // 0=ninguno, 1=ON, 2=OFF
extern bool autoJustEntered;         // reservado si luego quieres usarlo

// ---- MODO SMART ----
extern int  smartPrevVentDecision;   // 0=ninguno, 1=ON, 2=OFF
extern int  smartPrevBombaDecision;  // 0=ninguno, 1=ON, 2=OFF

// Indica si debemos imprimir / notificar las decisiones al entrar a SMART
extern bool smartJustEntered;

// Para bloqueo de comandos en Auto/Smart (bomba / ventilador)
extern bool bloqueoNotificadoVent;
extern bool bloqueoNotificadoBomba;


// ======================================================
//                 COLA DE COMANDOS MQTT
// ======================================================
extern volatile bool comandoPendiente;
extern String ultimoTopic;
extern String ultimoComando;


// ======================================================
//                     SUELO
// ======================================================
#define SOIL_RAW_SECO    4000
#define SOIL_RAW_HUMEDO  2700


// ======================================================
//                     TIEMPOS
// ======================================================
#define DHT_PERIOD_MS        2000
#define SOIL_PERIOD_MS       1000
#define SENSOR_LOG_PERIOD_MS 30000


// ======================================================
//                     CONFIG
// ======================================================
#define BAUDRATE       115200
#define RECONNECT_MS   2000

#define STACK_SIZE     4096
#define PRIORITY_MQTT  1
#define PRIORITY_LOGIC 1
#define PRIORITY_SENS  1


// ======================================================
//                 PROTOTIPOS DE FUNCIONES
// ======================================================
void mqttTask(void *parameter);
void logicTask(void *parameter);
void sensorTask(void *parameter);

void mqttCallback(char* topic, byte* message, unsigned int length);
void aplicarPWM();

// -------- DIMMER del foco --------
void initDimmerFoco();
int  nivelToGrados(int nivel);   // mapea 0–255 → FOCO_LEVEL_MIN_DEG–FOCO_LEVEL_MAX_DEG

// WiFi / NVS
bool conectar_wifi();
void cargarWiFiDesdeNVS();
void guardarWiFiEnNVS(const char* ssid, const char* pass);
void borrarWiFiEnNVS();

// Cliente MQTT global
extern PubSubClient client;

#endif
