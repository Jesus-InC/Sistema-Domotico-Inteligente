#ifndef FUNCIONES_H
#define FUNCIONES_H

#include <WiFi.h>
#include <PubSubClient.h>

// MACROSS

// WIFI
#define WIFI_SSID       "Familia Ibarra21"
#define WIFI_PASSWORD   "Fibarra**21"

// MQTT
#define MQTT_BROKER     "192.168.1.24"
#define MQTT_PORT       1883
#define MQTT_CLIENT_ID  "ESP32_FOCO"

// TOPICS
#define TOPIC_FOCO_CMD      "casa/foco/cmd"
#define TOPIC_FOCO_STATUS   "casa/foco/status"

// HARDWARE
#define PIN_FOCO       17

// TIEMPOS
#define BAUDRATE       115200
#define RECONNECT_MS   2000
#define WIFI_DELAY_MS  500

// FreeRTOS
#define STACK_SIZE     4096
#define PRIORITY_WIFI  1
#define PRIORITY_MQTT  1
#define PRIORITY_LOGIC 1


void wifiTask(void *parameter);
void mqttTask(void *parameter);
void logicTask(void *parameter);

void mqttCallback(char* topic, byte* message, unsigned int length);
void conectar_wifi();
void mqttReconnect();

extern PubSubClient client;

#endif
