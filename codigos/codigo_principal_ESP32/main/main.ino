#include "funciones.h"

void setup() {
    Serial.begin(BAUDRATE);
    delay(1500);   // IMPORTANTE: evita resets
    xTaskCreate(wifiTask, "WiFi Task", STACK_SIZE, NULL, PRIORITY_WIFI, NULL);
    xTaskCreate(mqttTask, "MQTT Task", STACK_SIZE, NULL, PRIORITY_MQTT, NULL);
    xTaskCreate(logicTask, "Logic Task", STACK_SIZE, NULL, PRIORITY_LOGIC, NULL);
}

void loop() {
    // FreeRTOS maneja todo
}