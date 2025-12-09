void setup() {

    Serial.begin(BAUDRATE);

    pinMode(PIN_BOMBA, OUTPUT);
    digitalWrite(PIN_BOMBA, LOW);
    delay(150);

    bool wifiOK = conectar_wifi();

    // Iniciar WebServer SIEMPRE (STA o AP)
    iniciarWebServer();

    // MQTT solo si hay WiFi STA
    if (wifiOK) {
        client.setServer(MQTT_BROKER, MQTT_PORT);
        client.setCallback(mqttCallback);
    }

    delay(1000);

    xTaskCreate(mqttTask,   "MQTT Task",   STACK_SIZE, NULL, PRIORITY_MQTT,  NULL);
    xTaskCreate(sensorTask, "Sensor Task", STACK_SIZE, NULL, PRIORITY_SENS,  NULL);
    xTaskCreate(logicTask,  "Logic Task",  STACK_SIZE, NULL, PRIORITY_LOGIC, NULL);
}

void loop() {}
