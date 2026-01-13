#include "funciones.h"
#include "webserver.h"

void setup() {

    Serial.begin(BAUDRATE);
    delay(300);

    Serial.println("\n=== SmartUCB – Boot ===");

    // ----------------------------------------------------
    //  ESTADOS INICIALES
    // ----------------------------------------------------
    modoActual = MODE_MANUAL;

    focoEstado  = false;
    bombaEstado = false;

    ventEncendido       = false;
    ventVelocidad       = 0;
    ventUltimaVelocidad = VENT_PWM_DEFAULT;

    autoPrevVentDecision   = 0;
    autoPrevBombaDecision  = 0;
    smartPrevVentDecision  = 0;
    smartPrevBombaDecision = 0;

    bloqueoNotificadoVent  = false;
    bloqueoNotificadoBomba = false;

    smartJustEntered = false;

    // ----------------------------------------------------
    //  HARDWARE FISICO A OFF
    // ----------------------------------------------------
    pinMode(PIN_FOCO, OUTPUT);
    digitalWrite(PIN_FOCO, LOW);

    pinMode(PIN_BOMBA, OUTPUT);
    digitalWrite(PIN_BOMBA, LOW);

    delay(150);

    // ----------------------------------------------------
    //  INICIALIZAR DIMMER (TRIAC + ZC + TIMER)
    // ----------------------------------------------------
    initDimmerFoco();
    delay(200);

    // ----------------------------------------------------
    //  CONECTAR WIFI (STA o AP)
    // ----------------------------------------------------
    bool wifiOK = conectar_wifi();

    // ----------------------------------------------------
    //  INICIAR WEBSERVER (siempre)
    // ----------------------------------------------------
    iniciarWebServer();

    // ----------------------------------------------------
    //  CONFIGURAR MQTT (solo si STA conectado)
    // ----------------------------------------------------
    if (wifiOK) {
        client.setServer(MQTT_BROKER, MQTT_PORT);
        client.setCallback(mqttCallback);
    }

    delay(800);

    // ----------------------------------------------------
    //  TAREAS FREERTOS
    // ----------------------------------------------------
    xTaskCreate(
        mqttTask, "MQTT Task",
        STACK_SIZE, NULL, PRIORITY_MQTT, NULL
    );

    xTaskCreate(
        sensorTask, "Sensor Task",
        STACK_SIZE, NULL, PRIORITY_SENS, NULL
    );

    xTaskCreate(
        logicTask, "Logic Task",
        STACK_SIZE, NULL, PRIORITY_LOGIC, NULL
    );

    Serial.println("[BOOT] Sistema principal iniciado.");
}

void loop() {
    // FreeRTOS se encarga de todo
}
