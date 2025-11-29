#include "funciones.h"

WiFiClient espClient;
PubSubClient client(espClient);

volatile bool focoEstado = false;
volatile bool comandoPendiente = false;
String ultimoComando = "";


// Tópicos recibidos
void mqttCallback(char* topic, byte* message, unsigned int length) {
    String payload = "";
    for (int i = 0; i < length; i++) payload += (char)message[i];

    Serial.print("[MQTT] Mensaje recibido | Topic: ");
    Serial.print(topic);
    Serial.print(" | Payload: ");
    Serial.println(payload);

    if (String(topic) == TOPIC_FOCO_CMD) {
        ultimoComando = payload;
        comandoPendiente = true;
    }
}


// Tareas del Wifi
void conectar_wifi() {
    Serial.println("[WiFi] Conectando a red...");
    Serial.print("[WiFi] SSID: ");
    Serial.println(WIFI_SSID);

    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    while (WiFi.status() != WL_CONNECTED) {
        Serial.print(".");
        vTaskDelay(WIFI_DELAY_MS / portTICK_PERIOD_MS);
    }

    Serial.println("\n[WiFi] Conectado!");
    Serial.print("[WiFi] IP: ");
    Serial.println(WiFi.localIP());
}

void wifiTask(void *parameter) {
    conectar_wifi();
    vTaskDelete(NULL);
}


// Tareas del MQTT
void mqttReconnect() {
    while (!client.connected()) {

        Serial.println("[MQTT] Intentando conectar al broker...");

        if (client.connect(MQTT_CLIENT_ID)) {
            Serial.println("[MQTT] Conectado correctamente!");

            client.subscribe(TOPIC_FOCO_CMD);
            Serial.print("[MQTT] Suscrito a: ");
            Serial.println(TOPIC_FOCO_CMD);

            client.publish(TOPIC_FOCO_STATUS, "OFF", true);

        } else {
            Serial.print("[MQTT] Falló. Código: ");
            Serial.println(client.state());
            Serial.println("[MQTT] Reintentando en 2s...");
            vTaskDelay(RECONNECT_MS / portTICK_PERIOD_MS);
        }
    }
}

void mqttTask(void *parameter) {

    vTaskDelay(2000 / portTICK_PERIOD_MS);  // aseguramos Serial listo

    client.setServer(MQTT_BROKER, MQTT_PORT);
    client.setCallback(mqttCallback);

    mqttReconnect();

    while (true) {
        if (!client.connected()) mqttReconnect();
        client.loop();  
        vTaskDelay(10 / portTICK_PERIOD_MS);
    }
}


// Lógica del foco
void logicTask(void *parameter) {

    pinMode(PIN_FOCO, OUTPUT);
    digitalWrite(PIN_FOCO, LOW);

    Serial.println("[LOGIC] Control del foco iniciado");

    while (true) {

        if (comandoPendiente) {
            comandoPendiente = false;

            if (ultimoComando == "ON") {
                focoEstado = true;
                digitalWrite(PIN_FOCO, HIGH);
                Serial.println("[FOCO] ENCENDIDO");
                client.publish(TOPIC_FOCO_STATUS, "ON", true);
            }
            else if (ultimoComando == "OFF") {
                focoEstado = false;
                digitalWrite(PIN_FOCO, LOW);
                Serial.println("[FOCO] APAGADO");
                client.publish(TOPIC_FOCO_STATUS, "OFF", true);
            }
        }

        vTaskDelay(20 / portTICK_PERIOD_MS);
    }
}