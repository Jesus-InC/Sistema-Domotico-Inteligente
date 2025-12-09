#include "funciones.h"

Preferences prefs;

// ===== Variables NVS =====
String wifi_ssid_nvs = "";
String wifi_pass_nvs = "";

// ===== MQTT =====
WiFiClient espClient;
PubSubClient client(espClient);

// ===== ESTADOS =====
volatile bool focoEstado  = false;
volatile bool bombaEstado = false;
volatile uint8_t modoActual = MODE_MANUAL;

bool ventEncendido = false;
volatile int ventVelocidad = 0;
volatile int ventUltimaVelocidad = VENT_PWM_DEFAULT;

// Sensores
float tempActual = 0;
float humActual  = 0;
int sueloRaw     = 0;
int sueloPorc    = 0;

// ML
volatile uint8_t ml_home = 1;
volatile uint8_t ml_rain = 0;

// MQTT queue
volatile bool comandoPendiente = false;
String ultimoTopic   = "";
String ultimoComando = "";

DHT dht(DHT_PIN, DHT22);

// SMART flag
volatile bool smartJustEntered = false;

// =======================================================
//                    N V S — WIFI
// =======================================================

void cargarWiFiDesdeNVS() {
    prefs.begin("config", true);

    wifi_ssid_nvs = prefs.getString("wifi_ssid", "");
    wifi_pass_nvs = prefs.getString("wifi_pass", "");

    prefs.end();

    Serial.println("[NVS] SSID leído: " + wifi_ssid_nvs);
    Serial.println("[NVS] PASS leído: " + wifi_pass_nvs);
}

void guardarWiFiEnNVS(const char* ssid, const char* pass) {
    prefs.begin("config", false);

    prefs.putString("wifi_ssid", ssid);
    prefs.putString("wifi_pass", pass);

    prefs.end();
    Serial.println("[NVS] Credenciales WiFi guardadas.");
}

void borrarWiFiEnNVS() {
    prefs.begin("config", false);

    prefs.remove("wifi_ssid");
    prefs.remove("wifi_pass");

    prefs.end();
    Serial.println("[NVS] WiFi borrada.");
}

// =======================================================
//                      WIFI
// =======================================================
bool conectar_wifi() {

    Serial.println("\n==============================");
    Serial.println("   SISTEMA WIFI INTELIGENTE   ");
    Serial.println("==============================");

    cargarWiFiDesdeNVS();

    // ---- CASO 1: No hay datos → crear AP ----
    if (wifi_ssid_nvs.isEmpty() || wifi_pass_nvs.isEmpty()) {

        Serial.println("[WiFi] No hay credenciales guardadas.");
        Serial.println("[WiFi] Iniciando AP...");

        WiFi.mode(WIFI_AP);
        bool ok = WiFi.softAP("SmartHome-Config", "12345678");

        Serial.print("[AP] IP: ");
        Serial.println(WiFi.softAPIP());

        return false;   // Estamos en AP
    }

    // ---- CASO 2: Sí hay datos → intentar conexión ----
    Serial.println("[WiFi] Intentando conectar a: " + wifi_ssid_nvs);

    WiFi.mode(WIFI_STA);
    WiFi.begin(wifi_ssid_nvs.c_str(), wifi_pass_nvs.c_str());

    uint8_t intentos = 0;
    while (WiFi.status() != WL_CONNECTED && intentos < 30) {
        Serial.print(".");
        delay(300);
        intentos++;
    }

    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\n[WiFi] Conexión exitosa!");
        Serial.print("[WiFi] IP: ");
        Serial.println(WiFi.localIP());
        return true;   // STA conectado
    }

    // ---- Si falla → activar AP ----
    Serial.println("\n[WiFi] ERROR, no se pudo conectar.");
    Serial.println("[WiFi] Cambiando a modo AP...");

    WiFi.mode(WIFI_AP);
    WiFi.softAP("SmartHome-Config", "12345678");

    Serial.print("[AP] IP: ");
    Serial.println(WiFi.softAPIP());

    return false;
}



// =======================================================
//                      MQTT
// =======================================================
void mqttCallback(char* topic, byte* msg, unsigned int length) {
    String payload = "";
    for (int i = 0; i < length; i++) payload += (char)msg[i];

    ultimoTopic = topic;
    ultimoComando = payload;
    comandoPendiente = true;

    Serial.printf("[MQTT] Topic: %s | Payload: %s\n", topic, payload.c_str());
}

void mqttReconnect() {
    while (!client.connected()) {
        Serial.println("[MQTT] Intentando conectar...");

        if (client.connect(MQTT_CLIENT_ID)) {
            Serial.println("[MQTT] Conectado!");

            client.subscribe(TOPIC_FOCO_CMD);
            client.subscribe(TOPIC_BOMBA_CMD);
            client.subscribe(TOPIC_VENT_CMD);
            client.subscribe(TOPIC_VENT_VEL);
            client.subscribe(TOPIC_MODE_CMD);
            client.subscribe(TOPIC_ML_HOME);
            client.subscribe(TOPIC_ML_RAIN);

        } else {
            Serial.print("[MQTT] Falló. Estado=");
            Serial.println(client.state());
            delay(RECONNECT_MS);
        }
    }
}

void mqttTask(void *parameter) {
    if (WiFi.status() == WL_CONNECTED) {
        client.setServer(MQTT_BROKER, MQTT_PORT);
        client.setCallback(mqttCallback);
    }

    while (true) {
        if (WiFi.status() == WL_CONNECTED) {
            if (!client.connected()) mqttReconnect();
            client.loop();
        }
        vTaskDelay(20 / portTICK_PERIOD_MS);
    }
}

// =======================================================
//                      PWM
// =======================================================
void aplicarPWM() {
    ledcWriteChannel(VENT_PWM_CHANNEL, ventVelocidad);
}


// =======================================================
// Lógica principal
// =======================================================
void logicTask(void *parameter) {

    pinMode(PIN_FOCO, OUTPUT);
    pinMode(PIN_BOMBA, OUTPUT);

    // PWM ventilador
    bool ok = ledcAttachChannel(PIN_VENT, VENT_PWM_FREQ, VENT_PWM_RES, VENT_PWM_CHANNEL);
    Serial.printf("[PWM] Canal asignado: %d | OK=%d\n", VENT_PWM_CHANNEL, ok);

    digitalWrite(PIN_FOCO, LOW);
    digitalWrite(PIN_BOMBA, LOW);

    ventEncendido   = false;
    ventVelocidad   = 0;
    ventUltimaVelocidad = VENT_PWM_DEFAULT;
    aplicarPWM();

    Serial.println("[LOGIC] Iniciada");

    static int lastSmartVentCode  = 0;
    static int lastSmartBombaCode = 0;

    while (true) {

        bool modoManual = (modoActual == MODE_MANUAL);

        // =====================================================
        // PROCESAR COMANDOS MQTT
        // =====================================================
        if (comandoPendiente) {
            comandoPendiente = false;

            // --- Cambio de modo ---
            if (ultimoTopic == TOPIC_MODE_CMD) {
                String m = ultimoComando;
                m.toLowerCase();

                uint8_t prev = modoActual;

                if      (m == "manual") modoActual = MODE_MANUAL;
                else if (m == "auto")   modoActual = MODE_AUTO;
                else if (m == "smart")  modoActual = MODE_SMART;

                if (modoActual != prev) smartJustEntered = true;

                continue;
            }

            // ML
            if (ultimoTopic == TOPIC_ML_HOME) { ml_home = ultimoComando.toInt(); continue; }
            if (ultimoTopic == TOPIC_ML_RAIN) { ml_rain = ultimoComando.toInt(); continue; }

            // FOCO (manual always allowed)
            if (ultimoTopic == TOPIC_FOCO_CMD) {
                bool on = (ultimoComando == "ON");
                focoEstado = on;
                digitalWrite(PIN_FOCO, on ? HIGH : LOW);
                client.publish(TOPIC_FOCO_STATUS, on ? "ON" : "OFF", true);
                continue;
            }

            // Manual restrictions
            if (!modoManual) continue;

            // BOMBA
            if (ultimoTopic == TOPIC_BOMBA_CMD) {
                bool on = (ultimoComando == "ON");
                bombaEstado = on;
                digitalWrite(PIN_BOMBA, on ? HIGH : LOW);
                client.publish(TOPIC_BOMBA_STATUS, on ? "ON" : "OFF", true);
                continue;
            }

            // VENT on/off
            if (ultimoTopic == TOPIC_VENT_CMD) {
                bool on = (ultimoComando == "ON");
                ventEncendido = on;
                ventVelocidad = on ? ventUltimaVelocidad : 0;
                aplicarPWM();
                continue;
            }

            // VENT velocidad
            if (ultimoTopic == TOPIC_VENT_VEL) {
                ventVelocidad = constrain(ultimoComando.toInt(), 0, 255);
                ventUltimaVelocidad = ventVelocidad;
                if (ventEncendido) aplicarPWM();
                continue;
            }
        }

        // =====================================================
        // AUTOMÁTICO
        // =====================================================
        if (modoActual == MODE_AUTO) {
            // Ventilador auto
            if (tempActual >= TEMP_SETPOINT) {
                ventEncendido = true;
                ventVelocidad = VENT_PWM_DEFAULT;
                aplicarPWM();
            } else if (tempActual <= TEMP_SETPOINT - TEMP_HISTERESIS) {
                ventEncendido = false;
                ventVelocidad = 0;
                aplicarPWM();
            }

            // Bomba auto
            if (sueloPorc <= SOIL_SETPOINT) {
                bombaEstado = true;
                digitalWrite(PIN_BOMBA, HIGH);
            } else if (sueloPorc >= SOIL_SETPOINT + SOIL_HISTERESIS) {
                bombaEstado = false;
                digitalWrite(PIN_BOMBA, LOW);
            }
        }

        // =====================================================
        // SMART
        // =====================================================
        if (modoActual == MODE_SMART) {
            // Ventilador inteligente
            if (tempActual >= TEMP_SETPOINT && ml_home == 1) {
                ventEncendido = true;
                ventVelocidad = VENT_PWM_DEFAULT;
                aplicarPWM();
            } else {
                ventEncendido = false;
                ventVelocidad = 0;
                aplicarPWM();
            }

            // Bomba inteligente
            if (sueloPorc <= SOIL_SETPOINT && ml_rain == 0) {
                bombaEstado = true;
                digitalWrite(PIN_BOMBA, HIGH);
            } else {
                bombaEstado = false;
                digitalWrite(PIN_BOMBA, LOW);
            }

            smartJustEntered = false;
        }

        vTaskDelay(20 / portTICK_PERIOD_MS);
    }
}

// =======================================================
// SENSOR TASK
// =======================================================
void sensorTask(void *parameter) {

    dht.begin();

    uint32_t lastDHT   = 0;
    uint32_t lastSoil  = 0;
    uint32_t lastLog   = 0;

    uint8_t prevModo = modoActual;

    while (true) {

        uint32_t ahora = millis();

        // ====== DHT22 ======
        if (ahora - lastDHT >= DHT_PERIOD_MS) {
            lastDHT = ahora;

            float t = dht.readTemperature();
            float h = dht.readHumidity();

            if (!isnan(t)) tempActual = t;
            if (!isnan(h)) humActual  = h;
        }

        // ====== SUELO ======
        if (ahora - lastSoil >= SOIL_PERIOD_MS) {
            lastSoil = ahora;

            int raw = analogRead(SOIL_PIN);
            sueloRaw = raw;

            int porc = map(raw, SOIL_RAW_SECO, SOIL_RAW_HUMEDO, 0, 100);
            porc = constrain(porc, 0, 100);

            sueloPorc = porc;
        }

        vTaskDelay(50 / portTICK_PERIOD_MS);
    }
}
