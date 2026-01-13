#include "funciones.h"

Preferences prefs;

// -------------------------
//  Variables NVS
// -------------------------
String wifi_ssid_nvs = "";
String wifi_pass_nvs = "";

// -------------------------
//  MQTT
// -------------------------
WiFiClient espClient;
PubSubClient client(espClient);

// -------------------------
//  ESTADOS / GLOBALES
// -------------------------
volatile bool focoEstado  = false;
volatile bool bombaEstado = false;
volatile uint8_t modoActual = MODE_MANUAL;

volatile bool comandoPendiente = false;
String ultimoTopic   = "";
String ultimoComando = "";

DHT dht(DHT_PIN, DHT22);

bool        ventEncendido        = false;
volatile int ventVelocidad       = 0;
volatile int ventUltimaVelocidad = VENT_PWM_DEFAULT;

// --------- DIMMER FOCO ---------
volatile int focoNivel          = 0;
volatile int focoGradosObjetivo = -1;
volatile int focoGE             = 0;

// Timer hardware
hw_timer_t* focoTimer = nullptr;

// Sensores
float tempActual = 0;
float humActual  = 0;
int   sueloRaw   = 0;
int   sueloPorc  = 0;

// ML
volatile uint8_t ml_home = 1;
volatile uint8_t ml_rain = 0;

uint8_t prev_ml_home = 1;
uint8_t prev_ml_rain = 0;

// Auto
int  autoPrevVentDecision  = 0;
int  autoPrevBombaDecision = 0;
bool autoJustEntered       = false;

// Smart
int  smartPrevVentDecision  = 0;
int  smartPrevBombaDecision = 0;

bool smartJustEntered = false;

// Bloqueos
bool bloqueoNotificadoVent  = false;
bool bloqueoNotificadoBomba = false;

bool mlWelcomeSent = false;


// --------------------------------------------------
//  MAPEO DIMMER: nivel 0–255 → ángulo 28–179
// --------------------------------------------------
int nivelToGrados(int nivel) {
    if (nivel <= 0) return -1;

    int g = map(nivel, 1, 255, FOCO_LEVEL_MAX_DEG, FOCO_LEVEL_MIN_DEG);
    g = constrain(g, FOCO_LEVEL_MIN_DEG, FOCO_LEVEL_MAX_DEG);

    return g;
}


// ======================================================
//                       TELEGRAM
// ======================================================
bool tg_send(const String &msg) {
    if (WiFi.status() != WL_CONNECTED) return false;

    HTTPClient http;
    String url = "https://api.telegram.org/bot" + String(TG_BOT_TOKEN) +
                 "/sendMessage?chat_id=" + String(TG_CHAT_ID) +
                 "&text=" + msg;

    http.begin(url);
    int code = http.GET();
    http.end();
    return (code == 200);
}


// ======================================================
//                  NVS — WIFI
// ======================================================
void cargarWiFiDesdeNVS() {
    prefs.begin("config", true);

    wifi_ssid_nvs = prefs.getString("wifi_ssid", "");
    wifi_pass_nvs = prefs.getString("wifi_pass", "");

    prefs.end();
}

void guardarWiFiEnNVS(const char* ssid, const char* pass) {
    prefs.begin("config", false);
    prefs.putString("wifi_ssid", ssid);
    prefs.putString("wifi_pass", pass);
    prefs.end();
}

void borrarWiFiEnNVS() {
    prefs.begin("config", false);
    prefs.remove("wifi_ssid");
    prefs.remove("wifi_pass");
    prefs.end();
}


// ======================================================
//                          WIFI
// ======================================================
bool conectar_wifi() {

    cargarWiFiDesdeNVS();

    if (wifi_ssid_nvs.isEmpty() || wifi_pass_nvs.isEmpty()) {

        WiFi.mode(WIFI_AP);
        WiFi.softAP("SmartHome-Config", "12345678");
        return false;
    }

    WiFi.mode(WIFI_STA);
    WiFi.begin(wifi_ssid_nvs.c_str(), wifi_pass_nvs.c_str());

    for (int i = 0; i < 30; i++) {
        if (WiFi.status() == WL_CONNECTED) break;
        delay(300);
    }

    if (WiFi.status() == WL_CONNECTED) {
        return true;
    }

    WiFi.mode(WIFI_AP);
    WiFi.softAP("SmartHome-Config", "12345678");
    return false;
}


// ======================================================
//                          MQTT
// ======================================================
void mqttCallback(char* topic, byte* msg, unsigned int len) {
    String payload = "";
    for (unsigned int i = 0; i < len; i++) payload += (char)msg[i];

    ultimoTopic   = topic;
    ultimoComando = payload;
    comandoPendiente = true;
}


void mqttReconnect() {
    while (!client.connected()) {
        if (client.connect(MQTT_CLIENT_ID)) {

            client.subscribe(TOPIC_FOCO_CMD);
            client.subscribe(TOPIC_BOMBA_CMD);
            client.subscribe(TOPIC_VENT_CMD);
            client.subscribe(TOPIC_VENT_VEL);
            client.subscribe(TOPIC_MODE_CMD);
            client.subscribe(TOPIC_ML_HOME);
            client.subscribe(TOPIC_ML_RAIN);

        } else {
            delay(RECONNECT_MS);
        }
    }
}


void mqttTask(void *p) {
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


// ======================================================
//                     PWM VENTILADOR
// ======================================================
void aplicarPWM() {
    // TU MÉTODO PERSONALIZADO
    ledcWriteChannel(VENT_PWM_CHANNEL, ventVelocidad);
}


// ======================================================
//                 DIMMER — ISR
// ======================================================
void IRAM_ATTR isrCruceCeroFoco() {
    focoGE = 0;
    digitalWrite(PIN_FOCO_TRIAC, LOW);
}

void IRAM_ATTR isrFocoTimer() {
    focoGE++;

    int grados = focoGradosObjetivo;

    if (!focoEstado || grados < 0) return;

    if (focoGE == grados) {
        digitalWrite(PIN_FOCO_TRIAC, HIGH);
    }
}


// ======================================================
//                   INIT DIMMER
// ======================================================
void initDimmerFoco() {

    pinMode(PIN_FOCO_TRIAC, OUTPUT);
    digitalWrite(PIN_FOCO_TRIAC, LOW);

    pinMode(PIN_FOCO_ZC, INPUT);
    attachInterrupt(digitalPinToInterrupt(PIN_FOCO_ZC), isrCruceCeroFoco, CHANGE);

    focoEstado          = false;
    focoNivel           = 0;
    focoGradosObjetivo  = -1;
    focoGE              = 0;

    focoTimer = timerBegin(1000000);
    timerAttachInterrupt(focoTimer, &isrFocoTimer);
    timerAlarm(focoTimer, FOCO_TGE_US, true, true);
}


// ======================================================
//                     LOGIC TASK
// ======================================================
void logicTask(void *p) {

    pinMode(PIN_FOCO, OUTPUT);
    pinMode(PIN_BOMBA, OUTPUT);

    // ==========================
    //     PWM DEL VENTILADOR
    // ==========================
    ledcAttachChannel(PIN_VENT, VENT_PWM_FREQ, VENT_PWM_RES, VENT_PWM_CHANNEL);
    Serial.printf("[PWM] Ventilador configurado en canal %d\n", VENT_PWM_CHANNEL);

    ventEncendido = false;
    ventVelocidad = 0;

    while (true) {

        bool manual = (modoActual == MODE_MANUAL);

        // ============ MQTT =============
        if (comandoPendiente) {

            comandoPendiente = false;

            // ====== MODO ======
            if (ultimoTopic == TOPIC_MODE_CMD) {

                String m = ultimoComando;
                m.toLowerCase();

                if      (m == "manual") modoActual = MODE_MANUAL;
                else if (m == "auto")   modoActual = MODE_AUTO;
                else if (m == "smart")  modoActual = MODE_SMART;

                bloqueoNotificadoVent  = false;
                bloqueoNotificadoBomba = false;

                autoPrevVentDecision  = 0;
                autoPrevBombaDecision = 0;
                smartPrevVentDecision  = 0;
                smartPrevBombaDecision = 0;

                smartJustEntered = (modoActual == MODE_SMART);
                continue;
            }

            // ML
            if (ultimoTopic == TOPIC_ML_HOME) { ml_home = ultimoComando.toInt(); continue; }
            if (ultimoTopic == TOPIC_ML_RAIN) { ml_rain = ultimoComando.toInt(); continue; }

            // FOCO (siempre permitido)
            if (ultimoTopic == TOPIC_FOCO_CMD) {
                bool on = (ultimoComando == "ON");
                focoEstado = on;

                digitalWrite(PIN_FOCO, on ? HIGH : LOW);

                if (!on) {
                    focoNivel          = 0;
                    focoGradosObjetivo = -1;
                } else {
                    if (focoNivel <= 0) focoNivel = FOCO_LEVEL_DEFAULT;
                    focoGradosObjetivo = nivelToGrados(focoNivel);
                }
                continue;
            }

            // BLOQUEOS AUTO/SMART
            if (!manual) {
                if (ultimoTopic == TOPIC_BOMBA_CMD && !bloqueoNotificadoBomba) {
                    bloqueoNotificadoBomba = true;
                }

                if (ultimoTopic == TOPIC_VENT_CMD && !bloqueoNotificadoVent) {
                    bloqueoNotificadoVent = true;
                }

                continue;
            }

            // BOMBA MANUAL
            if (ultimoTopic == TOPIC_BOMBA_CMD) {
                bool on = (ultimoComando == "ON");
                bombaEstado = on;
                digitalWrite(PIN_BOMBA, on ? HIGH : LOW);
                continue;
            }

            // VENTILADOR MANUAL ON/OFF
            if (ultimoTopic == TOPIC_VENT_CMD) {
                bool on = (ultimoComando == "ON");

                if (on) {
                    if (ventUltimaVelocidad <= 0)
                        ventUltimaVelocidad = VENT_PWM_DEFAULT;

                    ventVelocidad = ventUltimaVelocidad;
                    ventEncendido = true;
                } else {
                    ventEncendido = false;
                    ventVelocidad = 0;
                }

                aplicarPWM();
                continue;
            }

            // VENTILADOR MANUAL VELOCIDAD
            if (ultimoTopic == TOPIC_VENT_VEL) {

                int vel = constrain(ultimoComando.toInt(), 0, 255);

                if (vel == 0) {
                    ventEncendido = false;
                    ventVelocidad = 0;
                    ventUltimaVelocidad = 0;
                } else {
                    ventEncendido = true;
                    ventVelocidad = vel;
                    ventUltimaVelocidad = vel;
                }

                aplicarPWM();
                continue;
            }
        }

        // ======================================================
        //                         AUTO
        // ======================================================
        if (modoActual == MODE_AUTO) {

            if (tempActual >= TEMP_SETPOINT) {
                ventEncendido = true;
                ventVelocidad = VENT_PWM_DEFAULT;
            } else if (tempActual <= TEMP_SETPOINT - TEMP_HISTERESIS) {
                ventEncendido = false;
                ventVelocidad = 0;
            }

            aplicarPWM();

            if (sueloPorc <= SOIL_SETPOINT) {
                bombaEstado = true;
            } else if (sueloPorc >= SOIL_SETPOINT + SOIL_HISTERESIS) {
                bombaEstado = false;
            }

            digitalWrite(PIN_BOMBA, bombaEstado ? HIGH : LOW);
        }

        // ======================================================
        //                         SMART
        // ======================================================
        if (modoActual == MODE_SMART) {

            if (tempActual >= TEMP_SETPOINT) {
                if (ml_home == 1) {
                    ventEncendido = true;
                    ventVelocidad = VENT_PWM_DEFAULT;
                } else {
                    ventEncendido = false;
                    ventVelocidad = 0;
                }
            } else if (tempActual <= TEMP_SETPOINT - TEMP_HISTERESIS) {
                ventEncendido = false;
                ventVelocidad = 0;
            }

            aplicarPWM();

            if (sueloPorc <= SOIL_SETPOINT) {
                bombaEstado = (ml_rain == 0);
            } else if (sueloPorc >= SOIL_SETPOINT + SOIL_HISTERESIS) {
                bombaEstado = false;
            }

            digitalWrite(PIN_BOMBA, bombaEstado ? HIGH : LOW);

            smartJustEntered = false;
        }

        vTaskDelay(20 / portTICK_PERIOD_MS);
    }
}


// ======================================================
//                        SENSORES
// ======================================================
void sensorTask(void *p) {

    dht.begin();

    uint32_t lastDHT  = 0;
    uint32_t lastSoil = 0;

    while (true) {
        uint32_t now = millis();

        if (now - lastDHT >= DHT_PERIOD_MS) {
            lastDHT = now;
            float t = dht.readTemperature();
            float h = dht.readHumidity();
            if (!isnan(t)) tempActual = t;
            if (!isnan(h)) humActual  = h;
        }

        if (now - lastSoil >= SOIL_PERIOD_MS) {
            lastSoil = now;
            int raw = analogRead(SOIL_PIN);
            sueloRaw = raw;

            int porc = map(raw, SOIL_RAW_SECO, SOIL_RAW_HUMEDO, 0, 100);
            sueloPorc = constrain(porc, 0, 100);
        }

        vTaskDelay(50 / portTICK_PERIOD_MS);
    }
}
