#include <Arduino.h>
#include <AsyncTCP.h>
#include <ESPAsyncWebServer.h>

#include "funciones.h"
#include "index_html.h"
#include "webserver.h"

// ===================
//  WEBSERVER INSTANCE
// ===================
AsyncWebServer server(80);

// ===================
//  JSON DEL ESTADO
// ===================
String jsonEstado(bool bloqueado = false) {
    String json = "{";
    json += "\"modo\":" + String(modoActual) + ",";
    json += "\"foco\":" + String(focoEstado) + ",";
    json += "\"foco_nivel\":" + String(focoNivel) + ",";
    json += "\"bomba\":" + String(bombaEstado) + ",";
    json += "\"vent\":" + String(ventEncendido) + ",";
    json += "\"vel\":" + String(ventVelocidad) + ",";
    json += "\"temp\":" + String(tempActual, 1) + ",";
    json += "\"hum\":" + String(humActual, 1) + ",";
    json += "\"suelo\":" + String(sueloPorc) + ",";
    json += "\"wifi_ssid\":\"" + wifi_ssid_nvs + "\",";
    json += "\"bloqueado\":" + String(bloqueado ? "true" : "false");
    json += "}";
    return json;
}

// ========================
//  BLOQUEAR EN AUTO/SMART
// ========================
bool bloquearSiNoManual(const char* dispositivo) {

    if (modoActual == MODE_MANUAL) return false;

    String msg = String("[BLOQUEO] Comando manual de ") + dispositivo +
                 " bloqueado en modo " +
                 ((modoActual == MODE_AUTO) ? "AUTO" : "SMART");

    Serial.println(msg);
    tg_send(msg);

    return true;
}

// ========================
//  INICIAR WEBSERVER
// ========================
void iniciarWebServer() {

    Serial.println("[WEB] Iniciando WebServer...");

    // Página web principal
    server.on("/", HTTP_GET, [](AsyncWebServerRequest *req){
        req->send_P(200, "text/html", INDEX_HTML);
    });

    // ====================================================
    //  FOCO ON/OFF (SIEMPRE PERMITIDO)
    // ====================================================
    server.on("/api/foco/on", HTTP_GET, [](AsyncWebServerRequest *req){

        focoEstado = true;

        // Si estaba en nivel 0 → usa nivel por defecto
        if (focoNivel <= 0) {
            focoNivel = FOCO_LEVEL_DEFAULT;
        }
        focoGradosObjetivo = nivelToGrados(focoNivel);

        digitalWrite(PIN_FOCO, HIGH);

        tg_send("FOCO ENCENDIDO 💡");
        req->send(200, "application/json", jsonEstado());
    });

    server.on("/api/foco/off", HTTP_GET, [](AsyncWebServerRequest *req){

        focoEstado = false;
        focoNivel = 0;
        focoGradosObjetivo = -1;

        digitalWrite(PIN_FOCO, LOW);

        tg_send("FOCO APAGADO");
        req->send(200, "application/json", jsonEstado());
    });

    // ====================================================
    //  FOCO → NIVEL DIMMER  (SIEMPRE PERMITIDO)
    // ====================================================
    server.on("/api/foco/level", HTTP_GET, [](AsyncWebServerRequest *req){

        if (!req->hasParam("v")) {
            req->send(400, "application/json", "{\"error\":\"faltó parámetro v\"}");
            return;
        }

        int nivel = constrain(req->getParam("v")->value().toInt(), 0, 255);
        focoNivel = nivel;

        if (nivel == 0) {
            focoEstado = false;
            focoGradosObjetivo = -1;
            digitalWrite(PIN_FOCO, LOW);
            tg_send("FOCO APAGADO (nivel 0)");
        } else {
            focoEstado = true;
            focoGradosObjetivo = nivelToGrados(nivel);
            digitalWrite(PIN_FOCO, HIGH);

            tg_send("FOCO → Nivel " + String(nivel) + "/255 💡");
        }

        req->send(200, "application/json", jsonEstado());
    });

    // ====================================================
    //  BOMBA (solo en MANUAL)
    // ====================================================
    server.on("/api/bomba/on", HTTP_GET, [](AsyncWebServerRequest *req){

        if (bloquearSiNoManual("BOMBA")) {
            req->send(200, "application/json", jsonEstado(true));
            return;
        }

        bombaEstado = true;
        digitalWrite(PIN_BOMBA, HIGH);
        tg_send("BOMBA DE AGUA ENCENDIDA 💧");
        req->send(200, "application/json", jsonEstado());
    });

    server.on("/api/bomba/off", HTTP_GET, [](AsyncWebServerRequest *req){

        if (bloquearSiNoManual("BOMBA")) {
            req->send(200, "application/json", jsonEstado(true));
            return;
        }

        bombaEstado = false;
        digitalWrite(PIN_BOMBA, LOW);
        tg_send("BOMBA DE AGUA APAGADA");
        req->send(200, "application/json", jsonEstado());
    });

    // ====================================================
    //  VENTILADOR
    // ====================================================
    server.on("/api/vent/on", HTTP_GET, [](AsyncWebServerRequest *req){

        if (bloquearSiNoManual("VENTILADOR")) {
            req->send(200, "application/json", jsonEstado(true));
            return;
        }

        if (ventUltimaVelocidad <= 0)
            ventUltimaVelocidad = VENT_PWM_DEFAULT;

        ventEncendido = true;
        ventVelocidad = ventUltimaVelocidad;

        aplicarPWM();

        tg_send("VENTILADOR ENCENDIDO 🪭 a velocidad " + String(ventVelocidad));
        req->send(200, "application/json", jsonEstado());
    });

    server.on("/api/vent/off", HTTP_GET, [](AsyncWebServerRequest *req){

        if (bloquearSiNoManual("VENTILADOR")) {
            req->send(200, "application/json", jsonEstado(true));
            return;
        }

        ventEncendido = false;
        ventVelocidad = 0;
        aplicarPWM();

        tg_send("VENTILADOR APAGADO");
        req->send(200, "application/json", jsonEstado());
    });

    // Velocidad del ventilador
    server.on("/api/vent/vel", HTTP_GET, [](AsyncWebServerRequest *req){

        if (bloquearSiNoManual("VENTILADOR")) {
            req->send(200, "application/json", jsonEstado(true));
            return;
        }

        if (req->hasParam("v")) {
            int vel = constrain(req->getParam("v")->value().toInt(), 0, 255);

            if (vel == 0) {
                ventEncendido = false;
                ventVelocidad = 0;
            } else {
                ventEncendido       = true;
                ventVelocidad       = vel;
                ventUltimaVelocidad = vel;
            }

            aplicarPWM();
            tg_send("[VENT] Velocidad ajustada → " + String(ventVelocidad));
        }

        req->send(200, "application/json", jsonEstado());
    });

    // ====================================================
    //  MODOS
    // ====================================================
    server.on("/api/mode/manual", HTTP_GET, [](AsyncWebServerRequest *req){

        modoActual = MODE_MANUAL;
        smartJustEntered = false;

        autoPrevVentDecision  = 0;
        autoPrevBombaDecision = 0;
        smartPrevVentDecision = 0;
        smartPrevBombaDecision = 0;

        tg_send("[MODO] Activado: MANUAL");
        req->send(200, "application/json", jsonEstado());
    });

    server.on("/api/mode/auto", HTTP_GET, [](AsyncWebServerRequest *req){

        modoActual = MODE_AUTO;
        smartJustEntered = false;

        autoPrevVentDecision  = 0;
        autoPrevBombaDecision = 0;

        tg_send("[MODO] Activado: AUTO");
        req->send(200, "application/json", jsonEstado());
    });

    server.on("/api/mode/smart", HTTP_GET, [](AsyncWebServerRequest *req){

        modoActual = MODE_SMART;
        smartJustEntered = true;

        smartPrevVentDecision  = 0;
        smartPrevBombaDecision = 0;

        tg_send("[MODO] Activado: SMART");
        req->send(200, "application/json", jsonEstado());
    });

    // ====================================================
    //  STATUS
    // ====================================================
    server.on("/api/state", HTTP_GET, [](AsyncWebServerRequest *req){
        req->send(200, "application/json", jsonEstado());
    });

    // ====================================================
    //  WIFI CONFIG
    // ====================================================
    server.on("/api/wifi", HTTP_POST,
        [](AsyncWebServerRequest *req){},
        NULL,
        [](AsyncWebServerRequest *req, uint8_t *data, size_t len, size_t, size_t){

            String body;
            for (size_t i = 0; i < len; i++) body += (char)data[i];

            int ssid_pos = body.indexOf("ssid\":\"") + 7;
            int pass_pos = body.indexOf("pass\":\"") + 7;
            int ssid_end = body.indexOf("\"", ssid_pos);
            int pass_end = body.indexOf("\"", pass_pos);

            String ssid = body.substring(ssid_pos, ssid_end);
            String pass = body.substring(pass_pos, pass_end);

            guardarWiFiEnNVS(ssid.c_str(), pass.c_str());

            req->send(200, "application/json", "{\"ok\":true}");
            delay(500);
            ESP.restart();
        }
    );

    server.on("/api/wifi_forget", HTTP_POST, [](AsyncWebServerRequest *req){
        borrarWiFiEnNVS();
        req->send(200, "application/json", "{\"ok\":true}");
        delay(500);
        ESP.restart();
    });

    server.begin();
    Serial.println("[WEB] Servidor iniciado en puerto 80");
}
