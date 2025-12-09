#include <Arduino.h>
#include <AsyncTCP.h>
#include <ESPAsyncWebServer.h>

#include "funciones.h"
#include "index_html.h"
#include "webserver.h"

AsyncWebServer server(80);

String jsonEstado() {
    String json = "{";
    json += "\"modo\":" + String(modoActual) + ",";
    json += "\"foco\":" + String(focoEstado) + ",";
    json += "\"bomba\":" + String(bombaEstado) + ",";
    json += "\"vent\":" + String(ventEncendido) + ",";
    json += "\"vel\":" + String(ventVelocidad) + ",";
    json += "\"temp\":" + String(tempActual, 1) + ",";
    json += "\"hum\":" + String(humActual, 1) + ",";
    json += "\"suelo\":" + String(sueloPorc) + ",";
    json += "\"wifi_ssid\":\"" + wifi_ssid_nvs + "\"";
    json += "}";
    return json;
}

void iniciarWebServer() {

    Serial.println("[WEB] Iniciando WebServer...");

    server.on("/", HTTP_GET, [](AsyncWebServerRequest *req){
        req->send_P(200, "text/html", INDEX_HTML);
    });

    // ====== CONTROL MANUAL ======
    server.on("/api/foco/on", HTTP_GET, [](AsyncWebServerRequest *req){
        focoEstado = true;
        digitalWrite(PIN_FOCO, HIGH);
        req->send(200, "application/json", jsonEstado());
    });

    server.on("/api/foco/off", HTTP_GET, [](AsyncWebServerRequest *req){
        focoEstado = false;
        digitalWrite(PIN_FOCO, LOW);
        req->send(200, "application/json", jsonEstado());
    });

    server.on("/api/bomba/on", HTTP_GET, [](AsyncWebServerRequest *req){
        bombaEstado = true;
        digitalWrite(PIN_BOMBA, HIGH);
        req->send(200, "application/json", jsonEstado());
    });

    server.on("/api/bomba/off", HTTP_GET, [](AsyncWebServerRequest *req){
        bombaEstado = false;
        digitalWrite(PIN_BOMBA, LOW);
        req->send(200, "application/json", jsonEstado());
    });

    // ====== VENTILADOR ======
    server.on("/api/vent/on", HTTP_GET, [](AsyncWebServerRequest *req){
        ventEncendido = true;
        ventVelocidad = ventUltimaVelocidad;
        aplicarPWM();
        req->send(200, "application/json", jsonEstado());
    });

    server.on("/api/vent/off", HTTP_GET, [](AsyncWebServerRequest *req){
        ventEncendido = false;
        ventVelocidad = 0;
        aplicarPWM();
        req->send(200, "application/json", jsonEstado());
    });

    server.on("/api/vent/vel", HTTP_GET, [](AsyncWebServerRequest *req){
        if (req->hasParam("v")) {
            int vel = req->getParam("v")->value().toInt();
            ventVelocidad = vel;
            ventUltimaVelocidad = vel;
            if (ventEncendido) aplicarPWM();
        }
        req->send(200, "application/json", jsonEstado());
    });

    // ====== MODOS ======
    server.on("/api/mode/manual", HTTP_GET, [](AsyncWebServerRequest *req){
        modoActual = MODE_MANUAL;
        req->send(200, "application/json", jsonEstado());
    });

    server.on("/api/mode/auto", HTTP_GET, [](AsyncWebServerRequest *req){
        modoActual = MODE_AUTO;
        req->send(200, "application/json", jsonEstado());
    });

    server.on("/api/mode/smart", HTTP_GET, [](AsyncWebServerRequest *req){
        modoActual = MODE_SMART;
        req->send(200, "application/json", jsonEstado());
    });

    // ====== ESTADO ======
    server.on("/api/state", HTTP_GET, [](AsyncWebServerRequest *req){
        req->send(200, "application/json", jsonEstado());
    });

    // ====== WIFI: GUARDAR ======
    server.on("/api/wifi", HTTP_POST,
        [](AsyncWebServerRequest *req){},
        NULL,
        [](AsyncWebServerRequest *req, uint8_t *data, size_t len, size_t, size_t){

            String body = "";
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
    });

    // ====== WIFI: OLVIDAR ======
    server.on("/api/wifi_forget", HTTP_POST, [](AsyncWebServerRequest *req){
        borrarWiFiEnNVS();
        req->send(200, "application/json", "{\"ok\":true}");
        delay(500);
        ESP.restart();
    });

    server.begin();
    Serial.println("[WEB] Servidor iniciado en puerto 80");
}
