#ifndef INDEX_HTML_H
#define INDEX_HTML_H
const char INDEX_HTML[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>SmartUCB Domótica</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
:root {
  --bg:#0b1220; --bg-card:#111827; --primary:#22c55e; --primary-soft:#22c55e22;
  --accent:#38bdf8; --danger:#ef4444; --text:#e5e7eb; --muted:#9ca3af;
  --border:#1f2933; --radius:14px; --shadow:0 12px 30px rgba(0,0,0,0.4);
}
*{box-sizing:border-box;margin:0;padding:0;font-family:system-ui;}
body{
  background:radial-gradient(circle at top,#1d283a 0,#020617 45%,#000 100%);
  color:var(--text);min-height:100vh;padding:16px;
}
.container{max-width:1100px;margin:auto;display:flex;flex-direction:column;gap:16px;}

header{
  display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:8px;
}

.pill{
  padding:4px 10px;border-radius:999px;font-size:0.75rem;
  border:1px solid var(--primary-soft);background:#020617cc;
  display:inline-flex;align-items:center;gap:6px;
}
.pill-dot{
  width:7px;height:7px;border-radius:50%;
  background:#999;box-shadow:0 0 12px #777;
}

.card{
  background:linear-gradient(145deg,#020617ee,#020617dd);
  border-radius:var(--radius);border:1px solid var(--border);
  padding:14px;box-shadow:var(--shadow);
}
.card-header{
  display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;
}

.grid-2{display:grid;grid-template-columns:repeat(2,1fr);gap:10px;}

button{
  border:none;cursor:pointer;border-radius:999px;padding:7px 10px;font-size:0.8rem;
  background:#111827;color:var(--text);border:1px solid #1f2937;
}
button.primary{
  background:linear-gradient(135deg,#22c55e,#16a34a);color:#022c22;
}
button.danger{
  background:linear-gradient(135deg,#f97373,#ef4444);color:#450a0a;
}
button.sm{padding:5px 9px;font-size:0.75rem;}
button.full{width:100%;}

.mode-btn.active{
  background:var(--primary-soft);color:var(--primary);border-color:var(--primary);
}

.sensor-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;}

input[type="range"]{width:100%;}

.popup{
  position:fixed;top:20px;right:20px;background:#ef4444dd;
  color:white;padding:10px 16px;border-radius:10px;font-size:0.85rem;
  z-index:999;animation:fadeIn .3s;
}
@keyframes fadeIn{
  from{opacity:0;transform:translateY(-10px);}
  to{opacity:1;}
}
</style>
</head>

<body>
<div class="container">

<header>
  <div>
    <div class="title" style="font-size:1.4rem;font-weight:600;">SmartUCB · Sistema Domótico</div>
    <div class="subtitle" style="font-size:0.9rem;color:var(--muted);">Panel de control – ESP32</div>
  </div>

  <div class="pill">
    <span class="pill-dot" id="status-dot"></span>
    <span id="status-text">Conectando...</span>
  </div>
</header>

<main>

<!-- ========================== CONTROL MANUAL ========================== -->
<section class="card">
  <div class="card-header">
    <div>
      <div class="card-title" style="font-size:0.95rem;font-weight:600;">Control Manual</div>
      <div class="card-sub" style="font-size:0.75rem;color:var(--muted);">Foco · Bomba · Ventilador</div>
    </div>
    <span class="badge" id="mode-badge">Modo: ?</span>
  </div>

  <div class="grid-2">

    <!-- =================== FOCO (CON DIMMER) =================== -->
    <div class="card" style="padding:10px;">
      <div class="card-header">
        <span class="card-title">Foco</span>
        <span class="badge" id="foco-status">OFF</span>
      </div>

      <div class="btn-group" style="margin-bottom:8px;">
        <button class="primary sm" onclick="sendCommand('foco','ON')">Encender</button>
        <button class="danger sm" onclick="sendCommand('foco','OFF')">Apagar</button>
      </div>

      <!-- Slider DIMMER -->
      <input id="foco-slider" type="range" min="0" max="255" value="0"
        oninput="updateFocoLabel(this.value)"
        onchange="setFocoLevel(this.value)">
      <span><span id="foco-val">0</span> / 255</span>
    </div>

    <!-- =================== BOMBA =================== -->
    <div class="card" style="padding:10px;">
      <div class="card-header">
        <span class="card-title">Bomba</span>
        <span class="badge" id="bomba-status">OFF</span>
      </div>
      <div class="btn-group">
        <button class="primary sm" onclick="sendCommand('bomba','ON')">Encender</button>
        <button class="danger sm" onclick="sendCommand('bomba','OFF')">Apagar</button>
      </div>
    </div>

  </div>

  <!-- =================== VENTILADOR =================== -->
  <div class="card" style="margin-top:10px;padding:10px;">
    <div class="card-header">
      <span class="card-title">Ventilador</span>
      <span class="badge" id="vent-status">OFF</span>
    </div>

    <div class="btn-group" style="margin-bottom:8px;">
      <button class="primary sm" onclick="sendCommand('vent','ON')">Encender</button>
      <button class="danger sm" onclick="sendCommand('vent','OFF')">Apagar</button>
    </div>

    <input id="vent-slider" type="range" min="0" max="255" value="0"
      oninput="updateVentLabel(this.value)"
      onchange="setVentSpeed(this.value)">
    <span><span id="vent-val">0</span> / 255</span>
  </div>

  <!-- =================== MODOS =================== -->
  <div class="card" style="margin-top:10px;padding:10px;">
    <div class="card-header">
      <span class="card-title">Modos de operación</span>
    </div>
    <div class="btn-group">
      <button id="btn-mode-manual" class="mode-btn full" onclick="setMode('manual')">Manual</button>
      <button id="btn-mode-auto"   class="mode-btn full" onclick="setMode('auto')">Auto</button>
      <button id="btn-mode-smart"  class="mode-btn full" onclick="setMode('smart')">Smart</button>
    </div>
  </div>

</section>


<!-- ========================== SENSORES + WIFI ========================== -->
<section class="card">

  <div class="card-header">
    <span class="card-title">Lectura de sensores</span>
    <span class="badge ok">Cada 3 s</span>
  </div>

  <div class="sensor-grid">
    <div class="sensor-card"><div>Temperatura</div><div id="temp-val">-- °C</div></div>
    <div class="sensor-card"><div>Humedad</div><div id="hum-val">-- %</div></div>
    <div class="sensor-card"><div>Suelo</div><div id="soil-val">-- %</div></div>
  </div>

  <hr style="border:none;border-top:1px solid #1f2937;margin:12px 0;">

  <div class="card-header"><span class="card-title">Configuración Wi-Fi</span></div>

  <div class="form-row">
    <label>SSID</label>
    <input id="wifi-ssid" placeholder="Nombre de la red">
  </div>

  <div class="form-row" style="margin-top:4px;">
    <label>Contraseña</label>
    <input id="wifi-pass" type="password" placeholder="********">
  </div>

  <button class="primary sm full" style="margin-top:6px;" onclick="saveWiFi()">Guardar Wi-Fi</button>
  <button class="sm full" style="margin-top:4px;" onclick="forgetWiFi()">Olvidar credenciales</button>

</section>

</main>

<footer style="font-size:0.75rem;color:var(--muted);text-align:right;margin-top:10px;">
SmartUCB · ESP32 · MQTT · Web UI
</footer>

</div>

<!-- ========================== JAVASCRIPT ========================== -->
<script>

let ssidLoaded = false;
let popupTimeout = null;

// ----------------------------
//   Indicador de ESTADO
// ----------------------------
function setStatus(text, ok = true){
  const dot = document.getElementById("status-dot");
  const msg = document.getElementById("status-text");
  msg.textContent = text;

  if(ok){
    dot.style.background = "#22c55e";
    dot.style.boxShadow = "0 0 8px #22c55e";
  } else {
    dot.style.background = "#ef4444";
    dot.style.boxShadow = "0 0 8px #ef4444";
  }
}

// ----------------------------
//     POPUP BLOQUEOS
// ----------------------------
function showPopup(msg){
  const p = document.createElement("div");
  p.className = "popup";
  p.innerText = msg;
  document.body.appendChild(p);
  clearTimeout(popupTimeout);
  popupTimeout = setTimeout(()=> p.remove(), 3000);
}

// ----------------------------
//    ACTUALIZACIÓN UI
// ----------------------------
function updateVentLabel(val){
  document.getElementById("vent-val").textContent = val;
}

function updateFocoLabel(val){
  document.getElementById("foco-val").textContent = val;
}

function applyState(s){
  if (!s) return;

  setStatus("Conectado", true);

  if (s.bloqueado){
    showPopup("⚠️ Acción bloqueada (Modo AUTO / SMART)");
  }

  // ----- Actuadores -----
  document.getElementById("foco-status").textContent  = s.foco ? "ON":"OFF";
  document.getElementById("bomba-status").textContent = s.bomba? "ON":"OFF";
  document.getElementById("vent-status").textContent  = s.vent ? "ON":"OFF";

  // Slider ventilador
  document.getElementById("vent-slider").value = s.vel ?? 0;
  updateVentLabel(s.vel ?? 0);

  // Slider dimmer foco
  if (s.foco_nivel !== undefined){
    document.getElementById("foco-slider").value = s.foco_nivel;
    updateFocoLabel(s.foco_nivel);
  }

  // ----- Sensores -----
  document.getElementById("temp-val").textContent =
    (s.temp !== undefined ? s.temp.toFixed(1) : "--") + " °C";

  document.getElementById("hum-val").textContent =
    (s.hum !== undefined ? s.hum.toFixed(1) : "--") + " %";

  document.getElementById("soil-val").textContent =
    (s.suelo !== undefined ? s.suelo : "--") + " %";

  // ----- WiFi -----
  if (!ssidLoaded && s.wifi_ssid){
    document.getElementById("wifi-ssid").value = s.wifi_ssid;
    ssidLoaded = true;
  }

  // ----- Modo -----
  applyModeUI(
    s.modo===1 ? "auto" :
    s.modo===2 ? "smart" : "manual"
  );
}

// ----------------------------
//    COMANDOS AL ESP32
// ----------------------------
function sendCommand(dev, action){
  fetch(`/api/${dev}/${action.toLowerCase()}`)
    .then(r => r.json())
    .then(applyState)
    .catch(()=> showPopup("Error enviando comando"));
}

function setVentSpeed(v){
  fetch(`/api/vent/vel?v=${v}`)
    .then(r => r.json())
    .then(applyState);
}

function setFocoLevel(v){
  fetch(`/api/foco/level?v=${v}`)
    .then(r => r.json())
    .then(applyState);
}

function setMode(m){
  fetch(`/api/mode/${m}`)
    .then(r => r.json())
    .then(applyState);
}

function applyModeUI(m){
  document.getElementById("mode-badge").textContent = "Modo: "+m.toUpperCase();
  ["manual","auto","smart"].forEach(x =>
    document.getElementById(`btn-mode-${x}`).classList.toggle("active", x===m)
  );
}

// ----------------------------
//       WIFI
// ----------------------------
function saveWiFi(){
  const ssid=document.getElementById("wifi-ssid").value.trim();
  const pass=document.getElementById("wifi-pass").value.trim();
  if(!ssid) return alert("Ingrese SSID");

  fetch("/api/wifi",{
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify({ssid,pass})
  })
  .then(()=>alert("WiFi guardado. Reiniciando ESP…"));
}

function forgetWiFi(){
  if(!confirm("¿Seguro que deseas borrar la Wi-Fi?")) return;
  fetch("/api/wifi_forget",{method:"POST"})
    .then(()=>alert("Credenciales borradas. Reiniciando…"));
}

// ----------------------------
//    AUTO REFRESH
// ----------------------------
function refreshState(){
  fetch("/api/state")
    .then(r => r.json())
    .then(applyState)
    .catch(()=>{
      setStatus("Sin conexión", false);
      showPopup("Sin conexión…");
    });
}

setInterval(refreshState,3000);
window.onload = refreshState;

</script>

</body>
</html>
)rawliteral";
#endif
