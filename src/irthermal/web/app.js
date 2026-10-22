"use strict";

/* The whole front end. No build step and no framework: the page is an <img>
   pointed at the MJPEG endpoint plus a socket carrying the numbers, and both of
   those are already done by the time this file runs. */

const API = "/api";
const TELEMETRY_FPS = 4;
const RECONNECT_MS = 2000;

const el = (id) => document.getElementById(id);

let cameraId = null;
let socket = null;
let reconnectTimer = null;
let syncControls = false;

let x = 0;
let y = 0;

// --- helpers ---------------------------------------------------------------

async function api(path, options) {
  const response = await fetch(API + path, options);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body.detail) detail = JSON.stringify(body.detail);
    } catch (ignored) {
      /* not every error body is JSON */
    }
    throw new Error(detail);
  }
  return response.status === 204 ? null : response.json();
}

function showError(message) {
  const box = el("error");
  box.textContent = message;
  box.hidden = message === null;
}

function degrees(value) {
  return value === null || value === undefined ? "—" : `${value.toFixed(1)} °C`;
}

// --- stream ----------------------------------------------------------------

function streamQuery() {
  const query = new URLSearchParams();
  const colormap = el("colormap").value;
  if (colormap) query.set("colormap", colormap);

  if (!el("auto-scale").checked) {
    const min = el("min-celsius").value;
    const max = el("max-celsius").value;
    if (min !== "") query.set("min_celsius", min);
    if (max !== "") query.set("max_celsius", max);
  }

  if (el("equalize").checked) query.set("equalize", "true");
  query.set("upscale", el("upscale").value);
  query.set("fps", el("fps").value);
  return query;
}

function restartStream() {
  if (!cameraId) return;
  // A cache buster: reusing the same URL can leave the browser attached to the
  // stream it is already reading rather than opening one with the new settings.
  const query = streamQuery();
  query.set("t", Date.now().toString());
  el("stream").src = `${API}/cameras/${cameraId}/stream.mjpg?${query}`;
}

// --- telemetry socket ------------------------------------------------------

function connectSocket() {
  if (socket) {
    socket.onclose = null;
    socket.close();
  }

  const scheme = location.protocol === "https:" ? "wss" : "ws";
  const url = `${scheme}://${location.host}${API}/cameras/${cameraId}/telemetry?fps=${TELEMETRY_FPS}`;
  socket = new WebSocket(url);
  syncControls = true;

  socket.onopen = () => setLink(true);
  socket.onmessage = (event) => render(JSON.parse(event.data));
  socket.onclose = () => {
    setLink(false);
    clearTimeout(reconnectTimer);
    reconnectTimer = setTimeout(connectSocket, RECONNECT_MS);
  };
}

function setLink(live) {
  const badge = el("link");
  badge.textContent = live ? "live" : "offline";
  badge.className = live ? "badge live" : "badge offline";
}

// --- rendering -------------------------------------------------------------

function render(message) {
  el("seq").textContent = `frame ${message.sequence}`;
  el("rate").textContent = `${message.fps.toFixed(1)} fps`;

  renderPoints(message);
  renderTelemetry(message);

  if (syncControls) {
    syncControls = false;
    el("range-mode").value = message.adjustments.range_mode >= 400 ? "high" : "normal";
    const form = el("parameters");
    form.emissivity.placeholder = message.telemetry.emissivity.toFixed(2);
    form.distance_m.placeholder = message.telemetry.distance_m;
    form.humidity.placeholder = message.telemetry.humidity.toFixed(2);
    form.temperature_offset.placeholder = message.adjustments.user_offset.toFixed(1);
  }
}

function row(name, measurement, derived) {
  const tr = document.createElement("tr");
  if (derived) tr.className = "derived";
  for (const value of [name, measurement.x, measurement.y, degrees(measurement.celsius)]) {
    const td = document.createElement("td");
    td.textContent = value;
    tr.appendChild(td);
  }
  return tr;
}

function renderPoints(message) {
  const body = el("points").tBodies[0];
  body.replaceChildren();

  for (const [name, measurement] of Object.entries(message.points)) {
    body.appendChild(row(name, measurement, false));
  }
  body.appendChild(row("minimum", message.minimum, true));
  body.appendChild(row("maximum", message.maximum, true));
  body.appendChild(row("centre", message.center, true));

  const mean = document.createElement("tr");
  mean.className = "derived";
  for (const value of ["mean", "", "", degrees(message.mean_celsius)]) {
    const td = document.createElement("td");
    td.textContent = value;
    mean.appendChild(td);
  }
  body.appendChild(mean);
}

function renderTelemetry(message) {
  const t = message.telemetry;
  const entries = [
    ["Shutter", degrees(t.shutter_temp_c)],
    ["Core", degrees(t.core_temp_c)],
    ["Sensor", degrees(t.fpa_temp_c)],
    ["Air", degrees(t.air_temp_c)],
    ["Reflected", degrees(t.reflected_temp_c)],
    ["Emissivity", t.emissivity.toFixed(2)],
    ["Humidity", t.humidity.toFixed(2)],
    ["Distance", `${t.distance_m} m`],
    ["Offset", degrees(message.adjustments.user_offset)],
    ["Range", message.adjustments.range_mode >= 400 ? "high" : "normal"],
    ["Correction", message.corrected ? "software" : "camera"],
    ["Firmware", t.firmware],
    ["Serial", t.serial],
  ];

  const list = el("telemetry");
  list.replaceChildren();
  for (const [term, value] of entries) {
    const dt = document.createElement("dt");
    dt.textContent = term;
    const dd = document.createElement("dd");
    dd.textContent = value;
    list.append(dt, dd);
  }
}

// --- camera selection ------------------------------------------------------

function selectCamera(id) {
  cameraId = id;
  location.hash = id;
  el("camera").value = id;
  restartStream();
  connectSocket();
}

async function loadCameras() {
  const cameras = await api("/cameras");
  const select = el("camera");
  select.replaceChildren();

  for (const camera of cameras) {
    const option = document.createElement("option");
    option.value = camera.id;
    option.textContent = camera.width
      ? `${camera.id} — ${camera.width}×${camera.height}`
      : camera.id;
    select.appendChild(option);
  }

  const wanted = location.hash.slice(1);
  const chosen = cameras.find((camera) => camera.id === wanted) || cameras[0];
  if (!chosen) throw new Error("no cameras configured");

  el("source").textContent = chosen.source || "no source";
  selectCamera(chosen.id);
}

async function loadColormaps() {
  const select = el("colormap");
  const fallback = document.createElement("option");
  fallback.value = "";
  fallback.textContent = "camera default";
  select.appendChild(fallback);

  for (const name of await api("/colormaps")) {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    select.appendChild(option);
  }
}

// --- wiring ----------------------------------------------------------------

function bind() {
  el("camera").addEventListener("change", (event) => selectCamera(event.target.value));

  for (const id of ["colormap", "equalize", "min-celsius", "max-celsius"]) {
    el(id).addEventListener("change", restartStream);
  }

  el("auto-scale").addEventListener("change", (event) => {
    const manual = !event.target.checked;
    el("min-celsius").disabled = !manual;
    el("max-celsius").disabled = !manual;
    restartStream();
  });

  const sliders = [
    ["upscale", "upscale-value", "×"],
    ["fps", "fps-value", ""],
  ];
  for (const [id, output, suffix] of sliders) {
    el(id).addEventListener("input", (event) => {
      el(output).textContent = event.target.value + suffix;
    });
    el(id).addEventListener("change", restartStream);
  }

  el("calibrate").addEventListener("click", async (event) => {
    event.target.disabled = true;
    try {
      await api(`/cameras/${cameraId}/calibrate`, { method: "POST" });
      showError(null);
    } catch (error) {
      showError(`Calibration failed: ${error.message}`);
    } finally {
      event.target.disabled = false;
    }
  });

  el("range-mode").addEventListener("change", async (event) => {
    try {
      await api(`/cameras/${cameraId}/range`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode: event.target.value }),
      });
      showError(null);
    } catch (error) {
      showError(`Range change failed: ${error.message}`);
    }
  });

  el("parameters").addEventListener("submit", async (event) => {
    event.preventDefault();
    const body = {};
    for (const [name, value] of new FormData(event.target)) {
      if (value !== "") body[name] = Number(value);
    }
    if (Object.keys(body).length === 0) return;

    try {
      await api(`/cameras/${cameraId}/parameters`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      showError(null);
      syncControls = true;
    } catch (error) {
      showError(`Could not apply parameters: ${error.message}`);
    }
  });

  el("stream").addEventListener("error", () => {
    showError("Stream interrupted; retrying.");
    setTimeout(restartStream, RECONNECT_MS);
  });
}

async function main() {
  bind();
  try {
    await loadColormaps();
    await loadCameras();
    showError(null);
  } catch (error) {
    showError(`Could not reach the server: ${error.message}`);
  }
}


document.addEventListener("DOMContentLoaded", () => {
    const stream = document.getElementById("stream");
    const coordinates = document.getElementById("coordinates");
    const upscale = Number.parseFloat(el("upscale-value").textContent);

    if (!stream || !coordinates) {
        console.error("Missing video");
        return;
    }

    stream.addEventListener("mousemove", (event) => {
        const rect = stream.getBoundingClientRect();
        x = Math.floor((event.clientX - rect.left) / upscale);
        y = Math.floor((event.clientY - rect.top) / upscale);

        coordinates.textContent =
            `X: ${x}, ` +
            `Y: ${y}`;
    });

    stream.addEventListener("mouseleave", () => {
        x = 0;
        y = 0;
        coordinates.textContent = "X: —, Y: —";
    });
});

main();
