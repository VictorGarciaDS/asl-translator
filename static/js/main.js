import { setupCamera } from "./camera.js";
import { loadModels } from "./models.js";
import { resizeCanvas } from "./utils.js";
import { processFrame } from "./frameProcessor.js";
import { renderFrame } from "./frameRenderer.js";

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

window.addEventListener("resize", () => resizeCanvas(canvas));
resizeCanvas(canvas);

let frameCounter = 0;
const FRAMES_PARA_ENVIAR = 2; // Envía cada 10 frames
let handLandmarker, faceLandmarker, poseLandmarker;
let NECK_POINTS = [];
let ignoredPosePoints = new Set([]);
let cleanPose = [];

// Opcional: vuelve a ajustar si el usuario gira la pantalla o cambia tamaño
window.addEventListener("resize", resizeCanvas);

let forehead = [];
let ceja_izquierda = [];
let ceja_derecha = [];
let sien_izquierda = [];
let sien_derecha = [];
let ojo_izquierdo = [];
let ojo_derecho = [];
let iris_izquierdo = [];
let iris_derecho = [];
let nariz_izquierda = [];
let nariz_derecha = [];
let nariz_baja = [];
let mejilla_izquierda = [];
let mejilla_derecha = [];
let boca = [];
let menton = [];
let allLandmarks = [];

// --- LOOP PRINCIPAL DE PREDICCIÓN ---
async function predictFrame() {
  if (!handLandmarker || !faceLandmarker || !poseLandmarker) return;

  const results = await processFrame(video, ctx, { handLandmarker, faceLandmarker, poseLandmarker });
  renderFrame(ctx, canvas, video, results, allLandmarks, frameCounter, ignoredPosePoints);

  requestAnimationFrame(predictFrame);
}

// --- INICIAR TODO ---
(async () => {
  try {
    await setupCamera(video);
    const models = await loadModels();

    handLandmarker = models.handLandmarker;
    faceLandmarker = models.faceLandmarker;
    poseLandmarker = models.poseLandmarker;

    // ✅ Muy importante para evitar errores como "Task not initialized with image mode"
    await handLandmarker.setOptions({ runningMode: "VIDEO" });
    await faceLandmarker.setOptions({ runningMode: "VIDEO" });
    await poseLandmarker.setOptions({ runningMode: "VIDEO" });

    predictFrame();
  } catch (err) {
    console.error("Error al iniciar:", err);
  }
})();