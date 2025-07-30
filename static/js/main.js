import { setupCamera } from "./camera.js";
import { loadModels } from "./models.js";
import { drawLandmarks, drawConnections } from "./drawing.js";
import { resizeCanvas, enviarLandmarksAlServidor } from "./utils.js";
import * as conns from "./connections.js";
import { calcularNeckPoints } from "./neckPoints.js";

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

window.addEventListener("resize", () => resizeCanvas(canvas));
resizeCanvas(canvas);

let frameCounter = 0;
const FRAMES_PARA_ENVIAR = 10;
let handLandmarker, faceLandmarker, poseLandmarker;

async function predictFrame() {
  if (!handLandmarker || !faceLandmarker || !poseLandmarker) return;

  const timestamp = performance.now();

  const [handsResult, faceResult, poseResult] = await Promise.all([
    handLandmarker.detectForVideo(video, timestamp),
    faceLandmarker.detectForVideo(video, timestamp),
    poseLandmarker.detectForVideo(video, timestamp)
  ]);

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  let allLandmarks = [];

  // 🖐️ Manos
  if (handsResult?.handedness?.length > 0 && handsResult?.landmarks?.length > 0) {
    handsResult.handedness.forEach((hand, i) => {
      const landmarks = handsResult.landmarks[i];
      drawLandmarks(ctx, landmarks, "red");
      drawConnections(ctx, landmarks, conns.HAND_CONNECTIONS);
      allLandmarks.push({ tipo: "mano", lado: hand[0].categoryName, landmarks });
    });
  }

  // 😊 Cara
  if (faceResult?.faceLandmarks?.length > 0) {
    const landmarks = faceResult.faceLandmarks[0];
    drawLandmarks(ctx, landmarks, "green");
    drawConnections(ctx, landmarks, conns.FACE_CONNECTIONS);
    allLandmarks.push({ tipo: "cara", landmarks });
  }

  // 🕺 Cuerpo
  if (poseResult?.landmarks?.length > 0) {
    const landmarks = poseResult.landmarks;
    drawLandmarks(ctx, landmarks, "orange");
    drawConnections(ctx, landmarks, conns.POSE_CONNECTIONS);
    allLandmarks.push({ tipo: "cuerpo", landmarks });
  }

  // 🧠 Cuello
  let NECK_POINTS = [];
  if (poseResult?.landmarks?.length > 0 && faceResult?.faceLandmarks?.length > 0) {
    NECK_POINTS = calcularNeckPoints(ctx, poseResult.landmarks, faceResult.faceLandmarks[0]);
    allLandmarks.push({ tipo: "cuello", landmarks: NECK_POINTS });
  }

  // 📤 Enviar landmarks al servidor
  if (++frameCounter >= FRAMES_PARA_ENVIAR) {
    enviarLandmarksAlServidor(allLandmarks);
    frameCounter = 0;
  }

  requestAnimationFrame(predictFrame);
}

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