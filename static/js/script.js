import {
  FilesetResolver,
  HandLandmarker,
  FaceLandmarker,
  PoseLandmarker,
  DrawingUtils,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

let handLandmarker, faceLandmarker, poseLandmarker;

// --- CONEXIONES ENTRE LANDMARKS ---
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20], [0, 17], [0, 9]
];

// Solo conexiones hasta los tobillos (landmark 25)
const POSE_CONNECTIONS = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24],
  [23, 24], [23, 25], [24, 25] // hasta tobillos, no pies
];

// --- FUNCIONES DE DIBUJO ---
function drawLandmarks(landmarks, color) {
  if (!landmarks) return;
  ctx.fillStyle = color;
  for (const landmark of landmarks) {
    if (!landmark) continue;
    ctx.beginPath();
    ctx.arc(landmark.x * canvas.width, landmark.y * canvas.height, 5, 0, 2 * Math.PI);
    ctx.fill();
  }
}

function drawConnections(landmarks, connections) {
  if (!landmarks) return;
  ctx.strokeStyle = "white";
  ctx.lineWidth = 2;
  for (const [startIdx, endIdx] of connections) {
    const start = landmarks[startIdx];
    const end = landmarks[endIdx];
    if (start && end) {
      ctx.beginPath();
      ctx.moveTo(start.x * canvas.width, start.y * canvas.height);
      ctx.lineTo(end.x * canvas.width, end.y * canvas.height);
      ctx.stroke();
    }
  }
}

// --- CONFIGURACIÓN DE LA CÁMARA ---
async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;
  await new Promise(resolve => {
    video.onloadedmetadata = () => resolve();
  });
  await video.play();
}

// --- CARGA DE MODELOS DE MEDIAPIPE ---
async function loadModels() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
  );

  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: "/static/models/hand_landmarker.task" },
    runningMode: "VIDEO",
    numHands: 2,
  });

  faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: "/static/models/face_landmarker.task" },
    outputFaceBlendshapes: false,
    runningMode: "VIDEO",
  });

  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: "/static/models/pose_landmarker_lite.task" },
    runningMode: "VIDEO",
  });
}

// --- LOOP PRINCIPAL DE PREDICCIÓN ---
async function predictFrame() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  const now = performance.now();

  try {
    const handResult = await handLandmarker.detectForVideo(video, now);
    const faceResult = await faceLandmarker.detectForVideo(video, now);
    const poseResult = await poseLandmarker.detectForVideo(video, now);

    let poseLandmarks = poseResult.landmarks?.[0] || [];
    let hands = handResult.landmarks || [];

    // --- FILTRAR LANDMARKS DE LA CARA Y MANOS EN POSE ---
    const ignoredPosePoints = new Set([
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,     // cara
      15, 16, 17, 18, 19, 20, 21, 22        // muñecas, dedos
    ]);

    // --- TAMBIÉN ELIMINAR PUNTOS DESPUÉS DE LA CADERA (solo hasta 25) ---
    const cleanPose = poseLandmarks.map((p, i) =>
      (ignoredPosePoints.has(i) || i > 25) ? null : p
    );

    // --- DIBUJAR POSE ---
    drawConnections(cleanPose, POSE_CONNECTIONS);
    drawLandmarks(cleanPose, "blue");

    // --- DIBUJAR MANOS ---
    for (const hand of hands) {
      drawConnections(hand, HAND_CONNECTIONS);
      drawLandmarks(hand, "red");
    }

    // --- DIBUJAR CARA (solo landmarks de faceLandmarker, no de pose) ---
    for (const face of faceResult.faceLandmarks || []) {
      drawLandmarks(face, "green");
    }

    // --- CONECTAR CODOS A PALMAS (heurística por cercanía) ---
    const codoIzq = poseLandmarks[13];
    const codoDer = poseLandmarks[14];

    for (const hand of hands) {
      const palma = hand[0];
      if (codoIzq && codoDer && palma) {
        const distIzq = Math.hypot(palma.x - codoIzq.x, palma.y - codoIzq.y);
        const distDer = Math.hypot(palma.x - codoDer.x, palma.y - codoDer.y);
        const cercaIzq = distIzq < distDer;

        ctx.strokeStyle = "white";
        ctx.beginPath();
        ctx.moveTo(
          (cercaIzq ? codoIzq.x : codoDer.x) * canvas.width,
          (cercaIzq ? codoIzq.y : codoDer.y) * canvas.height
        );
        ctx.lineTo(palma.x * canvas.width, palma.y * canvas.height);
        ctx.stroke();
      }
    }

  } catch (err) {
    console.error("Error en inferencia:", err.message);
  }

  requestAnimationFrame(predictFrame);
}

// --- INICIAR TODO ---
(async () => {
  try {
    await setupCamera();
    await loadModels();
    predictFrame();
  } catch (err) {
    console.error("Error al iniciar:", err);
  }
})();