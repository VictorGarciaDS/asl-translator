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

const POSE_CONNECTIONS = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24],
  [23, 24], [23, 25], [24, 25]
];

// --- FUNCIONES DE DIBUJO ---
function drawLandmarks(landmarks, color) {
  if (!landmarks) return;
  ctx.fillStyle = color;
  for (const landmark of landmarks) {
    if (!landmark) continue;
    ctx.beginPath();
    ctx.arc(landmark.x * canvas.width, landmark.y * canvas.height, 3, 0, 2 * Math.PI);
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

    const poseLandmarks = poseResult.landmarks?.[0] || [];
    const hands = handResult.landmarks || [];

    // --- FILTRAR LANDMARKS DE LA POSE ---
    const ignoredPosePoints = new Set([
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,     // cara
      15, 16, 17, 18, 19, 20, 21, 22        // muñecas, dedos
    ]);
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

    // --- LANDMARKS FACIALES (REGIONES) ---
    for (const face of faceResult.faceLandmarks || []) {
      const cejas = [
        // Ceja derecha
        face[46], face[53], face[52], face[65], face[55],
        face[70], face[63], face[105], face[66], face[107],
        // Ceja izquierda
        face[276], face[283], face[282], face[295], face[285],
        face[300], face[293], face[334], face[296], face[336],
      ];
      const ojos = [
        // Ojo derecho
        face[33], face[7], face[163], face[144], face[145],
        face[153], face[154], face[155], face[133], face[246],
        face[161], face[160], face[159], face[158], face[157], face[173],
        // Ojo izquierdo
        face[263], face[249], face[390], face[373], face[374],
        face[380], face[381], face[382], face[362], face[466],
        face[388], face[387], face[386], face[385], face[384], face[398],
      ];
      const nariz = [
        face[168], face[6], face[197], face[195], face[5],
        face[4], face[1], face[19], face[94], face[2],//eje del tabique
        face[115], face[220], face[45], face[275], face[440], face[344],// eje transversal de la nariz
        face[98], face[97], face[326], face[327], face[294],
        face[129], face[64], face[49], face[209], face[126],
        face[217], face[174], face[196], face[419], face[399],
        face[437], face[355], face[429], face[279], face[358],
        face[294]
      ];
      const boca = [
        face[61], face[146], face[91], face[181], face[84],
        face[17], face[314], face[405], face[321], face[375],
        face[291], face[185], face[40], face[39], face[37],
        face[0], face[267], face[269], face[270], face[409],
        face[78], face[95], face[88], face[178], face[87],
        face[14], face[317], face[402], face[318], face[324],
        face[308], face[191], face[80], face[81], face[82],
        face[13], face[312], face[311], face[310], face[415],
        face[16], face[315], face[404], face[320], face[307],
        face[306], face [406], face[304], face[303], face[302],
        face[11], face[72], face[73], face[74], face[184],
        face[76], face[77], face[90], face[180], face[85]
      ];
      const menton = [
        face[32], face[194], face[83], face[18], face[313],
        face[418], face[262], face[369], face[377], face[152],
        face[148], face[140]
      ];

      drawLandmarks(cejas, "#cc00ff");  // púrpura
      drawLandmarks(ojos, "#00ffff");   // cian
      drawLandmarks(nariz, "#ffcc00");  // amarillo
      drawLandmarks(boca, "#ff6600");   // naranja
      drawLandmarks(menton, "#00ff00"); // verde
    }

    // --- CONECTAR CODOS A PALMAS (heurística) ---
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