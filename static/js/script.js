import {
  FilesetResolver,
  HandLandmarker,
  FaceLandmarker,
  PoseLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

let handLandmarker, faceLandmarker, poseLandmarker;

// --- MANUAL CONNECTIONS ---
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],       // pulgar
  [0, 5], [5, 6], [6, 7], [7, 8],       // índice
  [5, 9], [9, 10], [10, 11], [11, 12],  // medio
  [9, 13], [13, 14], [14, 15], [15, 16],// anular
  [13, 17], [17, 18], [18, 19], [19, 20],// meñique
  [0, 17], [0, 9]                       // palma
];

const POSE_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 7],
  [0, 4], [4, 5], [5, 6], [6, 8],
  [9, 10],
  [11, 12], [11, 13], [13, 15],
  [12, 14], [14, 16],
  [15, 17], [16, 18],
  [11, 23], [12, 24],
  [23, 24], [23, 25], [24, 26],
  [25, 27], [26, 28],
  [27, 29], [28, 30],
  [29, 31], [30, 32]
];

// FACE_OVAL: solo contorno exterior del rostro
const FACE_OVAL = [
  [10, 338], [338, 297], [297, 332], [332, 284], [284, 251], [251, 389], [389, 356],
  [356, 454], [454, 323], [323, 361], [361, 288], [288, 397], [397, 365],
  [365, 379], [379, 378], [378, 400], [400, 377], [377, 152], [152, 148],
  [148, 176], [176, 149], [149, 150], [150, 136], [136, 172], [172, 58],
  [58, 132], [132, 93], [93, 234], [234, 127], [127, 162], [162, 21],
  [21, 54], [54, 103], [103, 67], [67, 109], [109, 10]  // cerrar óvalo
];

// --- DRAW FUNCTIONS ---
function drawLandmarks(landmarks, color) {
  if (!landmarks) return;
  ctx.fillStyle = color;
  for (const landmark of landmarks) {
    ctx.beginPath();
    ctx.arc(landmark.x * canvas.width, landmark.y * canvas.height, 5, 0, 2 * Math.PI);
    ctx.fill();
  }
}

function drawConnections(landmarks, connections, color) {
  if (!landmarks) return;
  ctx.strokeStyle = color;
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

// --- CAMERA ---
async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;
  await new Promise(resolve => {
    video.onloadedmetadata = () => resolve();
  });
  await video.play();
}

// --- LOAD MODELS ---
async function loadModels() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
  );

  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "/static/models/hand_landmarker.task",
    },
    runningMode: "VIDEO",
    numHands: 2,
  });

  faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "/static/models/face_landmarker.task",
    },
    outputFaceBlendshapes: false,
    runningMode: "VIDEO",
  });

  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "/static/models/pose_landmarker_lite.task",
    },
    runningMode: "VIDEO",
  });
}

// --- MAIN LOOP ---
async function predictFrame() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  const now = performance.now();

  try {
    const handResult = await handLandmarker.detectForVideo(video, now);
    const faceResult = await faceLandmarker.detectForVideo(video, now);
    const poseResult = await poseLandmarker.detectForVideo(video, now);

    if (handResult.landmarks) {
      for (const hand of handResult.landmarks) {
        drawConnections(hand, HAND_CONNECTIONS, "red");
        drawLandmarks(hand, "red");
      }
    }

    if (faceResult.faceLandmarks) {
      for (const face of faceResult.faceLandmarks) {
        drawConnections(face, FACE_OVAL, "green");
        drawLandmarks(face, "green");
      }
    }

    if (poseResult.landmarks) {
      for (const pose of poseResult.landmarks) {
        drawConnections(pose, POSE_CONNECTIONS, "blue");
        drawLandmarks(pose, "blue");
      }
    }
  } catch (err) {
    console.error("Error en inferencia:", err.message);
  }

  requestAnimationFrame(predictFrame);
}

// --- INIT ---
(async () => {
  try {
    await setupCamera();
    await loadModels();
    predictFrame();
  } catch (err) {
    console.error("Error al iniciar:", err);
  }
})();