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

async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;
  await new Promise(resolve => {
    video.onloadedmetadata = () => resolve();
  });
  await video.play();
}

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

function drawLandmarks(landmarks, color) {
  if (!landmarks) return;
  ctx.fillStyle = color;
  for (const landmark of landmarks) {
    ctx.beginPath();
    ctx.arc(landmark.x * canvas.width, landmark.y * canvas.height, 5, 0, 2 * Math.PI);
    ctx.fill();
  }
}

async function predictFrame() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  const now = performance.now(); // ✅ Usamos performance.now() como timestamp (más preciso que Date.now())

  try {
    const handResult = await handLandmarker.detectForVideo(video, now);
    const faceResult = await faceLandmarker.detectForVideo(video, now);
    const poseResult = await poseLandmarker.detectForVideo(video, now);

    console.log('Manos detectadas:', handResult.landmarks?.length || 0);
    console.log('Rostros detectados:', faceResult.faceLandmarks?.length || 0);
    console.log('Pose detectada:', poseResult.landmarks ? 1 : 0);

    if (handResult.landmarks) {
      for (const hand of handResult.landmarks) drawLandmarks(hand, "red");
    }
    if (faceResult.faceLandmarks) {
      for (const face of faceResult.faceLandmarks) drawLandmarks(face, "green");
    }
    if (poseResult.landmarks) {
      for (const personLandmarks of poseResult.landmarks) {
        drawLandmarks(personLandmarks, "blue");
      }
    }
  } catch (err) {
    console.error("Error en inferencia:", err.message);
  }

  requestAnimationFrame(predictFrame);
}

(async () => {
  try {
    await setupCamera();
    await loadModels();
    predictFrame();
  } catch (err) {
    console.error("Error al iniciar:", err);
  }
})();