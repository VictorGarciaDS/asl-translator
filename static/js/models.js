import {
  FilesetResolver,
  HandLandmarker,
  FaceLandmarker,
  PoseLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

export async function loadModels() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
  );

  const handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: "/static/models/hand_landmarker.task" },
    runningMode: "VIDEO",
    numHands: 2,
  });

  const faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: "/static/models/face_landmarker.task" },
    outputFaceBlendshapes: false,
    runningMode: "VIDEO",
  });

  const poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: "/static/models/pose_landmarker_lite.task" },
    runningMode: "VIDEO",
  });

  return { handLandmarker, faceLandmarker, poseLandmarker };
}