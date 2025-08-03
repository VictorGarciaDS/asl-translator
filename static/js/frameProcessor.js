/**
 * Procesa un frame y devuelve todos los landmarks detectados
 */

import { calcularNeckPoints } from "./neckPoints.js";

export async function processFrame(video, ctx, landmarkers) {
    const { handLandmarker, faceLandmarker, poseLandmarker } = landmarkers;
    const timestamp = performance.now();

    const [handsResult, faceResult, poseResult] = await Promise.all([
        handLandmarker.detectForVideo(video, timestamp),
        faceLandmarker.detectForVideo(video, timestamp),
        poseLandmarker.detectForVideo(video, timestamp)
      ]);
    
      const hands = handsResult?.landmarks || [];
      const poseLandmarks = poseResult.landmarks?.[0] || [];
    
      let ignoredPosePoints = new Set([
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,     // cara
        15, 16, 17, 18, 19, 20, 21, 22        // muñecas, dedos
      ]);
    
      let cleanPose = poseLandmarks.map((p, i) =>
        (ignoredPosePoints.has(i) || i > 24) ? null : p
      );
    
      // 🧠 Cuello (calculado a partir de cara y cuerpo)
      let NECK_POINTS = [];
      if (poseLandmarks.length > 0 && faceResult.faceLandmarks.length > 0) {
        NECK_POINTS = calcularNeckPoints(ctx, poseLandmarks, faceResult.faceLandmarks[0]);
      }
    
      return { hands, faceResult, poseLandmarks, cleanPose, NECK_POINTS, ignoredPosePoints };
}