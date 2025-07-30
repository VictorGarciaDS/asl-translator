import { drawLandmarks, drawConnections } from "./drawing.js";

// Esta función calcula los puntos del cuello (NECK_POINTS) a partir de la pose y cara detectada
export function calcularNeckPoints(ctx, pose, face) {
  if (!pose.length || !face.length) return [];

  const menton = face[152];
  const baseCuello = {
    x: (pose[11].x + pose[12].x) / 2,
    y: (pose[11].y + pose[12].y) / 2,
    z: (pose[11].z + pose[12].z) / 2,
  };

  const traquea1 = {
    x: (baseCuello.x + 2 * menton.x) / 3,
    y: (baseCuello.y + 2 * menton.y) / 3,
    z: (baseCuello.z + 2 * menton.z) / 3,
  };

  const traquea2 = {
    x: (2 * baseCuello.x + menton.x) / 3,
    y: (2 * baseCuello.y + menton.y) / 3,
    z: (2 * baseCuello.z + menton.z) / 3,
  };

  const offset_face = Math.hypot(face[377].x - face[148].x, face[377].y - face[148].y);
  const dir_face = {
    x: (face[377].x - face[148].x) / offset_face,
    y: (face[377].y - face[148].y) / offset_face,
  };

  const traquea1_izq = {
    x: traquea1.x - dir_face.x * offset_face / 2,
    y: traquea1.y - dir_face.y * offset_face / 2,
    z: traquea1.z,
  };

  const traquea1_der = {
    x: traquea1.x + dir_face.x * offset_face / 2,
    y: traquea1.y + dir_face.y * offset_face / 2,
    z: traquea1.z,
  };

  const offset_pose = Math.hypot(pose[12].x - pose[11].x, pose[12].y - pose[11].y);
  const dir_pose = {
    x: (pose[11].x - pose[12].x) / offset_pose,
    y: (pose[11].y - pose[12].y) / offset_pose,
  };

  const baseCuello_izq = {
    x: baseCuello.x - dir_pose.x * offset_pose / 20,
    y: baseCuello.y - dir_pose.y * offset_pose / 20,
    z: baseCuello.z,
  };

  const baseCuello_der = {
    x: baseCuello.x + dir_pose.x * offset_pose / 20,
    y: baseCuello.y + dir_pose.y * offset_pose / 20,
    z: baseCuello.z,
  };

  const dir_bisectriz_raw = {
    x: dir_face.x + dir_pose.x,
    y: dir_face.y + dir_pose.y,
  };

  const norm_bisectriz = Math.hypot(dir_bisectriz_raw.x, dir_bisectriz_raw.y);
  const dir_bisectriz = {
    x: dir_bisectriz_raw.x / norm_bisectriz,
    y: dir_bisectriz_raw.y / norm_bisectriz,
  };

  const offset_bisectriz = (offset_face + offset_pose) / 20;

  const traquea2_izq = {
    x: traquea2.x - dir_bisectriz.x * offset_bisectriz,
    y: traquea2.y - dir_bisectriz.y * offset_bisectriz,
    z: traquea2.z,
  };

  const traquea2_der = {
    x: traquea2.x + dir_bisectriz.x * offset_bisectriz,
    y: traquea2.y + dir_bisectriz.y * offset_bisectriz,
    z: traquea2.z,
  };

  const NECK_POINTS = [
    face[377],          // 0
    traquea1_der,       // 1
    traquea2_der,       // 2
    baseCuello_der,     // 3
    baseCuello_izq,     // 4
    traquea2_izq,       // 5
    traquea1_izq,       // 6
    face[148],          // 7
  ];

  const NECK_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3],
    [3, 4], [4, 5], [5, 6],
    [6, 7]
  ];

  drawLandmarks(ctx, NECK_POINTS, "blue");
  drawConnections(ctx, NECK_POINTS, NECK_CONNECTIONS);

  return NECK_POINTS;
}