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

/**
 * Procesa un frame y devuelve todos los landmarks detectados
 */
async function processFrame(video) {
  // Declarando primero las variables
  const timestamp = performance.now();
  const [handsResult, faceResult, poseResult] = await Promise.all([
    handLandmarker.detectForVideo(video, timestamp),
    faceLandmarker.detectForVideo(video, timestamp),
    poseLandmarker.detectForVideo(video, timestamp)
  ]);

  const hands = handsResult?.landmarks || [];
  const poseLandmarks = poseResult.landmarks?.[0] || [];

  ignoredPosePoints = new Set([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,     // cara
    15, 16, 17, 18, 19, 20, 21, 22        // muñecas, dedos
  ]);

  cleanPose = poseLandmarks.map((p, i) =>
    (ignoredPosePoints.has(i) || i > 24) ? null : p
  );

  // 🧠 Cuello (calculado a partir de cara y cuerpo)
  if (poseLandmarks.length > 0 && faceResult.faceLandmarks.length > 0) {
    NECK_POINTS = calcularNeckPoints(ctx, poseLandmarks, faceResult.faceLandmarks[0]);
  }

  return { hands, faceResult, poseLandmarks, cleanPose, NECK_POINTS };
}

// 🖼️ Dibujar primero el video
function drawVideoFrame(ctx, canvas, video) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
}

// 🖐️ Manos
function drawHands(ctx, hands) {
  for (const hand of hands) {
    drawConnections(ctx, hand, conns.HAND_CONNECTIONS);
    drawLandmarks(ctx, hand, "red");
  }
}

// 🕺 Cuerpo
function drawPose(ctx, cleanPose) {
  drawConnections(ctx, cleanPose, conns.POSE_CONNECTIONS);
  drawLandmarks(ctx, cleanPose, "blue");
}

// 🧠 Cuello
function drawNeck(ctx, NECK_POINTS, allLandmarks) {
  if (NECK_POINTS.length > 0) {
    drawConnections(ctx, NECK_POINTS, [
      [0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7]
    ]);
    drawLandmarks(ctx, NECK_POINTS, "blue");
    allLandmarks.push({ tipo: "cuello", landmarks: NECK_POINTS });
  }
}

function renderFrame({ hands, faceResult, poseLandmarks, cleanPose, NECK_POINTS }) {
  drawVideoFrame(ctx, canvas, video)
  drawHands(ctx, hands);
  drawPose(ctx, cleanPose);
  drawNeck(ctx, NECK_POINTS, allLandmarks);

  try {
    // --- LANDMARKS FACIALES (REGIONES) ---
    for (const face of faceResult.faceLandmarks || []) {
      forehead = [
        face[67], face[109], face[10], face[338], face[297],
        face[299], face[9], face[69]
      ];
      ceja_izquierda = [
        face[276], face[283], face[282], face[295], face[285],
        face[300], face[293], face[334], face[296], face[336]
      ];
      ceja_derecha = [
        face[46], face[53], face[52], face[65], face[55],
        face[70], face[63], face[105], face[66], face[107]
      ];
      sien_izquierda = [
        face[389], face[251], face[301], face[383], face[372], face[264]
      ];
      sien_derecha = [
        face[162], face[21], face[71], face[156], face[143], face[34]
      ];
      ojo_izquierdo = [
        face[263], face[249], face[390], face[373], face[374],
        face[380], face[381], face[382], face[362], face[466],
        face[388], face[387], face[386], face[385], face[384],
        face[398]
      ];
      ojo_derecho = [
        face[33], face[7], face[163], face[144], face[145],
        face[153], face[154], face[155], face[133], face[246],
        face[161], face[160], face[159], face[158], face[157],
        face[173]
      ];
      iris_izquierdo = [
        ...face.slice(474, 477)
      ];
      iris_derecho = [
        ...face.slice(469, 472)
      ]
      nariz_izquierda = [
        face[6], face[197], face[195], face[5], face[4],//eje del tabique
        face[102], face[115], face[220], face[45],// eje transversal de la nariz
        face[122], face[188], face[114], face[217], face[126],
        face[142], face[129]
      ];
      nariz_derecha = [
        face[6], face[197], face[195], face[5], face[4],//eje del tabique
        face[275], face[440], face[344], face[331],// eje transversal de la nariz
        face[358], face[371], face[355], face[437], face[343],
        face[412], face[351]
      ];
      nariz_baja = [
        face[102], face[115], face[220], face[45],// eje transversal de la nariz
        face[275], face[440], face[344], face[331],// eje transversal de la nariz
        face[129], face[98], face[97], face[2], face[326], face[327], face[358]// base de la nariz
      ];
      mejilla_izquierda = [
        face[376], face[411], face[427], face[434], face[364],
        face[367], face[435], face[401]
      ];
      mejilla_derecha = [
        face[147], face[187], face[207], face[214], face[135],
        face[138], face[215], face[177]
      ];
      boca = [
        face[61], face[146], face[91], face[181], face[84],
        face[17], face[314], face[405], face[321], face[375],
        face[291], face[185], face[40], face[39], face[37],
        face[0], face[267], face[269], face[270], face[409],
        face[78], face[95], face[88], face[178], face[87],
        face[14], face[317], face[402], face[318], face[324],
        face[308], face[191], face[80], face[81], face[82],
        face[13], face[312], face[311], face[310], face[415],
        face[16], face[315], face[404], face[320], face[307],
        face[306], face [408], face[304], face[303], face[302],
        face[11], face[72], face[73], face[74], face[184],
        face[76], face[77], face[90], face[180], face[85]
      ];
      menton = [
        face[32], face[194], face[83], face[18], face[313],
        face[418], face[262], face[369], face[377], face[152],
        face[148], face[140]
      ];

      drawLandmarks(ctx, forehead, "green");
      drawConnections(ctx, face, conns.FOREHEAD_CONNECTIONS);
      drawLandmarks(ctx, ceja_izquierda, "green");
      drawConnections(ctx, face, conns.LEFT_EYEBROW_CONNECTIONS);
      drawLandmarks(ctx, ceja_derecha, "green");
      drawConnections(ctx, face, conns.RIGHT_EYEBROW_CONNECTIONS);
      drawLandmarks(ctx, sien_izquierda, "green");
      drawConnections(ctx, face, conns.LEFT_TEMPLE_CONNECTIONS);
      drawLandmarks(ctx, sien_derecha, "green");
      drawConnections(ctx, face, conns.RIGHT_TEMPLE_CONNECTIONS);
      drawLandmarks(ctx, ojo_izquierdo, "green");
      drawConnections(ctx, face, conns.LEFT_EYE_CONNECTIONS);
      drawLandmarks(ctx, ojo_derecho, "green");
      drawConnections(ctx, face, conns.RIGHT_EYE_CONNECTIONS);
      drawLandmarks(ctx, iris_izquierdo, "green");
      drawConnections(ctx, face, conns.LEFT_IRIS_CONNECTIONS);
      drawLandmarks(ctx, iris_derecho, "green");
      drawConnections(ctx, face, conns.RIGHT_IRIS_CONNECTIONS);
      drawLandmarks(ctx, nariz_izquierda, "green");
      drawConnections(ctx, face, conns.LEFT_NOSE);
      drawLandmarks(ctx, nariz_derecha, "green");
      drawConnections(ctx, face, conns.RIGHT_NOSE);
      drawLandmarks(ctx, nariz_baja, "green");
      drawConnections(ctx, face, conns.LOW_NOSE);
      drawLandmarks(ctx, mejilla_izquierda, "green");
      drawConnections(ctx, face, conns.LEFT_CHEEK_CONNECTIONS);
      drawLandmarks(ctx, mejilla_derecha, "green");
      drawConnections(ctx, face, conns.RIGHT_CHEEK_CONNECTIONS);
      drawLandmarks(ctx, boca, "green");
      drawConnections(ctx, face, conns.LIPS_CONNECTIONS);
      drawLandmarks(ctx, menton, "green");
      drawConnections(ctx, face, conns.CHIN_CONNECTIONS);
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

  if (++frameCounter >= FRAMES_PARA_ENVIAR) {
    const faceLandmarks = faceResult.faceLandmarks?.[0] || null;

    cleanPose = poseLandmarks
      .map((p, i) => ({ index: i, point: p })) // mantener índice original por si lo necesitas
      .filter(({ index }) => !ignoredPosePoints.has(index) && index <= 24)
      .map(({ point }) => point); // solo devolver los puntos válidos

    enviarLandmarksAlServidor({
      timestamp: Date.now(),
      forehead: forehead,
      left_eyebrow: ceja_izquierda,
      right_eyebrow: ceja_derecha,
      sien_izquierda: sien_izquierda,
      sien_derecha: sien_derecha,
      ojo_izquierdo: ojo_izquierdo,
      ojo_derecho: ojo_derecho,
      iris_izquierdo: iris_izquierdo,
      iris_derecho: iris_derecho,
      nariz_izquierda: nariz_izquierda,
      nariz_derecha: nariz_derecha,
      nariz_baja: nariz_baja,
      left_cheek: mejilla_izquierda,
      right_cheek: mejilla_derecha,
      boca: boca,
      menton: menton,
      neck: NECK_POINTS,
      pose: cleanPose,
      hands: hands,
    });
    frameCounter = 0;
  }
}

// --- LOOP PRINCIPAL DE PREDICCIÓN ---
async function predictFrame() {
  if (!handLandmarker || !faceLandmarker || !poseLandmarker) return;

  const results = await processFrame(video);
  renderFrame(results);

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