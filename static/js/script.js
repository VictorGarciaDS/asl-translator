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
  [11, 23], [12, 24], [23, 24]
];

const FOREHEAD_CONNECTIONS = [
  [67, 109], [109, 10], [10, 338], [338, 297], [297, 299], [299,9], [9, 69], [69, 67]
];

const EYEBROWS_CONNECTIONS = [
  [46, 53], [53, 52], [52, 65], [65, 55], [55, 107], [66, 107], [105, 66], [63, 105], [70, 63], [46, 70],// Ceja derecha
  [276, 283], [283, 282], [282, 295], [295, 285], [285, 336], [336, 296], [334, 296], [293, 334], [300, 293], [300, 276]// Ceja izquierda
];

const EYES_CONNECTIONS = [
  [33, 7], [7, 163], [163, 144], [144, 145], [145, 153], [153, 154], [154, 155], [155, 133], [173, 133],
  [157, 173], [158, 157], [159, 158], [160, 159], [161, 160], [246, 161], [33, 246],// Ojo derecho
  [263, 249], [249, 390], [390, 373], [373, 374], [374, 380], [380, 381], [381, 382], [382, 362],
  [263, 466], [466, 388], [388, 387], [387, 386], [386, 385], [385, 384], [384, 398], [398, 362]
];

const IRIS_CONNECTIONS = [
  [469, 470], [470, 471], [471, 472], [472, 469], // Iris derecho
  [474, 475], [475, 476], [476, 477], [477, 474] // Iris izquierdo
];

const TEMPLES_CONNECTIONS = [
  [162, 21], [21, 71], [71, 156], [156, 143], [143, 34], [34, 162],// Sien derecha
  [389, 251], [251, 301], [301, 383], [383, 372], [372, 264], [264, 389] // Sien izquierda
];

const NOSE_CONNECTIONS = [
  [6, 197], [197, 195], [195, 5], [5, 4], //eje del tabique
  [4, 275], [275, 440], [440, 344], [344, 331],//eje transversal derecho de la nariz
  [331, 358], [358, 371], [371, 355], [355, 437], [437, 343], [343, 412], [412, 351], [351, 6], //lateral derecho de la nariz
  [6, 122], [122, 188], [188 ,114], [114, 217], [126, 217], [126, 142], [142, 129], [129, 102], //lateral izquierdo de la nariz
  [102, 115], [115, 220], [220, 45], [45, 4],//eje transversal izquierdo de la nariz
  [129, 98], [98, 97], [97, 2], [2, 326], [326, 327], [327, 358]// base de la nariz
];

const LIPS_CONNECTIONS = [
  [61, 146], [146, 91], [91, 181], [181, 84], [84, 17], [17, 314], [314, 405], [405, 321], [321, 375], [375, 291],
  [409, 291],[270, 409], [269, 270], [267, 269], [0, 267], [37, 0], [39, 37], [40, 39], [185, 40], [61, 185], // Contorno exterior de los labios
  [78, 95], [95, 88], [88, 178], [178, 87], [87, 14], [14, 317], [317, 402], [402, 318], [318, 324], [324, 308],
  [415, 308], [310, 415], [311, 310], [312, 311], [13, 312], [82, 13], [81, 82], [80, 81], [191, 80], [78, 191],// Contorno interior de los labios
  [16, 315], [315, 404], [404, 320], [320, 307], [307, 306], [306, 408], [408, 304], [304, 303], [303, 302],
  [302, 11], [11, 72], [72, 73], [73, 74], [74, 184], [184, 76], [76, 77], [77, 90], [90, 180], [180, 85], [85, 16] //Contorno intermedio de los labios
];

const CHECKS_CONNECTIONS = [
  [147, 187], [187, 207], [207, 214], [214, 135], [135, 138], [138, 215], [215, 177], [177, 147],// Mejilla derecha
  [376, 411], [411, 427], [427, 434], [434, 364], [364, 367], [367, 435], [435, 401], [401, 376] // Mejilla izquierda
];

const CHIN_CONNECTIONS = [
  [32, 194], [194, 83], [83, 18], [18, 313], [313, 418],
  [418, 262], [262, 369], [369, 377], [377, 152], [152, 148], [148, 140], [140, 32]
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

    // --- PUNTOS DE LA TRÁQUEA (entre mentón y cuello) ---
    if (poseLandmarks.length > 0 && faceResult.faceLandmarks?.length > 0) {
      const pose = poseLandmarks;
      const face = faceResult.faceLandmarks[0]; // primera cara detectada

      const menton = face[152];

      const baseCuello = {
        x: (pose[11].x + pose[12].x) / 2,
        y: (pose[11].y + pose[12].y) / 2,
        z: (pose[11].z + pose[12].z) / 2,
      };

      // Tráquea: 1/3 y 2/3 hacia abajo
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

      // ----- Punto paralelo a 148 y 377 alrededor de traquea1 -----
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

      // ----- Punto paralelo a 11 y 12 alrededor de baseCuello -----
      const offset_pose = Math.hypot(pose[12].x -pose[11].x, pose[12].y - pose[11].y);
      const dir_pose = {
        x: (pose[11].x - pose[12].x) / offset_pose,
        y: (pose[11].y - pose[12].y) / offset_pose,
      }

      const baseCuello_izq = {
        x: baseCuello.x - dir_pose.x * offset_pose /20,
        y: baseCuello.y - dir_pose.y * offset_pose /20,
        z: baseCuello.z,
      }

      const baseCuello_der = {
        x: baseCuello.x + dir_pose.x * offset_pose /20,
        y: baseCuello.y + dir_pose.y * offset_pose /20,
        z: baseCuello.z,
      }

      drawLandmarks(
        [traquea2, traquea1_izq, traquea1_der, baseCuello_izq, baseCuello_der],
        "blue"
      );
    }

    // --- FILTRAR LANDMARKS DE LA POSE ---
    const ignoredPosePoints = new Set([
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,     // cara
      15, 16, 17, 18, 19, 20, 21, 22        // muñecas, dedos
    ]);
    const cleanPose = poseLandmarks.map((p, i) =>
      (ignoredPosePoints.has(i) || i > 24) ? null : p
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
      const forehead = [
        face[67], face[109], face[10], face[338], face[297],
        face[299], face[9], face[69]
      ];
      const cejas = [
        // Ceja derecha
        face[46], face[53], face[52], face[65], face[55],
        face[70], face[63], face[105], face[66], face[107],
        // Ceja izquierda
        face[276], face[283], face[282], face[295], face[285],
        face[300], face[293], face[334], face[296], face[336]
      ];
      const ojos = [
        // Ojo derecho
        face[33], face[7], face[163], face[144], face[145],
        face[153], face[154], face[155], face[133], face[246],
        face[161], face[160], face[159], face[158], face[157], face[173],
        // Ojo izquierdo
        face[263], face[249], face[390], face[373], face[374],
        face[380], face[381], face[382], face[362], face[466],
        face[388], face[387], face[386], face[385], face[384], face[398]
      ];
      const iris = [
        ...face.slice(469, 472), // Iris derecho
        ...face.slice(474, 477)  // Iris izquierda
      ];
      const temples = [
        face[162], face[21], face[71], face[156], face[143], face[34],// Sien derecha
        face[389], face[251], face[301], face[383], face[372], face[264] // Sien izqierda
      ];
      const nariz = [
        face[6], face[197], face[195], face[5], face[4], face[2],//eje del tabique
        face[102], face[115], face[220], face[45], face[275],
        face[440], face[344], face[331],// eje transversal de la nariz
        face[98], face[97], face[326], face[327], face[129],
        face[142], face[126], face[217], face [114], face[188],
        face[122], face[351], face[412], face[343], face[437],
        face[355], face[371], face[358]
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
        face[306], face [408], face[304], face[303], face[302],
        face[11], face[72], face[73], face[74], face[184],
        face[76], face[77], face[90], face[180], face[85]
      ];
      const mejillas = [
        face[147], face[187], face[207], face[214], face[135],
        face[138], face[215], face[177], // Mejilla derecha
        face[376], face[411], face[427], face[434], face[364],
        face[367], face[435], face[401] //Meji;lla izquierda
      ];
      const menton = [
        face[32], face[194], face[83], face[18], face[313],
        face[418], face[262], face[369], face[377], face[152],
        face[148], face[140]
      ];

      drawLandmarks(forehead, "green");
      drawConnections(face, FOREHEAD_CONNECTIONS);
      drawLandmarks(cejas, "green");
      drawConnections(face, EYEBROWS_CONNECTIONS);
      drawLandmarks(ojos, "green");
      drawConnections(face, EYES_CONNECTIONS);
      drawLandmarks(temples, "green");
      drawConnections(face, TEMPLES_CONNECTIONS);
      drawLandmarks(nariz, "green");
      drawConnections(face, NOSE_CONNECTIONS);
      drawLandmarks(boca, "green");
      drawConnections(face, LIPS_CONNECTIONS);
      drawLandmarks(mejillas, "green");
      drawConnections(face, CHECKS_CONNECTIONS);
      drawLandmarks(menton, "green");
      drawConnections(face, CHIN_CONNECTIONS);
      drawLandmarks(iris, "green");
      drawConnections(face, IRIS_CONNECTIONS);
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