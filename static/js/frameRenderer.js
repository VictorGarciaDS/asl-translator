// frameRenderer.js
import { drawLandmarks, drawConnections } from "./drawing.js";
import * as conns from "./connections.js";
import { enviarLandmarksAlServidor } from "./utils.js";

// 🖼️ Dibujar primero el video
export function drawVideoFrame(ctx, canvas, video) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
}

// 🖐️ Manos
export function drawHands(ctx, hands) {
  for (const hand of hands) {
    drawConnections(ctx, hand, conns.HAND_CONNECTIONS);
    drawLandmarks(ctx, hand, "red");
  }
}

// 🕺 Cuerpo
export function drawPose(ctx, cleanPose) {
  drawConnections(ctx, cleanPose, conns.POSE_CONNECTIONS);
  drawLandmarks(ctx, cleanPose, "blue");
}

// 🧠 Cuello
export function drawNeck(ctx, NECK_POINTS, allLandmarks) {
  if (NECK_POINTS.length > 0) {
    drawConnections(ctx, NECK_POINTS, [
      [0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7]
    ]);
    drawLandmarks(ctx, NECK_POINTS, "blue");
    allLandmarks.push({ tipo: "cuello", landmarks: NECK_POINTS });
  }
}

export function extractFacialRegions(face) {
  return {
    forehead : [
      face[67], face[109], face[10], face[338], face[297],
      face[299], face[9], face[69]
    ],
    ceja_izquierda : [
      face[276], face[283], face[282], face[295], face[285],
      face[300], face[293], face[334], face[296], face[336]
    ],
    ceja_derecha : [
      face[46], face[53], face[52], face[65], face[55],
      face[70], face[63], face[105], face[66], face[107]
    ],
    sien_izquierda : [
      face[389], face[251], face[301], face[383], face[372], face[264]
    ],
    sien_derecha : [
      face[162], face[21], face[71], face[156], face[143], face[34]
    ],
    ojo_izquierdo : [
      face[263], face[249], face[390], face[373], face[374],
      face[380], face[381], face[382], face[362], face[466],
      face[388], face[387], face[386], face[385], face[384],
      face[398]
    ],
    ojo_derecho : [
      face[33], face[7], face[163], face[144], face[145],
      face[153], face[154], face[155], face[133], face[246],
      face[161], face[160], face[159], face[158], face[157],
      face[173]
    ],
    iris_izquierdo : [
      ...face.slice(474, 477)
    ],
    iris_derecho : [
      ...face.slice(469, 472)
    ],
    nariz_izquierda : [
      face[6], face[197], face[195], face[5], face[4],//eje del tabique
      face[102], face[115], face[220], face[45],// eje transversal de la nariz
      face[122], face[188], face[114], face[217], face[126],
      face[142], face[129]
    ],
    nariz_derecha : [
      face[6], face[197], face[195], face[5], face[4],//eje del tabique
      face[275], face[440], face[344], face[331],// eje transversal de la nariz
      face[358], face[371], face[355], face[437], face[343],
      face[412], face[351]
    ],
    nariz_baja : [
      face[102], face[115], face[220], face[45],// eje transversal de la nariz
      face[275], face[440], face[344], face[331],// eje transversal de la nariz
      face[129], face[98], face[97], face[2], face[326], face[327], face[358]// base de la nariz
    ],
    mejilla_izquierda : [
      face[376], face[411], face[427], face[434], face[364],
      face[367], face[435], face[401]
    ],
    mejilla_derecha : [
      face[147], face[187], face[207], face[214], face[135],
      face[138], face[215], face[177]
    ],
    boca : [
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
    ],
    menton : [
      face[32], face[194], face[83], face[18], face[313],
      face[418], face[262], face[369], face[377], face[152],
      face[148], face[140]
    ]
  };
}

export function drawFacialRegions(ctx, face, regions) {
  const mapping = {
    forehead: conns.FOREHEAD_CONNECTIONS,
    ceja_izquierda: conns.LEFT_EYEBROW_CONNECTIONS,
    ceja_derecha: conns.RIGHT_EYEBROW_CONNECTIONS,
    sien_izquierda: conns.LEFT_TEMPLE_CONNECTIONS,
    sien_derecha: conns.RIGHT_TEMPLE_CONNECTIONS,
    ojo_izquierdo: conns.LEFT_EYE_CONNECTIONS,
    ojo_derecho: conns.RIGHT_EYE_CONNECTIONS,
    iris_izquierdo: conns.LEFT_IRIS_CONNECTIONS,
    iris_derecho: conns.RIGHT_IRIS_CONNECTIONS,
    nariz_izquierda: conns.LEFT_NOSE,
    nariz_derecha: conns.RIGHT_NOSE,
    nariz_baja: conns.LOW_NOSE,
    mejilla_izquierda: conns.LEFT_CHEEK_CONNECTIONS,
    mejilla_derecha: conns.RIGHT_CHEEK_CONNECTIONS,
    boca: conns.LIPS_CONNECTIONS,
    menton: conns.CHIN_CONNECTIONS
  };

  for (const [region, points] of Object.entries(regions)) {
    drawLandmarks(ctx, points, "green");
    drawConnections(ctx, face, mapping[region]);
  }
}

export function drawArmToPalmConnections(ctx, hands, poseLandmarks, canvas) {
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
}

export function renderFrame(ctx, canvas, video, data, allLandmarks, frameCounter, ignoredPosePoints, FRAMES_PARA_ENVIAR) {
  const { hands, faceResult, poseLandmarks, cleanPose, NECK_POINTS } = data;

  drawVideoFrame(ctx, canvas, video);
  drawHands(ctx, hands);
  drawPose(ctx, cleanPose);
  drawNeck(ctx, NECK_POINTS, allLandmarks);

  try {
    // --- LANDMARKS FACIALES (REGIONES) ---
    for (const face of faceResult.faceLandmarks || []) {
      const regions = extractFacialRegions(face);
      drawFacialRegions(ctx, face, regions);
    }
    drawArmToPalmConnections(ctx, hands, poseLandmarks, canvas);
  } catch (err) {
    console.error("Error en inferencia:", err.message);
  }

  console.log("Antes del if framecounter");
  if (frameCounter % FRAMES_PARA_ENVIAR === 0) {
    const faceLandmarks = faceResult.faceLandmarks?.[0] || null;

    console.log("Antes de cleanpose");
    const filteredPose = poseLandmarks
    .map((p, i) => ({ index: i, point: p }))
    .filter(({ index }) => !ignoredPosePoints.has(index) && index <= 24)
    .map(({ point }) => point);

    console.log("Antes de enviar");

    let regionsData = {};
    if (faceLandmarks) {
      regionsData = extractFacialRegions(faceLandmarks);
    }
    enviarLandmarksAlServidor({
      timestamp: Date.now(),
      forehead: regionsData.forehead,
      left_eyebrow: regionsData.ceja_izquierda,
      right_eyebrow: regionsData.ceja_derecha,
      sien_izquierda: regionsData.sien_izquierda,
      sien_derecha: regionsData.sien_derecha,
      ojo_izquierdo: regionsData.ojo_izquierdo,
      ojo_derecho: regionsData.ojo_derecho,
      iris_izquierdo: regionsData.iris_izquierdo,
      iris_derecho: regionsData.iris_derecho,
      nariz_izquierda: regionsData.nariz_izquierda,
      nariz_derecha: regionsData.nariz_derecha,
      nariz_baja: regionsData.nariz_baja,
      left_cheek: regionsData.mejilla_izquierda,
      right_cheek: regionsData.mejilla_derecha,
      boca: regionsData.boca,
      menton: regionsData.menton,
      neck: NECK_POINTS,
      pose: filteredPose,
      hands: hands,
    });
    console.log("Despues de enviar");
  }
}