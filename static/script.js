const videoElement = document.getElementById('video');
const canvasElement = document.getElementById('output');
const canvasCtx = canvasElement.getContext('2d');

// Configuración MediaPipe Hands
const hands = new Hands({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
});
hands.setOptions({
  maxNumHands: 2,
  modelComplexity: 1,
  minDetectionConfidence: 0.7,
  minTrackingConfidence: 0.5
});

// Configuración Pose
const pose = new Pose({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`
});
pose.setOptions({
  modelComplexity: 1,
  smoothLandmarks: true,
  enableSegmentation: false,
  minDetectionConfidence: 0.7,
  minTrackingConfidence: 0.5
});

// Configuración FaceMesh
const faceMesh = new FaceMesh({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
});
faceMesh.setOptions({
  maxNumFaces: 1,
  refineLandmarks: true,
  minDetectionConfidence: 0.7,
  minTrackingConfidence: 0.5
});

// Dibujar resultados
function onResults(results) {
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

  if (results.poseLandmarks) {
    drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS, {color: '#00FF00', lineWidth: 4});
    drawLandmarks(canvasCtx, results.poseLandmarks, {color: '#FF0000', lineWidth: 2});
  }

  if (results.faceMeshLandmarks) {
    drawConnectors(canvasCtx, results.faceMeshLandmarks, FACEMESH_TESSELATION, {color: '#C0C0C070', lineWidth: 1});
  }

  if (results.multiHandLandmarks) {
    for (const landmarks of results.multiHandLandmarks) {
      drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {color: '#FFFFFF', lineWidth: 5});
      drawLandmarks(canvasCtx, landmarks, {color: '#FF0000', lineWidth: 2});
    }
  }
  canvasCtx.restore();

  sendLandmarksToBackend(results);
}

function sendLandmarksToBackend(results) {
  const data = {
    pose: results.poseLandmarks || [],
    face: results.faceMeshLandmarks || [],
    hands: results.multiHandLandmarks || []
  };

  fetch('/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(data)
  })
    .then(response => response.json())
    .then(result => console.log('Respuesta del backend:', result))
    .catch(error => console.error('Error al enviar al backend:', error));
}

// Cámara compartida
const camera = new Camera(videoElement, {
  onFrame: async () => {
    await hands.send({image: videoElement});
    await pose.send({image: videoElement});
    await faceMesh.send({image: videoElement});
  },
  width: 640,
  height: 480
});
camera.start();

hands.onResults(onResults);
pose.onResults(onResults);
faceMesh.onResults(onResults);