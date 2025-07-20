const video = document.getElementById('video');
const predictionText = document.getElementById('prediction');

// Solicitar acceso a la cámara
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
    });

// Captura un frame y lo envía al backend
function captureAndSend() {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);

    const dataURL = canvas.toDataURL('image/jpeg');

    fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: dataURL })
    })
    .then(response => response.json())
    .then(data => {
        predictionText.textContent = data.prediction || data.error;
    })
    .catch(error => {
        predictionText.textContent = "Error al enviar imagen: " + error;
    });
}
