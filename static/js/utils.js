export function resizeCanvas(canvas) {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
}

export function enviarLandmarksAlServidor(data) {
  fetch("/api/landmarks", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  }).catch(err => console.error("❌ Error al enviar landmarks:", err));
}