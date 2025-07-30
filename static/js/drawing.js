export function drawLandmarks(ctx, landmarks, color) {
  if (!landmarks) return;
  ctx.fillStyle = color;
  for (const landmark of landmarks) {
    if (!landmark) continue;
    ctx.beginPath();
    ctx.arc(landmark.x * ctx.canvas.width, landmark.y * ctx.canvas.height, 3, 0, 2 * Math.PI);
    ctx.fill();
  }
}

export function drawConnections(ctx, landmarks, connections, color = "white", lineWidth = 2) {
  if (!landmarks) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  for (const [startIdx, endIdx] of connections) {
    const start = landmarks[startIdx];
    const end = landmarks[endIdx];
    if (start && end) {
      ctx.beginPath();
      ctx.moveTo(start.x * ctx.canvas.width, start.y * ctx.canvas.height);
      ctx.lineTo(end.x * ctx.canvas.width, end.y * ctx.canvas.height);
      ctx.stroke();
    }
  }
}