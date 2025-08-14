export async function setupCamera(video) {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;
  await new Promise(resolve => video.onloadedmetadata = resolve);
  await video.play();
}

export async function setupVideoFromURL(video, url) {
  return new Promise(resolve => {
    video.src = url;
    video.onloadedmetadata = () => {
      video.play();
      resolve();
    };
  });
}