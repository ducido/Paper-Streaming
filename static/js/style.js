document.getElementById('rotate-button').addEventListener('click', rotate);
let currentRotation = 0;
let stream;
let mediaRecorder;
const recordedChunks = [];
const recordButton = document.getElementById('record');


recordButton.addEventListener('click', () => {
   if (recordButton.textContent === 'Record video') {
       startRecording();
   } else {
       stopRecording();
   }
});
navigator.mediaDevices.getUserMedia({ video: true })
   .then(function (userStream) {
       stream = userStream;
       mediaRecorder = new MediaRecorder(stream, { mimeType: 'video/webm' });


       mediaRecorder.ondataavailable = (event) => {
           if (event.data.size > 0) {
               recordedChunks.push(event.data);
           }
       };


       mediaRecorder.onstop = () => {
           const blob = new Blob(recordedChunks, { type: 'video/webm' });
           const url = URL.createObjectURL(blob);
           const a = document.createElement('a');
           document.body.appendChild(a);
           a.style.display = 'none';
           a.href = url;
           a.download = 'recorded-video.webm';
           a.click();
           window.URL.revokeObjectURL(url);
       };
   })
   .catch(function (error) {
       console.error('Error accessing the camera:', error);
   });
function startRecording() {
   recordedChunks.length = 0; // Clear any previous data
   mediaRecorder.start();
   recordButton.textContent = 'Stop recording';
}


function stopRecording() {
   mediaRecorder.stop();
   recordButton.textContent = 'Record video';
}




function rotate() {
   currentRotation += 90;
   video.style.transform = `rotate(${currentRotation}deg)`;
   processed.style.transform = `rotate(${currentRotation}deg)`;
}
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const context = canvas.getContext('2d');
const processed = document.getElementById('processed');
const socket = io.connect('http://' + location.hostname + ':' + location.port);


       navigator.mediaDevices.getUserMedia({ video: true })
           .then(stream => {
               video.srcObject = stream;
               setInterval(captureAndSendFrame, 100);  // Adjust interval as needed
           });
      


       function captureAndSendFrame() {


           context.drawImage(video, 0, 0, canvas.width, canvas.height);
           const frame = canvas.toDataURL('image/jpeg');
           socket.emit('video_frame', frame);


   // Reset the video element's transformation after drawing
       ;


      
       }


       socket.on('processed_frame', data => {
          
           processed.src = 'data:image/jpeg;base64,' + data;
       });
