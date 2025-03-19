document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const progressArea = document.getElementById('progressArea');
    const resultArea = document.getElementById('resultArea');
    const progressBar = document.querySelector('.progress-bar');
    const statusText = document.getElementById('statusText');
    const scaleRange = document.getElementById('scaleRange');
    const scaleValue = document.getElementById('scaleValue');
    const interpolationRange = document.getElementById('interpolationRange');
    const interpolationValue = document.getElementById('interpolationValue');
    const downloadBtn = document.getElementById('downloadBtn');
    const originalImage = document.getElementById('originalImage');
    const originalVideo = document.getElementById('originalVideo');
    const upscaledImage = document.getElementById('upscaledImage');
    const upscaledVideo = document.getElementById('upscaledVideo');
    const originalContainer = document.getElementById('originalContainer');
    const upscaledContainer = document.getElementById('upscaledContainer');

    let progressInterval = null;
    let currentFile = null;

    // Update scale value display
    scaleRange.addEventListener('input', function() {
        scaleValue.textContent = this.value;
    });

    // Update interpolation value display
    interpolationRange.addEventListener('input', function() {
        interpolationValue.textContent = this.value;
    });

    // Handle file upload
    uploadForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const fileInput = document.getElementById('imageFile');
        const file = fileInput.files[0];
        if (!file) {
            alert('Please select a file');
            return;
        }

        // Show progress area
        progressArea.classList.remove('d-none');
        resultArea.classList.add('d-none');
        progressBar.style.width = '0%';
        statusText.textContent = 'Processing file...';

        // Create form data
        const formData = new FormData();
        formData.append('file', file);
        formData.append('scale', scaleRange.value / 100);
        formData.append('interpolation', interpolationRange.value / 100);
        formData.append('model', document.getElementById('aiModel').value);

        try {
            // Upload and process file
            const response = await fetch('/upscale', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }

            if (data.type === 'video') {
                // Start progress polling for video
                startProgressPolling();
            } else {
                // Show image result
                showResult(data.result_url, file);
            }
        } catch (error) {
            statusText.textContent = `Error: ${error.message}`;
            progressBar.style.width = '0%';
            progressBar.classList.remove('progress-bar-animated');
        }
    });

    // Start progress polling for video processing
    function startProgressPolling() {
        progressInterval = setInterval(async () => {
            try {
                const response = await fetch('/progress');
                const data = await response.json();
                
                if (data.progress !== null) {
                    progressBar.style.width = `${data.progress}%`;
                    statusText.textContent = `Processing: ${Math.round(data.progress)}%`;
                }
            } catch (error) {
                console.error('Error polling progress:', error);
            }
        }, 1000);
    }

    // Show result after processing
    function showResult(resultUrl, originalFile) {
        // Clear progress
        if (progressInterval) {
            clearInterval(progressInterval);
        }
        progressArea.classList.add('d-none');
        resultArea.classList.remove('d-none');

        // Show original file
        if (originalFile.type.startsWith('video/')) {
            originalImage.classList.add('d-none');
            originalVideo.classList.remove('d-none');
            originalVideo.src = URL.createObjectURL(originalFile);
        } else {
            originalImage.classList.remove('d-none');
            originalVideo.classList.add('d-none');
            originalImage.src = URL.createObjectURL(originalFile);
        }

        // Show upscaled file
        if (resultUrl.endsWith('.mp4')) {
            upscaledImage.classList.add('d-none');
            upscaledVideo.classList.remove('d-none');
            upscaledVideo.src = resultUrl;
        } else {
            upscaledImage.classList.remove('d-none');
            upscaledVideo.classList.add('d-none');
            upscaledImage.src = resultUrl;
        }

        // Setup download button
        currentFile = resultUrl;
        downloadBtn.onclick = () => {
            const link = document.createElement('a');
            link.href = resultUrl;
            link.download = `upscaled_${originalFile.name}`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        };
    }
});