document.getElementById('uploadForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const fileInput = document.getElementById('imageInput');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('please select an image');
        return;
    }
    
    const formData = new FormData();
    formData.append('image', file);
    
    document.getElementById('loading').classList.remove('hidden');
    document.getElementById('results').classList.add('hidden');
    document.getElementById('submitBtn').disabled = true;
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            document.getElementById('resultImage').src = data.image;
            
            let html = '';
            if (data.detections.length > 0) {
                data.detections.forEach(function(det) {
                    html += '<div class="detection-item">';
                    html += '<div class="detection-header">' + det.class_name + '</div>';
                    html += '<div class="confidence">Confidence: ' + (det.confidence * 100).toFixed(1) + '%</div>';
                    html += '<div class="bin-info">' + det.bin + '</div>';
                    html += '</div>';
                });
            } else {
                html = '<p>No waste detected</p>';
            }
            
            document.getElementById('detectionList').innerHTML = html;
            document.getElementById('results').classList.remove('hidden');
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        alert('Error: ' + error);
    } finally {
        document.getElementById('loading').classList.add('hidden');
        document.getElementById('submitBtn').disabled = false;
    }
});



