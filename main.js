document.getElementById('predict-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    const symptoms = document.getElementById('symptoms').value;
    const featureCheckboxes = document.querySelectorAll('input[type="checkbox"][name="features"]');
    const features = {};
    featureCheckboxes.forEach(cb => {
        features[cb.value] = cb.checked ? 1 : null;
    });
    // Get numeric lab values
    [
        'BNP (pg/ml)',
        'NTproBNP (pg/ml)',
        'Body weight (kg)',
        'DAS 28 (ESR, calculated)',
        'DAS 28 (CRP, calculated)',
        'Forced Vital Capacity (FVC - % predicted)',
        'DLCO/SB (% predicted)'
    ].forEach(field => {
        const val = document.querySelector(`input[name="${field}"]`).value;
        features[field] = val ? parseFloat(val) : features[field] || null;
    });

    const payload = {
        text: symptoms,
        features: features
    };

    const resultDiv = document.getElementById('result');
    const spinner = document.getElementById('loading-spinner');
    const submitBtn = document.getElementById('submit-btn');
    resultDiv.innerHTML = '';
    spinner.style.display = 'block';
    submitBtn.disabled = true;

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });
        if (!response.ok) throw new Error('Server error: ' + response.status);
        const data = await response.json();
        
        
        // Badge for prediction
        let badgeClass = '';
        let badgeText = '';
        if (data.prediction.toLowerCase().includes('likely')) {
            badgeClass = data.prediction.toLowerCase().includes('un') ? 'prediction-badge prediction-unlikely' : 'prediction-badge prediction-likely';
            badgeText = data.prediction;
        }
        resultDiv.innerHTML = `
            <h3>Prediction Result</h3>
            <div style="margin-bottom:0.6em;">
                ${badgeText ? `<span class="${badgeClass}" role="status">${badgeText}</span>` : ''}
            </div>
            <p><strong>Scleroderma Probability:</strong> ${(data.scleroderma_probability*100).toFixed(1)}%</p>
            <p><strong>Top 1 Recommended Test:</strong> ${data.top_1_recommended_test}</p>
            <p><strong>Top 3 Recommended Tests:</strong> ${data.top_3_recommended_tests.join(', ')}</p>
            <h3>Recommended Antibody Tests</h3>
            <ul style="margin-top:0;">${(data.recommended_antibody_tests || []).map(t => {
                if (Array.isArray(t)) {
                    // [antibody, association]
                    return `<li><b>${t[0]}</b><br><span style='font-size:0.97em;color:#444;'>${t[1]}</span></li>`;
                } else {
                    return `<li>${t}</li>`;
                }
            }).join('')}</ul>

        `;
        
        
        
        

            options: {
                indexAxis: 'y',
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: { title: { display: true, text: 'Importance' } },
                    y: { title: { display: true, text: 'Feature' } }
                }
            }
        });
    } catch (error) {
        resultDiv.innerHTML = `<span style="color:red;">Error: ${error.message}</span>`;
    }
});
