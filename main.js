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
    resultDiv.innerHTML = '<em>Loading...</em>';

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
        resultDiv.innerHTML = `
            <h3>Prediction Result</h3>
            <p><strong>Scleroderma Probability:</strong> ${data.scleroderma_probability}</p>
            <p><strong>Prediction:</strong> ${data.prediction}</p>
            <p><strong>Top 1 Recommended Test:</strong> ${data.top_1_recommended_test}</p>
            <p><strong>Top 3 Recommended Tests:</strong> ${data.top_3_recommended_tests.join(', ')}</p>
        `;
    } catch (error) {
        resultDiv.innerHTML = `<span style="color:red;">Error: ${error.message}</span>`;
    }
});
