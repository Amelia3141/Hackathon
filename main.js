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
        // SHAP textual explanation
        const topFeatures = data.shap_feature_names.map((f, i) => `${f} (${data.shap_feature_values[i].toFixed(3)})`).join(', ');
        resultDiv.innerHTML = `
            <h3>Prediction Result</h3>
            <p><strong>Scleroderma Probability:</strong> ${data.scleroderma_probability}</p>
            <p><strong>Prediction:</strong> ${data.prediction}</p>
            <p><strong>Top 1 Recommended Test:</strong> ${data.top_1_recommended_test}</p>
            <p><strong>Top 3 Recommended Tests:</strong> ${data.top_3_recommended_tests.join(', ')}</p>
            <h3>Model Explainability</h3>
            <p>This prediction was most influenced by: <strong>${topFeatures}</strong></p>
            <canvas id="shapChart" height="200"></canvas>
        `;
        // Draw SHAP bar chart
        if (window.shapChart) window.shapChart.destroy();
        const ctx = document.getElementById('shapChart').getContext('2d');
        window.shapChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data.shap_feature_names,
                datasets: [{
                    label: 'SHAP Value (Feature Importance)',
                    data: data.shap_feature_values,
                    backgroundColor: 'rgba(51, 102, 153, 0.7)'
                }]
            },
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
