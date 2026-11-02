// Initialize dynamic gradient tracking for hero sliders
function setSliderGradient(slider) {
    const val = parseFloat(slider.value);
    const min = parseFloat(slider.min || 0);
    const max = parseFloat(slider.max || 100);
    const percentage = ((val - min) / (max - min)) * 100;
    slider.style.background = `linear-gradient(to right, var(--accent-teal) ${percentage}%, rgba(255,255,255,0.1) ${percentage}%)`;
}

document.querySelectorAll('.hero-slider').forEach(slider => {
    slider.addEventListener('input', (e) => {
        const id = e.target.id;
        const val = parseFloat(e.target.value);
        let displayId;
        if(id === 'dti') displayId = 'val-dti';
        else if (id === 'emi_to_income') displayId = 'val-emi';
        else if (id === 'credit_utilization') displayId = 'val-credit';
        else if (id === 'sim-slider') displayId = 'sim-val';

        const display = document.getElementById(displayId);
        if(display) {
            if(id === 'emi_to_income') {
                display.innerText = val.toFixed(2);
            } else {
                display.innerText = val + '%';
            }
        }
        setSliderGradient(e.target);
    });
    setSliderGradient(slider);
});

// ============================================
// API Configuration
// ============================================
const API_URL = 'https://01mayankk-credify-ai.hf.space';

// ============================================
// Form Submission Handler
// ============================================
let currentRiskProb = 0;

document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    const form = e.target;
    document.getElementById('results-dashboard').style.display = 'none';
    const loader = document.getElementById('loading-overlay');
    loader.style.display = 'block';
    loader.scrollIntoView({ behavior: 'smooth', block: 'center' });

    const btn = document.getElementById('analyze-btn');
    btn.disabled = true;

    const payload = {
        dti: parseFloat(document.getElementById('dti').value),
        credit_utilization: parseFloat(document.getElementById('credit_utilization').value),
        emi_to_income: parseFloat(document.getElementById('emi_to_income').value),
        loan_to_income: parseFloat(document.getElementById('loan_to_income').value),
        active_loan_count: parseInt(document.getElementById('active_loan_count').value),
        delinquency_count: parseInt(document.getElementById('delinquency_count').value),
    };

    try {
        const response = await fetch(API_URL + '/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`API Error ${response.status}: ${errorText}`);
        }
        const json = await response.json();
        
        setTimeout(() => {
            loader.style.display = 'none';
            renderDashboard(json.data);
            document.getElementById('results-dashboard').scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 1500); // Artificial delay for premium synth feel

    } catch (err) {
        console.error("Fetch error details:", err);
        alert('Error: ' + err.message + '\n\nCheck browser console for details.');
        loader.style.display = 'none';
        btn.disabled = false;
    }
});

function renderDashboard(data) {
    document.getElementById('results-dashboard').style.display = 'block';
    
    // Enable btn
    const btn = document.getElementById('analyze-btn');
    btn.disabled = false;

    // Report ID
    document.getElementById('random-id').innerText = Math.random().toString(36).substring(2, 9).toUpperCase();

    // 1. Ring Configuration & Counter
    currentRiskProb = data.probability_high_risk;
    const probRing = document.getElementById('prob-ring');
    const probText = document.getElementById('prob-percentage');
    const ringLabel = document.getElementById('ring-label');
    const badge = document.getElementById('badge-container');
    
    // Circumference mapping for r=54 -> 339.29
    const offset = 339.29 - (currentRiskProb * 339.29);
    
    let ringColor, badgeHTML, labelText;
    if (data.risk_band === 'High Risk') {
        ringColor = 'var(--risk-danger)';
        badgeHTML = `<i class="fa-solid fa-triangle-exclamation"></i> Critical Risk`;
        labelText = "Action Required";
        badge.style.color = 'var(--risk-danger)';
        badge.style.borderColor = 'var(--risk-danger)';
    } else if (data.risk_band.includes('Medium')) {
        ringColor = 'var(--risk-warning)';
        badgeHTML = `<i class="fa-solid fa-circle-exclamation"></i> Elevated Risk`;
        labelText = "Moderate Profile";
        badge.style.color = 'var(--risk-warning)';
        badge.style.borderColor = 'var(--risk-warning)';
    } else {
        ringColor = 'var(--risk-success)';
        badgeHTML = `<i class="fa-solid fa-shield-check"></i> Safe Profile`;
        labelText = "Excellent Standing";
        badge.style.color = 'var(--risk-success)';
        badge.style.borderColor = 'var(--risk-success)';
    }

    setTimeout(() => {
        probRing.style.strokeDashoffset = offset;
        probRing.style.stroke = ringColor;
        badge.innerHTML = badgeHTML;
        ringLabel.innerText = labelText;
        
        // Counter animation
        let start = 0;
        const target = Math.round(currentRiskProb * 100);
        const duration = 2000;
        const stepTime = Math.max(10, Math.floor(duration / (target || 1)));
        
        let timer = setInterval(() => {
            if(start >= target) {
                clearInterval(timer);
                probText.innerText = target + "%";
            } else {
                start++;
                probText.innerText = start + "%";
            }
        }, stepTime);
    }, 200);

    // 2. Pros / Cons Injection
    const pList = document.getElementById('pros-list');
    const cList = document.getElementById('cons-list');
    pList.innerHTML = ''; cList.innerHTML = '';

    if (data.insights && data.insights.pros) {
        data.insights.pros.forEach(item => { const li = document.createElement('li'); li.innerText = item; pList.appendChild(li); });
    }
    if (data.insights && data.insights.cons) {
        data.insights.cons.forEach(item => { const li = document.createElement('li'); li.innerText = item; cList.appendChild(li); });
    }

    // 3. SHAP Feature Drivers
    const driverList = document.getElementById('driver-list');
    driverList.innerHTML = '';
    
    data.key_reasons.forEach(reason => {
        const li = document.createElement('li');
        li.className = 'adv-driver';
        
        const arrow = reason.direction === 'up' ? 'fa-arrow-up' : 'fa-arrow-down';
        const colorC = reason.type === 'risk' ? 'risk-c' : 'safe-c';
        const bgC = reason.type === 'risk' ? 'bg-risk' : 'bg-safe';
        const sign = reason.type === 'risk' ? '+' : '-';
        
        li.innerHTML = `
            <div class="adv-label ${colorC}"><i class="fa-solid ${arrow}"></i> ${reason.feature}</div>
            <div class="adv-bar-wrap">
                <div class="adv-bar-fill ${bgC}" style="width: 0%"></div>
            </div>
            <div class="adv-impact ${colorC}">${sign}${reason.impact}%</div>
        `;
        driverList.appendChild(li);

        setTimeout(() => {
            const bar = li.querySelector('.adv-bar-fill');
            if(bar) bar.style.width = `${Math.min(reason.impact * 2.5, 100)}%`;
        }, 500);
    });

    // 4. Update the What-If simulation to start at current utilization
    const utilInput = document.getElementById('credit_utilization').value;
    const simSlider = document.getElementById('sim-slider');
    simSlider.value = utilInput;
    document.getElementById('sim-val').innerText = utilInput + '%';
    setSliderGradient(simSlider);
    updateSimulation(); // Set initial

    // 5. Chart Position Marker
    const marker = document.getElementById('chart-marker');
    // Map probability (0 - 1) to X coordinate (0 - 400), roughly
    const cx = Math.max(20, Math.min(380, currentRiskProb * 400 * 2)); // *2 because most are < 50%
    const cy = cx > 200 ? 50 : 80; // Mock path projection
    marker.style.transform = `translate(${cx}px, ${cy}px)`;
}

// Simulation interactivity
document.getElementById('sim-slider').addEventListener('input', updateSimulation);

function updateSimulation() {
    const origUtil = parseFloat(document.getElementById('credit_utilization').value || 30);
    const simUtil = parseFloat(document.getElementById('sim-slider').value);
    
    // Mock delta effect logic: +/- 0.4% risk for every 1% util change
    const deltaVis = (simUtil - origUtil) * 0.004;
    const simulatedProb = Math.max(0, Math.min(1, currentRiskProb + deltaVis));
    
    const displayScore = document.getElementById('sim-score-display');
    const displayAdvice = document.getElementById('sim-advice');
    
    const percentInt = Math.round(simulatedProb * 100);
    
    if (deltaVis > 0) {
        displayScore.innerHTML = `<span class="sim-symbol risk-c"><i class="fa-solid fa-arrow-trend-up"></i></span> <span class="risk-c">${percentInt}%</span>`;
        displayAdvice.innerText = "Increasing utilization raises your risk footprint.";
    } else if (deltaVis < 0) {
        displayScore.innerHTML = `<span class="sim-symbol safe-c"><i class="fa-solid fa-arrow-trend-down"></i></span> <span class="safe-c">${percentInt}%</span>`;
        displayAdvice.innerText = "Decreasing utilization secures a healthier profile.";
    } else {
        displayScore.innerHTML = `<span class="sim-symbol text-secondary">-</span> <span>${Math.round(currentRiskProb * 100)}%</span>`;
        displayAdvice.innerText = "No change from current baseline.";
    }
}
