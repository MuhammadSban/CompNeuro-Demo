const API_URL = "http://127.0.0.1:8000/build";

const trainBtn = document.getElementById("train-btn");
const statusEl = document.getElementById("status");
const canvas = document.getElementById("boundary");
const ctx = canvas.getContext("2d");

trainBtn.addEventListener("click", async () => {
    // Parse the hidden layers input "4, 4, 2" → [4, 4, 2]
    const neuronsText = document.getElementById("neurons").value;
    const neurons = neuronsText.split(",").map(s => parseInt(s.trim()));

    const requestBody = {
        neurons: neurons,
        lr: parseFloat(document.getElementById("lr").value),
        act: document.getElementById("act").value,
        loss: "MSE",
        opt: "SGD"
    };

    trainBtn.disabled = true;
    statusEl.textContent = "Training... (this may take a few seconds)";

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(requestBody)
        });

        if (!response.ok) {
            throw new Error(`Server returned ${response.status}`);
        }

        const data = await response.json();
        drawBoundary(data);
        statusEl.textContent = "Done.";
    } catch (err) {
        statusEl.textContent = `Error: ${err.message}`;
    } finally {
        trainBtn.disabled = false;
    }
});

function drawBoundary(data) {
    const { prediction, X, y, grid_range } = data;
    const [xmin, xmax, ymin, ymax] = grid_range;

    const W = canvas.width;
    const H = canvas.height;
    const gridSize = prediction.length;       // 100
    const cellW = W / gridSize;
    const cellH = H / gridSize;

    // Paint the heatmap
    for (let i = 0; i < gridSize; i++) {
        for (let j = 0; j < gridSize; j++) {
            // Note: prediction[i][j] where i is the row (y), j is the column (x).
            // Canvas y grows downward, so we flip to keep math-style orientation.
            const value = prediction[gridSize - 1 - i][j];
            ctx.fillStyle = valueToColor(value);
            ctx.fillRect(j * cellW, i * cellH, cellW + 1, cellH + 1);
        }
    }

    // Overlay training points
    for (let k = 0; k < X.length; k++) {
        const [px, py] = X[k];
        const label = y[k];

        // Map data coordinates to canvas pixel coordinates
        const cx = ((px - xmin) / (xmax - xmin)) * W;
        const cy = H - ((py - ymin) / (ymax - ymin)) * H;

        ctx.beginPath();
        ctx.arc(cx, cy, 4, 0, 2 * Math.PI);
        ctx.fillStyle = label > 0 ? "#0877bd" : "#f59322";
        ctx.fill();
        ctx.strokeStyle = "black";
        ctx.lineWidth = 1;
        ctx.stroke();
    }
}

// Map a model output in [-1, 1] to an orange→white→blue color
function valueToColor(v) {
    const t = Math.max(-1, Math.min(1, v));   // clamp
    if (t >= 0) {
        // 0 (white) → 1 (blue)
        const r = Math.round(255 + (8 - 255) * t);
        const g = Math.round(255 + (119 - 255) * t);
        const b = Math.round(255 + (189 - 255) * t);
        return `rgb(${r}, ${g}, ${b})`;
    } else {
        // 0 (white) → -1 (orange)
        const r = Math.round(255 + (245 - 255) * -t);
        const g = Math.round(255 + (147 - 255) * -t);
        const b = Math.round(255 + (34 - 255) * -t);
        return `rgb(${r}, ${g}, ${b})`;
    }
}