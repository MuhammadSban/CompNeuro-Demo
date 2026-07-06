// ═══════════════════════════════════════════════════════════════════════════════
// COMPONENT ANALYSIS — parameter submission + Three.js plot
// ═══════════════════════════════════════════════════════════════════════════════
// Reads the four dropdowns, POSTs them to the FastAPI backend, and renders the
// returned point cloud in #caPlot on a grid baseline. Camera behaviour follows
// the Dimensions dropdown: 2D snaps to a locked front-on view, 3D is free orbit.
// Uses the same three@0.129.0 CDN build as 3D.js.

import * as THREE from "https://cdn.skypack.dev/three@0.129.0/build/three.module.js";
import { OrbitControls } from "https://cdn.skypack.dev/three@0.129.0/examples/jsm/controls/OrbitControls.js";

const CA_ENDPOINT = 'http://localhost:8000/analysis';   // SERVER URL GOES HERE

const GRID_SIZE = 10;   // world units across (points span roughly ±3)
const GRID_DIVS = 20;

// ═══════════════════════════════════════════════════════════════════════════════
// FASTAPI — PARAMETER SUBMISSION
// ═══════════════════════════════════════════════════════════════════════════════

// Reads the four Component Analysis dropdowns into an analysisRequest payload.
function getComponentParams() {
  const val = id => document.getElementById(id)?.value;

  return {
    type:      val('caMethod'),                     // 'pca' | 'tsne'
    dimension: parseInt(val('caDimensions'), 10),   // 2 | 3
    layer:     val('caLayer'),                       // 'input' | 'encoder1..3'
    label:     val('caLabels'),                      // 'none' | 'time' | 'phoneme' | 'word'
  };
}

// POSTs the current dropdown selection, updates the view to match the selected
// dimension, and plots the returned points. Fails gracefully (grid still shows)
// when the endpoint is offline.
async function submitComponentParams() {
  const params = getComponentParams();

  // Update grid + camera up front so switching Dimensions is reflected even
  // if the server is unreachable.
  applyViewMode(params.dimension);

  if (!CA_ENDPOINT) {
    console.warn('[ComponentAnalysis] CA_ENDPOINT is not set — cannot fetch results.');
    return null;
  }

  console.log('[ComponentAnalysis] Submitting:', params);

  try {
    const res = await fetch(CA_ENDPOINT, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(params),
    });
    if (!res.ok) {
      // Surface the backend's error message (FastAPI puts it in `detail`).
      let detail = res.statusText;
      try { detail = (await res.json()).detail ?? detail; } catch {}
      throw new Error(`HTTP ${res.status}: ${detail}`);
    }
    const data = await res.json();
    console.log('[ComponentAnalysis] Response:', data);
    drawPoints(data.primaryChannels ?? []);   // labels ignored for now
    return data;
  } catch (err) {
    console.error('[ComponentAnalysis] Submit failed:', err);
    return null;
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// THREE.JS PLOT
// ═══════════════════════════════════════════════════════════════════════════════

let scene, camera, renderer, controls, grid, pointCloud, currentDims = null;

function initPlot() {
  const container = document.getElementById('caPlot');
  if (!container) return;

  // Grid baseline replaces the placeholder text.
  container.querySelector('.caPlotPlaceholder')?.style.setProperty('display', 'none');

  scene = new THREE.Scene();

  const w = container.clientWidth  || 1;
  const h = container.clientHeight || 1;

  camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 1000);

  renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
  renderer.setSize(w, h);
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.domElement.style.position = 'absolute';
  renderer.domElement.style.inset = '0';
  container.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;

  applyViewMode(2);             // draws the grid + sets the starting front-on camera

  // Keep renderer/camera in sync with the container (it starts at 0×0 while the
  // tab is hidden, then the observer corrects the size once it becomes visible).
  const ro = new ResizeObserver(entries => {
    const { width, height } = entries[0].contentRect;
    if (width > 0 && height > 0) {
      renderer.setSize(width, height);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    }
  });
  ro.observe(container);

  animate();
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

// Snaps the view for the given dimension count. Both modes share the same
// front-on start (grid stands up in the XY plane, camera dead-on); 3D simply
// unlocks orbit so you can rotate around from there, 2D stays locked. Only
// re-snaps when the dimension actually changes, so rotating in 3D isn't reset
// by other dropdown changes.
function applyViewMode(dims) {
  if (!scene || dims === currentDims) return;
  currentDims = dims;

  if (grid) {
    scene.remove(grid);
    grid.geometry.dispose();
    grid.material.dispose();
  }
  grid = new THREE.GridHelper(GRID_SIZE, GRID_DIVS, 0x8a8a8a, 0x3a3a3a);
  grid.rotation.x = Math.PI / 2;                 // XZ floor → XY wall, facing camera
  scene.add(grid);

  // Shared snapped front-on camera; the only difference is orbit lock.
  controls.target.set(0, 0, 0);
  camera.position.set(0, 0, 14);
  camera.up.set(0, 1, 0);
  controls.enableRotate = (dims === 3);
  controls.update();
}

// Replaces the plotted point cloud. Each point is [x, y] or [x, y, z]; a missing
// z is treated as 0 so 2D embeddings sit flat on the front-on grid.
function drawPoints(points) {
  if (!scene) return;

  if (pointCloud) {
    scene.remove(pointCloud);
    pointCloud.geometry.dispose();
    pointCloud.material.dispose();
    pointCloud = null;
  }
  if (!points.length) return;

  const positions = new Float32Array(points.length * 3);
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    positions[i * 3]     = p[0];
    positions[i * 3 + 1] = p[1];
    positions[i * 3 + 2] = p.length > 2 ? p[2] : 0;
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const mat = new THREE.PointsMaterial({ color: 0xffc62a, size: 0.12 });
  pointCloud = new THREE.Points(geom, mat);
  scene.add(pointCloud);
}

// ═══════════════════════════════════════════════════════════════════════════════
// WIRING
// ═══════════════════════════════════════════════════════════════════════════════

initPlot();

// Re-submit (and re-render) whenever any dropdown changes.
['caMethod', 'caDimensions', 'caLayer', 'caLabels'].forEach(id => {
  document.getElementById(id)?.addEventListener('change', submitComponentParams);
});

// Best-effort initial load so the plot shows data immediately when the backend
// is up; the grid baseline renders regardless.
submitComponentParams();
