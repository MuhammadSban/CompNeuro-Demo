/**
 * DEMO: Vertex Coloring Heatmap
 *
 * Strategy:
 *   1. Load brain GLTF and electrodes (same files as the main project).
 *   2. Simulate electrode "neural activity" values (sine wave pattern).
 *   3. For every mesh vertex in the brain, compute an Inverse-Distance-Weighted
 *      (IDW) blend of the electrode values → a scalar in [0, 1].
 *   4. Map that scalar through the viridis colormap → RGB.
 *   5. Store the result in geometry.attributes.color and enable vertexColors on
 *      the material. Three.js then interpolates across triangle faces for free.
 *
 * Pros:  Very simple. Fast after the one-time CPU pass. No shader authoring.
 * Cons:  Resolution capped by mesh vertex density. Coarse meshes look blocky.
 *        Recomputing colors on every data update requires a CPU loop + GPU upload.
 *
 * INTEGRATION NOTE for the main demoUI:
 *   Replace the `loader.load(...)` block in 3D.js with the one below, and call
 *   `applyVertexHeatmap(object, electrodeWorldPositions, electrodeValues)` once
 *   the electrode fetch resolves. To animate, call it each frame with new values
 *   and set `geometry.attributes.color.needsUpdate = true`.
 *   The `k` parameter controls how many nearest neighbours are blended (default 8).
 *   Larger K → smoother gradients; smaller K → sharper locality.
 */

import * as THREE from "https://cdn.skypack.dev/three@0.129.0/build/three.module.js";
import { OrbitControls } from "https://cdn.skypack.dev/three@0.129.0/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader }    from "https://cdn.skypack.dev/three@0.129.0/examples/jsm/loaders/GLTFLoader.js";

// ─── Scene setup (mirrors original 3D.js) ───────────────────────────────────
const scene    = new THREE.Scene();
const camera   = new THREE.PerspectiveCamera(1, innerWidth / innerHeight, 0.1, 2000);
camera.position.z = 50;

const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
renderer.setSize(innerWidth, innerHeight);
document.getElementById("container3D").appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);

scene.add(new THREE.DirectionalLight(0xffffff, 1).translateX(500).translateY(500).translateZ(500));
scene.add(new THREE.AmbientLight(0x333333, 5));

// ─── Viridis colormap ────────────────────────────────────────────────────────
// 16-stop approximation of matplotlib's viridis.
const VIRIDIS = [
  [0.267, 0.005, 0.329], [0.278, 0.051, 0.375], [0.282, 0.100, 0.422],
  [0.278, 0.150, 0.460], [0.265, 0.200, 0.492], [0.246, 0.249, 0.517],
  [0.222, 0.296, 0.536], [0.196, 0.343, 0.551], [0.170, 0.391, 0.560],
  [0.143, 0.440, 0.563], [0.119, 0.491, 0.559], [0.105, 0.542, 0.542],
  [0.135, 0.591, 0.509], [0.230, 0.636, 0.458], [0.416, 0.680, 0.388],
  [0.993, 0.906, 0.144],
];

function viridis(t) {
  t = Math.max(0, Math.min(1, t));
  const n  = VIRIDIS.length - 1;
  const i  = Math.min(Math.floor(t * n), n - 1);
  const f  = t * n - i;
  const a  = VIRIDIS[i], b = VIRIDIS[i + 1];
  return [a[0] + (b[0]-a[0])*f, a[1] + (b[1]-a[1])*f, a[2] + (b[2]-a[2])*f];
}

// ─── IDW vertex coloring ─────────────────────────────────────────────────────
/**
 * @param {THREE.Object3D}  brainObject   - loaded GLTF scene root
 * @param {THREE.Vector3[]} elecPositions - electrode positions in WORLD space
 * @param {number[]}        elecValues    - one scalar [0..1] per electrode
 * @param {number}          k             - number of nearest neighbours to blend (default 8)
 * @param {number}          power         - IDW exponent (2 = standard)
 */
function applyVertexHeatmap(brainObject, elecPositions, elecValues, k = 8, power = 2) {
  const vWorld = new THREE.Vector3();

  brainObject.traverse(child => {
    if (!child.isMesh) return;

    const geo = child.geometry;
    if (!geo.attributes.position) return;

    // Clone material so we don't mutate shared instances
    child.material = child.material.clone();
    child.material.vertexColors = true;

    child.updateWorldMatrix(true, false);
    const worldMatrix = child.matrixWorld;

    const posAttr  = geo.attributes.position;
    const count    = posAttr.count;
    const colorBuf = new Float32Array(count * 3);

    // Pre-allocate K-slot arrays once per mesh (reused across vertices)
    const kDists = new Float32Array(k).fill(Infinity);
    const kVals  = new Float32Array(k);

    for (let i = 0; i < count; i++) {
      vWorld.fromBufferAttribute(posAttr, i).applyMatrix4(worldMatrix);

      // ── Pass 1: gather K nearest electrodes ─────────────────────────────
      kDists.fill(Infinity);
      kVals.fill(0);

      for (let j = 0; j < elecPositions.length; j++) {
        const dist = vWorld.distanceTo(elecPositions[j]);

        // Find the current worst (farthest) slot
        let worstSlot = 0, worstDist = kDists[0];
        for (let s = 1; s < k; s++) {
          if (kDists[s] > worstDist) { worstDist = kDists[s]; worstSlot = s; }
        }

        // Evict the worst slot if this electrode is closer
        if (dist < worstDist) {
          kDists[worstSlot] = dist;
          kVals[worstSlot]  = elecValues[j];
        }
      }

      // ── Pass 2: IDW over the K nearest only ─────────────────────────────
      let weightSum = 0, valueSum = 0;

      for (let s = 0; s < k; s++) {
        const dist = kDists[s];
        if (!isFinite(dist)) continue;            // unfilled slot

        if (dist < 1e-6) {                        // vertex coincides with electrode
          weightSum = 1; valueSum = kVals[s]; break;
        }

        const w    = 1 / Math.pow(dist, power);
        weightSum += w;
        valueSum  += w * kVals[s];
      }

      const t         = weightSum > 0 ? valueSum / weightSum : 0;
      const [r, g, b] = viridis(t);
      colorBuf[i*3]   = r;
      colorBuf[i*3+1] = g;
      colorBuf[i*3+2] = b;
    }

    geo.setAttribute('color', new THREE.BufferAttribute(colorBuf, 3));
  });
}

// ─── Load data ───────────────────────────────────────────────────────────────
const SCALE  = 0.006;               // same as original 3D.js
const OFFSET = new THREE.Vector3(-0.01, 0.4, -0.15);

let brainLoaded    = false;
let electrodesLoaded = false;
let brainObject, elecWorldPositions, elecValues;

function tryApply() {
  if (!brainLoaded || !electrodesLoaded) return;
  applyVertexHeatmap(brainObject, elecWorldPositions, elecValues);
  showLegend();
}

// Brain
const loader = new GLTFLoader();
loader.load(
  '../../human_brain/scene.gltf',
  gltf => {
    brainObject = gltf.scene;
    scene.add(brainObject);

    const box    = new THREE.Box3().setFromObject(brainObject);
    const center = box.getCenter(new THREE.Vector3());
    controls.target.copy(center);
    camera.position.set(center.x, center.y, center.z + 100);
    controls.update();

    brainLoaded = true;
    tryApply();
  },
  xhr => document.getElementById('status').textContent =
         `Loading brain… ${Math.round(100 * xhr.loaded / xhr.total)}%`,
  err => console.error('Brain load error', err)
);

// Electrodes + simulated values
fetch('../../elecs.json')
  .then(r => r.json())
  .then(coords => {
    elecWorldPositions = coords.map(([x, y, z]) =>
      new THREE.Vector3(
        OFFSET.x + x * SCALE,
        OFFSET.y + y * SCALE,
        OFFSET.z + z * SCALE
      )
    );

    // Simulated activity: sine wave over electrode index, in [0, 1].
    // Replace with real data in production (e.g. decoded from a WebSocket).
    elecValues = elecWorldPositions.map((_, i) =>
      0.5 + 0.5 * Math.sin(i * 0.18)
    );

    // Show electrode spheres (coloured by their own value)
    const group = new THREE.Group();
    elecWorldPositions.forEach((pos, i) => {
      const [r, g, b] = viridis(elecValues[i]);
      const mesh = new THREE.Mesh(
        new THREE.SphereGeometry(0.008, 8, 8),
        new THREE.MeshBasicMaterial({ color: new THREE.Color(r, g, b) })
      );
      mesh.position.copy(pos);
      group.add(mesh);
    });
    scene.add(group);

    electrodesLoaded = true;
    tryApply();
    document.getElementById('status').textContent = '';
  })
  .catch(err => console.error('Electrode load failed', err));

// ─── Legend ──────────────────────────────────────────────────────────────────
function showLegend() {
  const canvas = document.getElementById('legend');
  const ctx    = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  for (let i = 0; i < W; i++) {
    const [r, g, b] = viridis(i / W);
    ctx.fillStyle = `rgb(${Math.round(r*255)},${Math.round(g*255)},${Math.round(b*255)})`;
    ctx.fillRect(i, 0, 1, H);
  }
}

// ─── Animation loop ──────────────────────────────────────────────────────────
function animate() {
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
}
animate();

window.addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});
