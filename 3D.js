//Import Three.js
import * as THREE from "https://cdn.skypack.dev/three@0.129.0/build/three.module.js";

//Camera Orbit Control
import { OrbitControls } from "https://cdn.skypack.dev/three@0.129.0/examples/jsm/controls/OrbitControls.js";

//GLTF Loader
import { GLTFLoader } from "https://cdn.skypack.dev/three@0.129.0/examples/jsm/loaders/GLTFLoader.js";

const scene = new THREE.Scene();
const cameraRot = new THREE.PerspectiveCamera(1, window.innerWidth / window.innerHeight, 0.1, 2000);
let controls;
const loader = new GLTFLoader();

const DIST_MAX = 0.2;

// ── Weight smoothing ────────────────────────────────────────────────────────
// Each surface point blends the K nearest electrodes' weights instead of just
// the single nearest one. The nearest electrode contributes NEAREST_WEIGHT of
// the blend; the remaining (1 - NEAREST_WEIGHT) is split among the surrounding
// (K-1) electrodes by inverse distance. NEAREST_WEIGHT is adjustable live via
// setNearestWeight(v) (see bottom of file); K requires a rebuild (re-click Run).
const K_NEAREST     = 4;     // how many nearest electrodes factor into each point
const NEAREST_WEIGHT = 0.5;  // share given to the single closest electrode (0..1)
const SALIENCY_GAIN  = 2.0;  // contrast multiplier applied after [0,1] normalization

// ── Saliency methods ────────────────────────────────────────────────────────
// Different ways of scoring per-bipolar-electrode importance. Toggle between
// them (see the dropdown in demoUI.html) to compare which best highlights the
// relevant cortex. Each `parse` returns a flat array of one weight per bipolar
// electrode (429). `agent5_sage_regional_mse` is intentionally omitted: it gives
// 9 per-region values and there's no per-electrode region mapping in the repo to
// expand it to the 429 electrodes — add it here once that mapping exists.
const SALIENCY_METHODS = [
  { id: 'permutation', label: 'SAGE · Permutation', file: 'saliencies/agent1_sage_permutation_saliencies.json', parse: d => d },
  { id: 'kernel',      label: 'SAGE · Kernel',      file: 'saliencies/agent2_sage_kernel_saliencies.json',      parse: d => d },
  { id: 'gradient',    label: 'Gradient (no SAGE)', file: 'saliencies/agent6_gradient_nosage_saliencies.json',   parse: d => d },
];
let currentSaliencyMethod = SALIENCY_METHODS[0].id;

// ── Heatmap color schemes ────────────────────────────────────────────────────
// Maps the normalized weight t∈[0,1] to RGB. The actual colormaps live in GLSL
// (see makeHeatmapMaterial → heatmapColor); `index` is the value written to the
// uColorScheme uniform that selects the colormap at render time. The order/index
// here MUST match the if-chain in the shader. Single source of truth for the
// dropdown — add a scheme in both places to extend.
const COLOR_SCHEMES = [
  { id: 'thermal',   label: 'Thermal Heatmap', index: 0 },
  { id: 'inferno',   label: 'Inferno',         index: 1 },
  { id: 'viridis',   label: 'Viridis',         index: 2 },
  { id: 'coolwarm',  label: 'Cool–Warm',       index: 3 },
  { id: 'grayscale', label: 'Grayscale',       index: 4 },
];
let currentColorScheme = COLOR_SCHEMES[0].id;
const colorSchemeIndex = id =>
  (COLOR_SCHEMES.find(s => s.id === id) || COLOR_SCHEMES[0]).index;

// --- Shared state ---
let gltfScene        = null;
let elecWorldPositions = null;  // THREE.Vector3[]
let elecGroup        = null;    // standard electrode dots
let BPelecGroup      = null;    // bipolar electrode dots
let elecWeights      = null;    // number[] weights, one per electrode
let heatmapMaterial  = null;    // current heatmap ShaderMaterial (for live tuning)


// ═══════════════════════════════════════════════════════════════════════════════
// DATA SOURCE TOGGLE
// ═══════════════════════════════════════════════════════════════════════════════
// FastAPI is TEMPORARILY DISCONNECTED. While true, the visualization renders from
// the local JSON files in this directory instead of POSTing to the endpoint.
// Set to false to reconnect the live FastAPI /display endpoint (submitTrainingParams).
const USE_LOCAL_JSON = true;

// ═══════════════════════════════════════════════════════════════════════════════
// FASTAPI — TRAINING PARAMETER SUBMISSION  (disconnected while USE_LOCAL_JSON = true)
// ═══════════════════════════════════════════════════════════════════════════════
const FASTAPI_ENDPOINT = 'http://localhost:8000/display';   // SERVER URL GOES HERE

function getTrainingParams() {
  const get = id => document.getElementById(id);

  const parseLayers = (...ids) => ids
    .map(id => parseInt(get(id)?.value))
    .filter(v => Number.isFinite(v) && v > 0);

  const temperature = parseFloat(get('temperature')?.value ?? 0.5);

  return {
    encoderLayers: parseLayers('encLay0', 'encLay1', 'encLay2'),
    encEmbedding:  parseInt(get('encEmbed')?.value    ?? 64),
    decoderLayers: parseLayers('decLay0', 'decLay1', 'decLay2'),
    decEmbedding:  parseInt(get('decEmbedNew')?.value ?? 64),
    temperature,
    typeModel:     get('typeModel')?.value             ?? 'RNN',
    subject:       parseInt(get('subjectId')?.value   ?? 401),
    numEpochs:     Math.min(parseInt(get('numEpochs')?.value ?? 20), 1000),
  };
}

// POSTs training params and returns the parsed trainedDisplay response,
// or null if the endpoint isn't configured or the request fails.
async function submitTrainingParams() {
  if (!FASTAPI_ENDPOINT) {
    console.warn('[FastAPI] FASTAPI_ENDPOINT is not set — cannot fetch results.');
    return null;
  }

  const params = getTrainingParams();
  console.log('[FastAPI] Submitting:', params);

  try {
    const res = await fetch(FASTAPI_ENDPOINT, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(params),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);
    const data = await res.json();
    console.log('[FastAPI] Response:', data);
    return data;   // trainedDisplay: { electrodes, bipolarElectrodes, saliencies }
  } catch (err) {
    console.error('[FastAPI] Submit failed:', err);
    return null;
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// LOCAL JSON — temporary stand-in for the FastAPI /display response
// ═══════════════════════════════════════════════════════════════════════════════
// Returns the same trainedDisplay shape the endpoint would:
//   { electrodes, bipolarElectrodes, saliencies }
// pulled straight from the JSON files sitting next to this script. Saliencies
// come from the currently selected method (see SALIENCY_METHODS).
// Bipolar electrodes drive the heatmap (see runVisualization).
async function loadLocalDisplayData() {
  try {
    const method = SALIENCY_METHODS.find(m => m.id === currentSaliencyMethod) || SALIENCY_METHODS[0];
    const [electrodes, bipolarElectrodes, salRaw] = await Promise.all([
      fetch('elecs.json').then(r => r.json()),
      fetch('bipolar_electrodes.json').then(r => r.json()),
      fetch(method.file).then(r => r.json()),
    ]);
    const saliencies = method.parse(salRaw);
    console.log('[Local] Loaded JSON:', {
      method:            method.id,
      electrodes:        electrodes.length,
      bipolarElectrodes: bipolarElectrodes.length,
      saliencies:        saliencies.length,
    });
    return { electrodes, bipolarElectrodes, saliencies };
  } catch (err) {
    console.error('[Local] Failed to load JSON data:', err);
    return null;
  }
}

// Switch saliency method. If a run is already on screen, swap the weights and
// re-bake the heatmap in place (no electrode-geometry rebuild); otherwise the
// choice applies on the next Run Model click.
async function setSaliencyMethod(id) {
  const method = SALIENCY_METHODS.find(m => m.id === id);
  if (!method) return;
  currentSaliencyMethod = id;

  if (BPelecGroup && elecWorldPositions) {
    try {
      const raw = await fetch(method.file).then(r => r.json());
      elecWeights = method.parse(raw);
      bakeVertexHeat();   // re-color the brain for the new method
      console.log(`[Saliency] switched to ${method.id}`);
    } catch (err) {
      console.error('[Saliency] load failed:', err);
    }
  }
}
if (typeof window !== 'undefined') window.setSaliencyMethod = setSaliencyMethod;

// Switch heatmap color scheme. Cheap: just updates the uColorScheme uniform on
// the live material, so it recolors immediately without re-baking heat or
// rebuilding geometry. Takes effect on the next render frame; if no run is on
// screen yet, the choice is remembered and applied when the heatmap is built.
function setColorScheme(id) {
  const scheme = COLOR_SCHEMES.find(s => s.id === id);
  if (!scheme) return;
  currentColorScheme = id;
  if (heatmapMaterial) {
    heatmapMaterial.uniforms.uColorScheme.value = scheme.index;
    console.log(`[ColorScheme] switched to ${scheme.id}`);
  }
}
if (typeof window !== 'undefined') window.setColorScheme = setColorScheme;

// ═══════════════════════════════════════════════════════════════════════════════
// SHADERS — shared vertex shader, two fragment shaders
// ═══════════════════════════════════════════════════════════════════════════════

// Passed to both materials. Computes world-space position and normal for lighting.
const vertexShader = `
  varying vec3 vWorldPos;
  varying vec3 vWorldNormal;

  void main() {
    vec4 worldPos = modelMatrix * vec4(position, 1.0);
    vWorldPos = worldPos.xyz;
    vWorldNormal = normalize(mat3(modelMatrix) * normal);
    gl_Position = projectionMatrix * viewMatrix * worldPos;
  }
`;

// Reusable lighting helper — inlined into both fragment shaders via template literal.
const lightingGLSL = `
  vec3 applyLighting(vec3 baseColor, vec3 n) {
    vec3 key  = normalize(vec3( 0.5,  1.0,  0.8));
    vec3 fill = normalize(vec3(-1.0,  0.3,  0.4));
    vec3 back = normalize(vec3(-0.2, -0.4, -1.0));

    float kDiff = max(dot(n, key),  0.0);
    float fDiff = max(dot(n, fill), 0.0) * 0.28;
    float bDiff = max(dot(n, back), 0.0) * 0.35;

    vec3 viewDir = normalize(cameraPosition - vWorldPos);
    float fresnel = pow(1.0 - max(dot(n, viewDir), 0.0), 3.5);

    float ambient  = 0.42;
    float lighting = ambient + kDiff * 0.55 + fDiff + bDiff;

    vec3 lit = baseColor * lighting;
    lit += fresnel * 0.15 * vec3(0.85, 0.92, 1.0);
    return clamp(lit, 0.0, 1.0);
  }
`;

// --- Default material: brain-coloured, no heatmap, no electrodes needed ---
// Applied immediately on GLTF load. Fast to compile, no texture sampling.
const defaultFragmentShader = `
  varying vec3 vWorldPos;
  varying vec3 vWorldNormal;

  ${lightingGLSL}

  void main() {
    vec3 n = normalize(vWorldNormal);
    if (!gl_FrontFacing) n = -n;
    vec3 baseColor = vec3(0.80, 0.70, 0.66);   // warm pinkish-gray, brain-like
    gl_FragColor = vec4(applyLighting(baseColor, n), 1.0);
  }
`;

function makeDefaultMaterial() {
  return new THREE.ShaderMaterial({
    vertexShader,
    fragmentShader: defaultFragmentShader,
    side: THREE.DoubleSide,
  });
}

// --- Heatmap material: per-VERTEX baked (cheap to render) ---
// The expensive K-nearest-electrode blend is computed ONCE on the CPU per vertex
// (see bakeVertexHeat) and stored in the `aHeat` vertex attribute. The GPU just
// interpolates aHeat across triangles and maps it through the colormap +
// lighting — no per-fragment electrode loop, no texture fetches. This is the key
// performance fix: per-frame cost drops from O(fragments × electrodes) to ~O(1).
const heatmapVertexShader = `
  attribute float aHeat;
  varying vec3  vWorldPos;
  varying vec3  vWorldNormal;
  varying float vHeat;

  void main() {
    vec4 worldPos = modelMatrix * vec4(position, 1.0);
    vWorldPos     = worldPos.xyz;
    vWorldNormal  = normalize(mat3(modelMatrix) * normal);
    vHeat         = aHeat;
    gl_Position   = projectionMatrix * viewMatrix * worldPos;
  }
`;

function makeHeatmapMaterial() {
  const fragmentShader = `
    varying vec3  vWorldPos;
    varying vec3  vWorldNormal;
    varying float vHeat;

    uniform int uColorScheme;   // selects the colormap (see COLOR_SCHEMES)

    ${lightingGLSL}

    // 0 — Thermal: blue → cyan → green → yellow → red (default)
    vec3 cmThermal(float s) {
      if (s < 0.25) return mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 1.0), s * 4.0);
      if (s < 0.50) return mix(vec3(0.0, 1.0, 1.0), vec3(0.0, 1.0, 0.0), (s - 0.25) * 4.0);
      if (s < 0.75) return mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0), (s - 0.50) * 4.0);
      return           mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), (s - 0.75) * 4.0);
    }

    // 1 — Inferno (approx): black → violet → magenta → orange → pale yellow
    vec3 cmInferno(float s) {
      if (s < 0.25) return mix(vec3(0.1,0.1,0.1),    vec3(0.30,0.05,0.40), s * 4.0);
      if (s < 0.50) return mix(vec3(0.30,0.05,0.40), vec3(0.73,0.21,0.33), (s - 0.25) * 4.0);
      if (s < 0.75) return mix(vec3(0.73,0.21,0.33), vec3(0.98,0.55,0.04), (s - 0.50) * 4.0);
      return           mix(vec3(0.98,0.55,0.04),     vec3(0.99,1.00,0.64), (s - 0.75) * 4.0);
    }

    // 2 — Viridis (approx): deep purple → blue → teal → green → yellow
    vec3 cmViridis(float s) {
      if (s < 0.25) return mix(vec3(0.27,0.00,0.33), vec3(0.23,0.32,0.55), s * 4.0);
      if (s < 0.50) return mix(vec3(0.23,0.32,0.55), vec3(0.13,0.57,0.55), (s - 0.25) * 4.0);
      if (s < 0.75) return mix(vec3(0.13,0.57,0.55), vec3(0.37,0.79,0.38), (s - 0.50) * 4.0);
      return           mix(vec3(0.37,0.79,0.38),     vec3(0.99,0.91,0.14), (s - 0.75) * 4.0);
    }

    // 3 — Cool–Warm diverging: blue → light gray → red
    vec3 cmCoolWarm(float s) {
      if (s < 0.5) return mix(vec3(0.23,0.30,0.75), vec3(0.87,0.87,0.87), s * 2.0);
      return           mix(vec3(0.87,0.87,0.87),    vec3(0.71,0.02,0.15), (s - 0.5) * 2.0);
    }

    // 4 — Grayscale: black → white
    vec3 cmGray(float s) { return vec3(s); }

    vec3 heatmapColor(float t) {
      float s = clamp(t, 0.0, 1.0);
      if (uColorScheme == 1) return cmInferno(s);
      if (uColorScheme == 2) return cmViridis(s);
      if (uColorScheme == 3) return cmCoolWarm(s);
      if (uColorScheme == 4) return cmGray(s);
      return cmThermal(s);
    }

    void main() {
      vec3 baseColor = heatmapColor(vHeat);

      vec3 n = normalize(vWorldNormal);
      if (!gl_FrontFacing) n = -n;

      gl_FragColor = vec4(applyLighting(baseColor, n), 1.0);
    }
  `;

  return new THREE.ShaderMaterial({
    uniforms: {
      uColorScheme: { value: colorSchemeIndex(currentColorScheme) },
    },
    vertexShader: heatmapVertexShader,
    fragmentShader,
    side: THREE.DoubleSide,
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
// ELECTRODE GEOMETRY + HEATMAP — runs only when Run Model is clicked
// ═══════════════════════════════════════════════════════════════════════════════

const scaleFactor = 0.014;
const xOffset = -0.13;
const yOffset = -0.2;
const zOffset =  0.2;

function snapElectrodesToSurface() {
  const brainMeshes = [];
  gltfScene.traverse(child => { if (child.isMesh) brainMeshes.push(child); });
  gltfScene.updateMatrixWorld(true);

  brainMeshes.forEach(m => {
    const mats = Array.isArray(m.material) ? m.material : [m.material];
    mats.forEach(mat => { if (mat) mat.side = THREE.DoubleSide; });
  });

  // ── Surface snapping (single-calc, nearest-vertex) ──────────────────────────
  // Used ONLY to project electrode positions onto the cortical surface for the
  // heatmap weight contribution — the rendered electrode dots are left untouched
  // at their original grid positions. Same precompute-once idea as bakeVertexHeat:
  // gather the brain's surface vertices in world space ONCE, then map each
  // electrode to its nearest surface vertex with a single distance pass.
  // Cost is O(electrodes × vertices) one-time. Precision is bounded by mesh
  // vertex density (same trade-off as the per-vertex heatmap).

  // Gather every brain surface vertex in world space, once.
  const brainVerts = [];                 // flat [x,y,z, x,y,z, ...]
  const _v = new THREE.Vector3();
  brainMeshes.forEach(mesh => {
    const pos = mesh.geometry && mesh.geometry.attributes.position;
    if (!pos) return;
    mesh.updateWorldMatrix(true, false);
    const wm = mesh.matrixWorld;
    for (let i = 0; i < pos.count; i++) {
      _v.fromBufferAttribute(pos, i).applyMatrix4(wm);
      brainVerts.push(_v.x, _v.y, _v.z);
    }
  });

  // Return each electrode's nearest brain-surface vertex as a NEW position array,
  // WITHOUT moving the dots (they stay on their grid). Falls back to the dot's own
  // position if no surface vertex is found.
  const snappedPositions = group => {
    const out = [];
    let misses = 0;
    group.children.forEach(dot => {
      const ex = dot.position.x, ey = dot.position.y, ez = dot.position.z;

      let bestD2 = Infinity, bx = ex, by = ey, bz = ez, found = false;
      for (let k = 0; k < brainVerts.length; k += 3) {
        const dx = ex - brainVerts[k], dy = ey - brainVerts[k + 1], dz = ez - brainVerts[k + 2];
        const d2 = dx * dx + dy * dy + dz * dz;     // squared distance — no sqrt in the hot loop
        if (d2 < bestD2) {
          bestD2 = d2;
          bx = brainVerts[k]; by = brainVerts[k + 1]; bz = brainVerts[k + 2];
          found = true;
        }
      }

      if (!found) misses++;
      out.push(new THREE.Vector3(bx, by, bz));      // snapped (or original if no hit)
    });
    if (misses) console.warn(`[snap] ${misses}/${group.children.length} electrodes found no surface vertex`);
    return out;
  };

  // Only the bipolar set drives the heatmap, so only it is snapped — and only for
  // the weight contribution. The dots themselves keep their grid positions.
  elecWorldPositions = BPelecGroup ? snappedPositions(BPelecGroup) : [];
}

// Current nearest-electrode blend share (mutable; see setNearestWeight).
let nearestWeightValue = NEAREST_WEIGHT;

// CPU bake: for every brain vertex, blend the K nearest electrodes' weights and
// write the result into the `aHeat` attribute. Runs once on Run Model and again
// whenever the blend share changes — O(vertices × electrodes), one-time, never
// per frame. Mirrors the math the old per-fragment shader did, just at vertices.
function bakeVertexHeat() {
  if (!gltfScene || !elecWorldPositions) return;

  const E = elecWorldPositions.length;
  const ex = new Float64Array(E);
  const ey = new Float64Array(E);
  const ez = new Float64Array(E);

  // Normalize saliencies to [0,1] (by max |value|) so the different methods are
  // visually comparable regardless of their native scale; missing → 0 (cold).
  let maxAbs = 0;
  if (elecWeights) {
    for (let i = 0; i < E; i++) {
      const a = Math.abs(elecWeights[i]);
      if (Number.isFinite(a) && a > maxAbs) maxAbs = a;
    }
  }
  const scale = (maxAbs > 0 ? 1 / maxAbs : 1) * SALIENCY_GAIN;

  const ew = new Float64Array(E);   // normalized saliency weight (~[0,1] × gain)
  elecWorldPositions.forEach((p, i) => {
    ex[i] = p.x; ey[i] = p.y; ez[i] = p.z;
    ew[i] = (elecWeights && elecWeights[i] != null) ? Math.abs(elecWeights[i] ** 2) * scale : 0.0;
  });

  const K       = Math.min(K_NEAREST, E);
  const nw      = Math.min(Math.max(nearestWeightValue, 0), 1);
  const maxDist = DIST_MAX;

  const kD2 = new Float64Array(K);  // squared distances
  const kW  = new Float64Array(K);
  const v   = new THREE.Vector3();

  gltfScene.traverse(child => {
    if (!child.isMesh || !child.geometry || !child.geometry.attributes.position) return;
    child.updateWorldMatrix(true, false);
    const m     = child.matrixWorld;
    const pos   = child.geometry.attributes.position;
    const count = pos.count;
    const heat  = new Float32Array(count);

    for (let vi = 0; vi < count; vi++) {
      v.fromBufferAttribute(pos, vi).applyMatrix4(m);
      const vx = v.x, vy = v.y, vz = v.z;

      // Gather K nearest electrodes (unordered slots; evict the current farthest).
      for (let s = 0; s < K; s++) { kD2[s] = Infinity; kW[s] = 0; }
      for (let j = 0; j < E; j++) {
        const dx = vx - ex[j], dy = vy - ey[j], dz = vz - ez[j];
        const d2 = dx * dx + dy * dy + dz * dz;
        let worst = 0, worstD = kD2[0];
        for (let s = 1; s < K; s++) { if (kD2[s] > worstD) { worstD = kD2[s]; worst = s; } }
        if (d2 < worstD) { kD2[worst] = d2; kW[worst] = ew[j]; }
      }

      // Nearest = smallest slot.
      let minS = 0, minD2 = kD2[0];
      for (let s = 1; s < K; s++) { if (kD2[s] < minD2) { minD2 = kD2[s]; minS = s; } }
      const nearestW = kW[minS];
      const minDist  = Math.sqrt(minD2);

      // Surrounding (K-1): inverse-distance-squared blend of their weights.
      let surrSum = 0, surrNorm = 0;
      for (let s = 0; s < K; s++) {
        if (s === minS || !isFinite(kD2[s])) continue;
        const inf = 1 / (kD2[s] + 1e-6);
        surrSum  += kW[s] * inf;
        surrNorm += inf;
      }
      const surroundW    = surrNorm > 0 ? surrSum / surrNorm : nearestW;
      const smoothWeight = nw * nearestW + (1 - nw) * surroundW;

      let falloff = 1 - minDist / maxDist;
      if (falloff < 0) falloff = 0;
      heat[vi] = falloff * smoothWeight;
    }

    child.geometry.setAttribute('aHeat', new THREE.BufferAttribute(heat, 1));
  });
}

function applyHeatmapMaterial() {
  bakeVertexHeat();
  const mat = makeHeatmapMaterial();
  heatmapMaterial = mat;
  gltfScene.traverse(child => { if (child.isMesh) child.material = mat; });
}

// Adjust the nearest electrode's blend share (0..1). Re-bakes the per-vertex
// heat (a one-time CPU pass), since the blend is precomputed rather than
// per-fragment. Call from the console: setNearestWeight(0.7)
function setNearestWeight(w) {
  nearestWeightValue = Math.min(Math.max(Number(w), 0), 1);
  bakeVertexHeat();   // setAttribute replaces the buffer → GPU re-uploads
  return nearestWeightValue;
}
if (typeof window !== 'undefined') window.setNearestWeight = setNearestWeight;

// Builds the electrode visualisation and heatmap from a trainedDisplay response:
//   data.electrodes        — List[List[float]]  standard electrode MNI coords [x, y, z]
//   data.bipolarElectrodes — List[List[float]]  bipolar electrode MNI coords  [x, y, z]
//   data.saliencies        — List[float]         one weight per bipolar electrode
function runVisualization(data) {
  if (!gltfScene) {
    console.warn('[runViz] Brain model not loaded yet — try again in a moment.');
    return;
  }

  // Remove any electrode groups from a previous run
  if (elecGroup)   { scene.remove(elecGroup);   elecGroup   = null; }
  if (BPelecGroup) { scene.remove(BPelecGroup); BPelecGroup = null; }

  elecWeights = data.saliencies;

  // MNI [x, y, z] → Three.js world position
  const mniToWorld = ([x, z, y]) => new THREE.Vector3(
    xOffset + -1 * x * scaleFactor,
    yOffset +       y * scaleFactor,
    zOffset +       z * scaleFactor
  );

  // --- Standard electrodes (visual only) ---
  const elecGeo = new THREE.SphereGeometry(scaleFactor * 0.5, scaleFactor * 0.5, scaleFactor * 0.5);
  const elecMat = new THREE.MeshBasicMaterial({ color: 0x00ffcc });
  const electrodes = new THREE.Group();
  data.electrodes.forEach(coord => {
    const dot = new THREE.Mesh(elecGeo, elecMat);
    dot.position.copy(mniToWorld(coord));
    electrodes.add(dot);
  });
  elecGroup = electrodes;
  scene.add(electrodes);

  // --- Bipolar electrodes (drive the heatmap) ---
  const bpGeo = new THREE.SphereGeometry(scaleFactor * 0.7, scaleFactor * 0.7, scaleFactor * 0.7);
  const bpMat = new THREE.MeshBasicMaterial({ color: 0xff0011 });
  const bpGroup = new THREE.Group();
  elecWorldPositions = data.bipolarElectrodes.map(coord => {
    const pos = mniToWorld(coord);
    const dot = new THREE.Mesh(bpGeo, bpMat);
    dot.position.copy(pos);
    bpGroup.add(dot);
    return pos;
  });
  BPelecGroup = bpGroup;
  scene.add(bpGroup);

  if (elecWeights.length !== elecWorldPositions.length) {
    console.warn(`saliency count (${elecWeights.length}) != bipolar electrode count (${elecWorldPositions.length})`);
  }

  snapElectrodesToSurface();
  applyHeatmapMaterial();
}

// ═══════════════════════════════════════════════════════════════════════════════
// DOM WIRING
// ═══════════════════════════════════════════════════════════════════════════════

setTimeout(() => {
  // Live learning-rate display
  const tempEl    = document.getElementById('temperature');
  const lrEl      = document.getElementById('lrDisplay');
  const tempSubEl = document.getElementById('tempDisplay');
  if (tempEl) {
    const updateLr = () => {
      const t  = parseFloat(tempEl.value);
      const lr = 0.001 * Math.pow(100, t);
      if (lrEl)      lrEl.textContent     = lr.toFixed(4).replace(/\.?0+$/, '') || '0';
      if (tempSubEl) tempSubEl.textContent = `temperature = ${t.toFixed(2)}`;
    };
    tempEl.addEventListener('input', updateLr);
    updateLr();
  }

  // Run Model: load trainedDisplay → render.
  // While USE_LOCAL_JSON is true this reads the local JSON files instead of
  // POSTing to the (temporarily disconnected) FastAPI endpoint.
  document.getElementById('runModelBtn')?.addEventListener('click', async () => {
    const result = USE_LOCAL_JSON
      ? await loadLocalDisplayData()
      : await submitTrainingParams();
    if (result) runVisualization(result);
  });

  // Toggle electrodes visibility
  const toggleBtn = document.getElementById('toggleElecBtn');
  if (toggleBtn) {
    let electrodesVisible = true;
    toggleBtn.addEventListener('click', () => {
      electrodesVisible = !electrodesVisible;
      if (elecGroup)   elecGroup.visible   = electrodesVisible;
      if (BPelecGroup) BPelecGroup.visible = electrodesVisible;
      toggleBtn.classList.toggle('hidden', !electrodesVisible);
      toggleBtn.textContent = electrodesVisible ? 'Electrodes ●' : 'Electrodes ○';
    });
  }

  // Saliency-method selector — populated from the registry (single source of truth)
  const salSelect = document.getElementById('saliencyMethod');
  if (salSelect) {
    salSelect.innerHTML = SALIENCY_METHODS
      .map(m => `<option value="${m.id}">${m.label}</option>`)
      .join('');
    salSelect.value = currentSaliencyMethod;
    salSelect.addEventListener('change', () => setSaliencyMethod(salSelect.value));
  }

  // Color-scheme selector — populated from COLOR_SCHEMES (single source of truth)
  const csSelect = document.getElementById('colorScheme');
  if (csSelect) {
    csSelect.innerHTML = COLOR_SCHEMES
      .map(s => `<option value="${s.id}">${s.label}</option>`)
      .join('');
    csSelect.value = currentColorScheme;
    csSelect.addEventListener('change', () => setColorScheme(csSelect.value));
  }
}, 0);

// ═══════════════════════════════════════════════════════════════════════════════
// GLTF — loads on page open, applies default material immediately
// ═══════════════════════════════════════════════════════════════════════════════

loader.load(
  'human_brain_PittBrains/scene.gltf',
  function (gltf) {
    gltfScene = gltf.scene;

    // Apply lightweight default shader — no electrode data needed, compiles fast
    const defaultMat = makeDefaultMaterial();
    gltfScene.traverse(child => { if (child.isMesh) child.material = defaultMat; });

    scene.add(gltfScene);

    const box    = new THREE.Box3().setFromObject(gltfScene);
    const center = box.getCenter(new THREE.Vector3());
    controls.target.copy(center);
    cameraRot.position.set(center.x, center.y, center.z + 100);
    controls.update();
  },
  xhr => console.log((100 * xhr.loaded / xhr.total).toFixed(0) + '% loaded'),
  err => console.error(err)
);

// ═══════════════════════════════════════════════════════════════════════════════
// RENDERER + CONTROLS
// ═══════════════════════════════════════════════════════════════════════════════

const container3D = document.getElementById("container3D");
const renderer = new THREE.WebGLRenderer({ alpha: true });
const initW = container3D.clientWidth  || window.innerWidth;
const initH = container3D.clientHeight || window.innerHeight;
renderer.setSize(initW, initH);
renderer.setPixelRatio(window.devicePixelRatio);
container3D.appendChild(renderer.domElement);
cameraRot.position.z = 50;

controls = new OrbitControls(cameraRot, renderer.domElement);

function animate() {
  requestAnimationFrame(animate);
  renderer.render(scene, cameraRot);
}

const ro = new ResizeObserver(entries => {
  const { width, height } = entries[0].contentRect;
  if (width > 0 && height > 0) {
    renderer.setSize(width, height);
    cameraRot.aspect = width / height;
    cameraRot.updateProjectionMatrix();
  }
});
ro.observe(container3D);

animate();
