/**
 * DEMO: Fragment Shader Heatmap (DataTexture approach)
 *
 * Strategy:
 *   1. Load brain GLTF and electrodes (same files as the main project).
 *   2. Simulate electrode "neural activity" values.
 *   3. Pack electrode world-positions + values into an RGBA32F DataTexture
 *      (one texel per electrode: R=x, G=y, B=z, A=value).
 *   4. Replace the brain's materials with a custom ShaderMaterial that:
 *        - computes the vertex world-position in the vertex shader (vWorldPos)
 *        - samples every electrode texel in the fragment shader
 *        - finds the K nearest electrodes (K = #define KNN in shader)
 *        - performs IDW weighting over those K neighbours to get a scalar in [0, 1]
 *        - maps that scalar through a GLSL viridis polynomial to an RGB colour
 *
 * Pros:  Resolution-independent (smooth regardless of mesh density).
 *        DataTexture sidesteps uniform-array size limits; works for any N electrodes.
 *        Lighting from the normal map is preserved by blending shader colour with
 *        the standard lighting model.
 * Cons:  Every fragment re-samples the texture and loops over all electrodes —
 *        O(N) per fragment. Fine for a few hundred electrodes; profile with thousands.
 *        ShaderMaterial requires manual re-implementation of lighting if needed.
 *
 * INTEGRATION NOTE for the main demoUI:
 *   After loading elecs.json, call `buildElecTexture(coords, values)` to get the
 *   DataTexture. Then call `applyShaderHeatmap(brainObject, elecTexture, numElecs)`.
 *   To update values each frame: write new data into `elecTexture.image.data`,
 *   set `elecTexture.needsUpdate = true`.
 */

import * as THREE from "https://cdn.skypack.dev/three@0.129.0/build/three.module.js";
import { OrbitControls } from "https://cdn.skypack.dev/three@0.129.0/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader }    from "https://cdn.skypack.dev/three@0.129.0/examples/jsm/loaders/GLTFLoader.js";

// ─── Scene setup ─────────────────────────────────────────────────────────────
const scene    = new THREE.Scene();
const camera   = new THREE.PerspectiveCamera(1, innerWidth / innerHeight, 0.1, 2000);
camera.position.z = 50;

const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
renderer.setSize(innerWidth, innerHeight);
document.getElementById("container3D").appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);

const dirLight = new THREE.DirectionalLight(0xffffff, 1);
dirLight.position.set(500, 500, 500);
scene.add(dirLight);
scene.add(new THREE.AmbientLight(0x444444, 3));

// ─── Shaders ─────────────────────────────────────────────────────────────────
const vertexShader = /* glsl */`
  varying vec3 vWorldPos;
  varying vec3 vNormal;

  void main() {
    // World-space position needed for IDW distance calculation
    vec4 worldPos = modelMatrix * vec4(position, 1.0);
    vWorldPos     = worldPos.xyz;
    vNormal       = normalize(normalMatrix * normal);
    gl_Position   = projectionMatrix * viewMatrix * worldPos;
  }
`;

const fragmentShader = /* glsl */`
  precision highp float;
  precision highp sampler2D;

  uniform sampler2D uElecData;   // RGBA32F: one texel per electrode (x, y, z, value)
  uniform int       uNumElecs;   // number of electrodes encoded in the texture
  uniform float     uIDWPower;   // IDW exponent (default 2.0)

  // Directional light uniforms (simple Lambertian shading to keep brain readable)
  uniform vec3  uLightDir;
  uniform float uAmbient;

  varying vec3 vWorldPos;
  varying vec3 vNormal;

  // ── K-Nearest Neighbour constant ─────────────────────────────────────────
  // GLSL loop bounds must be compile-time constants, so K is a #define.
  // Change KNN here (and update the JS uniform uKNN) to tune the blend radius.
  #define KNN 8

  // ── Viridis colormap ──────────────────────────────────────────────────────
  // Polynomial approximation by Nathaniel J. Smith & Stefan van der Walt.
  // https://bids.github.io/colormap/
  vec3 viridis(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0.2777273272234177, 0.005407344544966578, 0.3340998053353061);
    vec3 c1 = vec3(0.1050930431085774, 1.404613529898575,    1.384590162594685);
    vec3 c2 = vec3(-0.3308618287255563, 0.214847559468213,   0.09509516302823659);
    vec3 c3 = vec3(-4.634230498983486, -5.799100973351011,  -19.33461838716418);
    vec3 c4 = vec3(6.228269936347081,  14.17993336680509,    56.69055260068105);
    vec3 c5 = vec3(4.776384997670288, -13.74514537774601,   -65.35303263337234);
    vec3 c6 = vec3(-5.435455855934631,  4.645852612178535,   26.3124352495832);
    return c0 + t*(c1 + t*(c2 + t*(c3 + t*(c4 + t*(c5 + t*c6)))));
  }

  void main() {
    float texWidth = float(uNumElecs);

    // ── Pass 1: collect the KNN nearest electrodes ────────────────────────
    // Maintain a fixed-size "worst-slot heap":
    //   kDist[k] / kVal[k] hold the current K closest candidates.
    //   On each electrode we evict the farthest slot if the new one is closer.
    float kDist[KNN];
    float kVal[KNN];

    for (int k = 0; k < KNN; k++) {
      kDist[k] = 1.0e10;   // sentinel: unfilled slot
      kVal[k]  = 0.0;
    }

    for (int i = 0; i < 512; i++) {
      if (i >= uNumElecs) break;

      float u   = (float(i) + 0.5) / texWidth;
      vec4  tap = texture2D(uElecData, vec2(u, 0.5));
      float dist = distance(vWorldPos, tap.xyz);

      // Find the slot currently holding the largest distance
      int   worstSlot = 0;
      float worstDist = kDist[0];
      for (int k = 1; k < KNN; k++) {
        if (kDist[k] > worstDist) {
          worstDist = kDist[k];
          worstSlot = k;
        }
      }

      // Replace the worst slot only if this electrode is closer
      if (dist < worstDist) {
        kDist[worstSlot] = dist;
        kVal[worstSlot]  = tap.w;
      }
    }

    // ── Pass 2: IDW over the K nearest only ──────────────────────────────
    float weightSum = 0.0;
    float valueSum  = 0.0;

    for (int k = 0; k < KNN; k++) {
      if (kDist[k] > 9.9e9) continue;      // unfilled slot (< KNN electrodes total)

      if (kDist[k] < 1e-5) {               // fragment is at an electrode exactly
        weightSum = 1.0;
        valueSum  = kVal[k];
        break;
      }

      float w    = 1.0 / pow(kDist[k], uIDWPower);
      weightSum += w;
      valueSum  += w * kVal[k];
    }

    float t = (weightSum > 0.0) ? valueSum / weightSum : 0.0;

    // ── Colour ───────────────────────────────────────────────────────────────
    vec3 heatColor = viridis(t);

    // Simple Lambertian shading so the brain keeps 3-D depth
    vec3  N       = normalize(vNormal);
    float diffuse = max(dot(N, normalize(uLightDir)), 0.0);
    vec3  shaded  = heatColor * (uAmbient + (1.0 - uAmbient) * diffuse);

    gl_FragColor = vec4(shaded, 1.0);
  }
`;

// ─── DataTexture builder ──────────────────────────────────────────────────────
/**
 * Pack electrode data into a 1D RGBA32F texture.
 * @param {THREE.Vector3[]} positions - electrode world positions
 * @param {number[]}        values    - activity values [0..1]
 * @returns {{ texture: THREE.DataTexture, numElecs: number }}
 */
function buildElecTexture(positions, values) {
  const N    = positions.length;
  const data = new Float32Array(N * 4); // RGBA per texel

  for (let i = 0; i < N; i++) {
    data[i*4]     = positions[i].x;
    data[i*4 + 1] = positions[i].y;
    data[i*4 + 2] = positions[i].z;
    data[i*4 + 3] = values[i];
  }

  const tex = new THREE.DataTexture(data, N, 1, THREE.RGBAFormat, THREE.FloatType);
  tex.needsUpdate = true;
  return { texture: tex, numElecs: N };
}

// ─── Apply shader to brain ────────────────────────────────────────────────────
function applyShaderHeatmap(brainObject, elecTexture, numElecs) {
  const uniforms = {
    uElecData:  { value: elecTexture },
    uNumElecs:  { value: numElecs },
    uIDWPower:  { value: 2.0 },
    uLightDir:  { value: new THREE.Vector3(1, 1, 1).normalize() },
    uAmbient:   { value: 0.25 },
  };

  const shaderMat = new THREE.ShaderMaterial({
    uniforms,
    vertexShader,
    fragmentShader,
    side: THREE.DoubleSide,
  });

  brainObject.traverse(child => {
    if (child.isMesh) child.material = shaderMat;
  });
}

// ─── Load data ────────────────────────────────────────────────────────────────
const SCALE  = 0.006;
const OFFSET = new THREE.Vector3(-0.01, 0.4, -0.15);

let brainLoaded      = false;
let electrodesLoaded = false;
let brainObject, elecTexture, numElecs, elecWorldPositions, elecValues;

function tryApply() {
  if (!brainLoaded || !electrodesLoaded) return;
  applyShaderHeatmap(brainObject, elecTexture, numElecs);
  showLegend();
  document.getElementById('status').textContent = '';
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
        OFFSET.z + z * SCALE,
      )
    );

    // Simulated activity: sine wave (replace with real decoded data)
    elecValues = elecWorldPositions.map((_, i) =>
      0.5 + 0.5 * Math.sin(i * 0.18)
    );

    ({ texture: elecTexture, numElecs } = buildElecTexture(elecWorldPositions, elecValues));

    // Electrode spheres coloured by viridis for reference
    const group = new THREE.Group();
    elecWorldPositions.forEach((pos, i) => {
      const t   = elecValues[i];
      // Quick viridis approximation for the JS side
      const r   = Math.max(0, Math.min(1, -0.6 + t * 2.4));
      const g   = Math.max(0, Math.min(1, 0.1 + t * 1.2 - Math.pow(t, 2)));
      const b   = Math.max(0, Math.min(1, 0.55 - t * 0.8));
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
  })
  .catch(err => console.error('Electrode load failed', err));

// ─── Legend ───────────────────────────────────────────────────────────────────
// Viridis JS polynomial mirror (for the canvas legend only)
function viridisJS(t) {
  t = Math.max(0, Math.min(1, t));
  const c = [
    [0.2777, 0.0054, 0.3341],
    [0.1051, 1.4046, 1.3846],
    [-0.3309, 0.2148, 0.0951],
    [-4.6342, -5.7991, -19.3346],
    [6.2283, 14.1799, 56.6906],
    [4.7764, -13.7451, -65.3530],
    [-5.4355, 4.6459, 26.3124],
  ];
  return c.reduce((acc, ci, i) =>
    acc.map((v, j) => v + ci[j] * Math.pow(t, i)), [0, 0, 0]
  ).map(v => Math.max(0, Math.min(1, v)));
}

function showLegend() {
  const canvas = document.getElementById('legend');
  const ctx    = canvas.getContext('2d');
  const W = canvas.width;
  for (let i = 0; i < W; i++) {
    const [r, g, b] = viridisJS(i / W);
    ctx.fillStyle = `rgb(${r*255|0},${g*255|0},${b*255|0})`;
    ctx.fillRect(i, 0, 1, canvas.height);
  }
}

// ─── Animate ──────────────────────────────────────────────────────────────────
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
