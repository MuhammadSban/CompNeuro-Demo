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

// --- Shared load state ---
let gltfScene = null;
let elecWorldPositions = null; // THREE.Vector3[]
let elecGroup = null;          // THREE.Group of electrode dot meshes
let BPelecGroup = null;        // THREE.Group of bipolar reference electrode dot meshes 
let elecWeights = null;        // number[] from FastAPI, one per electrode (defaults to 1.0)

// Source of per-electrode weights: a JSON list of numbers (0..1), one per electrode.
// Currently a local export; swap back to a FastAPI URL when serving live weights.
const WEIGHTS_URL = 'electrodeimportances.json';


// ═══════════════════════════════════════════════════════════════════════════════
// FASTAPI — TRAINING PARAMETER SUBMISSION
// ═══════════════════════════════════════════════════════════════════════════════
//
// Fill in FASTAPI_ENDPOINT before use, e.g.:
//   'http://localhost:8000/train'
//
// getTrainingParams() reads the slider values from the DOM.
// submitTrainingParams() POSTs them as JSON and returns the parsed response.
// The "Run Model" button in demoUI.html triggers submitTrainingParams().
//
// Sliders that are still TBD (lr, param5, param6) are disabled in the UI and
// will send their raw numeric value — update their encoding here once finalised.
// ═══════════════════════════════════════════════════════════════════════════════

const FASTAPI_ENDPOINT = '';   // TODO: set server URL

function getTrainingParams() {
  const get = id => document.getElementById(id);
  return {
    epochs:        parseInt(get('epochs')?.value   ?? 10),   // 1–20
    decoder_embed: parseInt(get('decEmbed')?.value ?? 100),  // 1–200
    decoder_rnn:   parseInt(get('decRNN')?.value   ?? 400),  // 1–800
    learning_rate: parseFloat(get('lr')?.value     ?? 50),   // TODO: encoding TBD
    param5:        parseFloat(get('param5')?.value ?? 50),   // future use
    param6:        parseFloat(get('param6')?.value ?? 50),   // future use
  };
}

async function submitTrainingParams() {
  if (!FASTAPI_ENDPOINT) {
    console.warn('[FastAPI] FASTAPI_ENDPOINT is not configured — params not sent.');
    console.log('[FastAPI] Would have sent:', getTrainingParams());
    return;
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
    return data;
  } catch (err) {
    console.error('[FastAPI] Submit failed:', err);
  }
}

// Wire the "Run Model" button once the DOM is ready.
// Using setTimeout(0) defers until after the module's synchronous setup,
// which ensures the button element exists even though the script tag is mid-body.
setTimeout(() => {
  document.getElementById('runModelBtn')
    ?.addEventListener('click', submitTrainingParams);
}, 0);

// ═══════════════════════════════════════════════════════════════════════════════


function snapElectrodesToSurface() {
  const brainMeshes = [];
  gltfScene.traverse(child => { if (child.isMesh) brainMeshes.push(child); });
  gltfScene.updateMatrixWorld(true);

  const brainCenter = new THREE.Box3()
    .setFromObject(gltfScene)
    .getCenter(new THREE.Vector3());

  const raycaster = new THREE.Raycaster();

  // Helper: snap every dot in a group to the nearest brain surface point
  function snapGroup(group) {
    group.children.forEach(dot => {
      const elecPos = dot.position.clone();
      const outDir = new THREE.Vector3().subVectors(elecPos, brainCenter).normalize();

      let bestPoint = null;
      let bestDist  = Infinity;

      // Cast outward then inward; take whichever surface hit is nearest
      
      [outDir, outDir.clone().negate()].forEach(dir => {
        raycaster.set(elecPos, dir);
        raycaster.intersectObjects(brainMeshes, false).forEach(hit => {
          if (hit.distance < bestDist) {
            bestDist  = hit.distance;
            bestPoint = hit.point;
          }
        });
      });

      if (bestPoint) dot.position.copy(bestPoint);
      
    });
  }

  // Snap both electrode sets so all dots sit on the surface visually
  if (elecGroup)   snapGroup(elecGroup);
  if (BPelecGroup) snapGroup(BPelecGroup);

  // Only bipolar positions drive the heatmap texture
  elecWorldPositions = BPelecGroup.children.map(dot => dot.position.clone());
}

function tryApplyDistanceMaterial() {
  if (!gltfScene || !BPelecGroup) return;

  snapElectrodesToSurface();

  const n = elecWorldPositions.length;

  // Pack electrode positions into an n×1 RGBA float texture
  const texData = new Float32Array(n * 4);
  elecWorldPositions.forEach((p, i) => {
    texData[i * 4]     = p.x;
    texData[i * 4 + 1] = p.y;
    texData[i * 4 + 2] = p.z;
    // Alpha channel carries the per-electrode weight (NN output).
    // Falls back to 1.0 until weights have been fetched.
    texData[i * 4 + 3] = (elecWeights && elecWeights[i] != null) ? elecWeights[i] : 1.0;
  });
  const electrodesTex = new THREE.DataTexture(texData, n, 1, THREE.RGBAFormat, THREE.FloatType);
  electrodesTex.needsUpdate = true;

  // NUM_ELECTRODES baked in as a #define so the loop bound is a compile-time constant
  const fragmentShader = `
    #define NUM_ELECTRODES ${n}

    uniform sampler2D electrodesTex;
    uniform float maxDist;

    varying vec3 vWorldPos;
    varying vec3 vWorldNormal;

    // t=0 → blue (far from all electrodes), t=1 → red (closest)
    vec3 heatmapColor(float t) {
      float s = clamp(t, 0.0, 1.0);
      if (s < 0.25) return mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 1.0), s * 4.0);
      if (s < 0.50) return mix(vec3(0.0, 1.0, 1.0), vec3(0.0, 1.0, 0.0), (s - 0.25) * 4.0);
      if (s < 0.75) return mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0), (s - 0.50) * 4.0);
      return           mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), (s - 0.75) * 4.0);
    }

    void main() {
      float minDist = 1.0e10;
      float closestAlpha = 1.0;
      for (int i = 0; i < NUM_ELECTRODES; i++) {
        float u = (float(i) + 0.5) / float(NUM_ELECTRODES);
        vec4 texel = texture2D(electrodesTex, vec2(u, 0.5));
        vec3 ePos = texel.rgb;
        float d = length(vWorldPos - ePos);
        if (d < minDist) {
          minDist = d;
          closestAlpha = texel.a;
        }
      }

      float t = (1.0 - clamp(minDist / maxDist, 0.0, 1.0)) * closestAlpha;
      vec3 baseColor = heatmapColor(t);

      // Simple Lambert shading so the brain reads as 3D
      vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
      float diffuse = max(dot(normalize(vWorldNormal), lightDir), 0.0);
      gl_FragColor = vec4(baseColor * (0.35 + 0.65 * diffuse), 1.0);
    }
  `;

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

  const mat = new THREE.ShaderMaterial({
    uniforms: {
      electrodesTex: { value: electrodesTex },
      maxDist:       { value: DIST_MAX },
    },
    vertexShader,
    fragmentShader,
    side: THREE.DoubleSide,   // render back faces too; without this, inner/inverted-normal faces show original material
  });

  gltfScene.traverse(child => {
    if (child.isMesh) child.material = mat;
  });
}

// --- GLTF ---
loader.load(
  'human_brain_PittBrains/scene.gltf',
  function (gltf) {
    gltfScene = gltf.scene;
    scene.add(gltfScene);
    const box = new THREE.Box3().setFromObject(gltfScene);
    const center = box.getCenter(new THREE.Vector3());
    controls.target.copy(center);
    cameraRot.position.set(center.x, center.y, center.z + 100);
    controls.update();
    tryApplyDistanceMaterial();
  },
  xhr => console.log(100 * (xhr.loaded / xhr.total) + '% loaded'),
  err => console.error(err)
);

// --- Electrodes ---
const scaleFactor = 0.0125; //Standard 3D Model of the MNI brain scaled to 0.01
const xOffset = 0;
const yOffset = -0.15;
const zOffset = 0.27;
fetch('elecs.json')
  .then(res => res.json())
  .then(coords => {
    const elecGeo = new THREE.SphereGeometry(scaleFactor * 0.5, scaleFactor * 0.5, scaleFactor * 0.5);
    const elecMat = new THREE.MeshBasicMaterial({ color: 0x00ffcc });
    const electrodes = new THREE.Group();

    //x, y, z in MNI coordinate space corresponds to -x, z, y in Cartesian
    // Visual only — does NOT set elecWorldPositions or drive the heatmap.
    // The bipolar fetch owns elecWorldPositions and calls tryApplyDistanceMaterial.
    coords.forEach(([x, z, y]) => {
      const pos = new THREE.Vector3(
        xOffset + -1 * x * scaleFactor,
        yOffset + y * scaleFactor,
        zOffset + z * scaleFactor
      );
      const dot = new THREE.Mesh(elecGeo, elecMat);
      dot.position.copy(pos);
      electrodes.add(dot);
    });

    elecGroup = electrodes;
    scene.add(electrodes);
  })
  .catch(err => console.error('standard electrode load failed', err));

  fetch('bipolar_electrodes.json')
    .then(res => res.json())
    .then(coords => {
      const elecGeo = new THREE.SphereGeometry(scaleFactor * 0.7, scaleFactor * 0.7, scaleFactor * 0.7);
      const BPelecMat = new THREE.MeshBasicMaterial({ color: 0xff0011 });
      const BPelectrodes = new THREE.Group();

      //Remember, xyz in MNI space corresponds to -x, z, y in Cartesian

      elecWorldPositions = coords.map(([x, z, y]) => {
        const pos = new THREE.Vector3(
          xOffset + -1 * x * scaleFactor, 
          yOffset + y * scaleFactor,
          zOffset + z * scaleFactor
        );

        const dot = new THREE.Mesh(elecGeo, BPelecMat);
        dot.position.copy(pos);
        BPelectrodes.add(dot);
        return pos;
      });
      
      BPelecGroup = BPelectrodes;
      scene.add(BPelectrodes);
      tryApplyDistanceMaterial();
    })
    .catch(err => console.error('bipolar electrode load failed', err));

// --- Electrode weights (from FastAPI) ---
// Fetches a JSON list of numbers (one per electrode, same order as elecs.json),
// stores them, and re-applies the brain material so the heatmap reflects them.
// Re-call this later (e.g. after the user retrains) to recolor without reloading.
async function loadWeights() {
  const res = await fetch(WEIGHTS_URL);
  if (!res.ok) throw new Error('weights HTTP ' + res.status);
  elecWeights = await res.json();   // e.g. [0.12, 0.87, ...]
  if (elecWorldPositions && elecWeights.length !== elecWorldPositions.length) {
    console.warn(
      `weight count (${elecWeights.length}) != electrode count ` +
      `(${elecWorldPositions.length}) — colors may be misaligned`
    );
  }
  tryApplyDistanceMaterial();       // rebuild the texture/material with new weights
}

// Fetch once on load. If the server isn't up yet, electrodes still render with
// the default weight of 1.0 — the warning just tells you weights weren't applied.
loadWeights().catch(err =>
  console.warn('weights not loaded (using default 1.0):', err)
);

// --- Renderer ---
const renderer = new THREE.WebGLRenderer({ alpha: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.getElementById("container3D").appendChild(renderer.domElement);
cameraRot.position.z = 50;

// --- Lights (still used by electrode spheres) ---
const topLight = new THREE.DirectionalLight(0xffffff, 3);
topLight.position.set(500, 500, 500);
scene.add(topLight);
scene.add(new THREE.AmbientLight(0x333333, 5));

// --- Controls ---
controls = new OrbitControls(cameraRot, renderer.domElement);

// --- Animate ---
function animate() {
  requestAnimationFrame(animate);
  renderer.render(scene, cameraRot);
}

window.addEventListener("resize", () => {
  cameraRot.aspect = window.innerWidth / window.innerHeight;
  cameraRot.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

animate();
