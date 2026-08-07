// ============================================================
// CONFIG
// ============================================================
const WS_URL = `ws://${location.host}/ws/state`;

// Vicon -> Three.js: permutazione ciclica, nessun segno da invertire.
// Vicon X (verso l'osservatore) -> Three Z, Vicon Y (destra) -> Three X,
// Vicon Z (alto) -> Three Y.
function viconToThree(p) {
  return new THREE.Vector3(p[1], p[2], p[0]);
}

// Inversa di viconToThree: da un punto Three.js a coordinate Vicon [x,y,z].
function threeToVicon(v) {
  return [v.z, v.x, v.y];
}

// Yaw (gradi, attorno all'asse Vicon Z / Three.js Y) -> vettore direzione
// orizzontale in coordinate Three.js, stessa convenzione di viconToThree
// (yaw=0 punta lungo Vicon +x, cioe' Three +z).
function yawDegToThreeDir(yawDeg) {
  const yawRad = THREE.MathUtils.degToRad(yawDeg);
  return new THREE.Vector3(Math.sin(yawRad), 0, Math.cos(yawRad));
}

// Freccia 3D "solida" (cono + cilindro) per indicare lo yaw di drone/target.
// Non usiamo THREE.ArrowHelper: il suo stelo e' una linea sottile la cui
// larghezza i browser WebGL ignorano quasi sempre (fissata a 1px), e alla
// scala della stanza (pochi metri) risultava un puntino praticamente
// invisibile. depthTest:false + renderOrder alto la tengono sempre visibile
// sopra sfera/wireframe, anche quando punta verso la camera.
const ARROW_FORWARD = new THREE.Vector3(0, 0, 1);
function makeYawArrow(color, shaftLen = 0.5, shaftRadius = 0.025, headLen = 0.2, headRadius = 0.08) {
  const mat = new THREE.MeshBasicMaterial({ color, depthTest: false });
  const shaftGeo = new THREE.CylinderGeometry(shaftRadius, shaftRadius, shaftLen, 10);
  shaftGeo.rotateX(Math.PI / 2);
  shaftGeo.translate(0, 0, shaftLen / 2);
  const headGeo = new THREE.ConeGeometry(headRadius, headLen, 10);
  headGeo.rotateX(Math.PI / 2);
  headGeo.translate(0, 0, shaftLen + headLen / 2);
  const group = new THREE.Group();
  group.add(new THREE.Mesh(shaftGeo, mat), new THREE.Mesh(headGeo, mat));
  group.renderOrder = 999;
  return group;
}
function setYawArrowDirection(arrowGroup, yawDeg) {
  arrowGroup.quaternion.setFromUnitVectors(ARROW_FORWARD, yawDegToThreeDir(yawDeg));
}

// ============================================================
// SCENA THREE.JS
// ============================================================
let scene, camera, renderer, controls;
let droneMesh, targetMesh, arucoMesh, roomWireframe;
let droneArrow, targetArrow;
let roomBoundsSet = false;
let latestRoomMin = null, latestRoomMax = null;

// -- piazzamento target custom: raycast contro un piano orizzontale
// all'altezza scelta dallo slider (coordinate Vicon), attivo solo quando
// target_mode=='custom' e' selezionato nel form. --
const raycaster = new THREE.Raycaster();
const placementPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
let pointerDownPos = null;

function initScene() {
  const container = document.getElementById('three-container');

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0d0f12);

  camera = new THREE.PerspectiveCamera(
    60, container.clientWidth / container.clientHeight, 0.05, 100
  );
  camera.position.set(4, 3.5, 4);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(container.clientWidth, container.clientHeight);
  container.appendChild(renderer.domElement);

  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0, 0);

  scene.add(new THREE.AxesHelper(1.5));
  scene.add(new THREE.GridHelper(6, 12, 0x444444, 0x222222));

  const droneGeo = new THREE.SphereGeometry(0.08, 16, 16);
  const droneMat = new THREE.MeshBasicMaterial({ color: 0x4da6ff });
  droneMesh = new THREE.Mesh(droneGeo, droneMat);
  scene.add(droneMesh);

  // freccia orientamento drone (yaw), figlia di droneMesh cosi' segue la
  // sua posizione automaticamente: va solo ruotata via setYawArrowDirection().
  // Bianca (non blu come la sfera): stesso colore la renderebbe quasi
  // invisibile per il basso contrasto contro la sfera stessa.
  droneArrow = makeYawArrow(0xffffff);
  droneMesh.add(droneArrow);

  const targetGeo = new THREE.SphereGeometry(0.06, 16, 16);
  const targetMat = new THREE.MeshBasicMaterial({ color: 0xff5555 });
  targetMesh = new THREE.Mesh(targetGeo, targetMat);
  targetMesh.visible = false;
  scene.add(targetMesh);

  // freccia orientamento target (yaw del waypoint attivo), figlia di
  // targetMesh: si nasconde/mostra insieme al target.
  targetArrow = makeYawArrow(0xffffff);
  targetMesh.add(targetArrow);

  const arucoGeo = new THREE.BoxGeometry(0.09, 0.09, 0.09);
  const arucoMat = new THREE.MeshBasicMaterial({ color: 0xbb66ff });
  arucoMesh = new THREE.Mesh(arucoGeo, arucoMat);
  arucoMesh.visible = false;
  scene.add(arucoMesh);

  window.addEventListener('resize', () => {
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
  });

  renderer.domElement.addEventListener('pointerdown', (ev) => {
    pointerDownPos = { x: ev.clientX, y: ev.clientY };
  });
  renderer.domElement.addEventListener('pointerup', (ev) => {
    if (!pointerDownPos) return;
    const dx = ev.clientX - pointerDownPos.x;
    const dy = ev.clientY - pointerDownPos.y;
    pointerDownPos = null;
    // solo un vero CLICK (non un drag di OrbitControls) piazza il target
    if (Math.hypot(dx, dy) > 4) return;
    handleSceneClick(ev);
  });

  animate();
}

// ============================================================
// PIAZZAMENTO TARGET CUSTOM (click sulla scena 3D)
// ============================================================
function handleSceneClick(ev) {
  const modeSelect = document.getElementById('select-target-mode');
  if (!modeSelect || modeSelect.value !== 'custom') return;
  if (!latestRoomMin || !latestRoomMax) return;

  const zInput = document.getElementById('custom-target-z');
  const zVicon = parseFloat(zInput.value);
  placementPlane.constant = -zVicon; // piano y = zVicon in coordinate Three (Vicon Z -> Three Y)

  const rect = renderer.domElement.getBoundingClientRect();
  const mouse = new THREE.Vector2(
    ((ev.clientX - rect.left) / rect.width) * 2 - 1,
    -((ev.clientY - rect.top) / rect.height) * 2 + 1,
  );
  raycaster.setFromCamera(mouse, camera);
  const hit = new THREE.Vector3();
  if (!raycaster.ray.intersectPlane(placementPlane, hit)) return;

  const [x, y, z] = threeToVicon(hit);
  if (x < latestRoomMin[0] || x > latestRoomMax[0] ||
      y < latestRoomMin[1] || y > latestRoomMax[1] ||
      z < latestRoomMin[2] || z > latestRoomMax[2]) {
    document.getElementById('custom-target-msg').textContent =
      `Outside room bounds: (${x.toFixed(2)}, ${y.toFixed(2)}, ${z.toFixed(2)})`;
    return;
  }

  fetch('/api/custom_target', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ x, y, z }),
  })
    .then((res) => res.json())
    .then((data) => {
      document.getElementById('custom-target-msg').textContent = data.message;
    });
}

function setRoomBounds(roomMin, roomMax) {
  latestRoomMin = roomMin;
  latestRoomMax = roomMax;

  const zSlider = document.getElementById('custom-target-z');
  if (zSlider && zSlider.dataset.rangeSet !== '1') {
    zSlider.min = roomMin[2];
    zSlider.max = roomMax[2];
    zSlider.value = Math.min(Math.max(1.0, roomMin[2]), roomMax[2]);
    zSlider.dataset.rangeSet = '1';
    document.getElementById('custom-target-z-val').textContent = `${parseFloat(zSlider.value).toFixed(2)} m`;
  }

  if (roomBoundsSet) return;
  roomBoundsSet = true;

  const corners = [];
  for (const dx of [roomMin[0], roomMax[0]]) {
    for (const dy of [roomMin[1], roomMax[1]]) {
      for (const dz of [roomMin[2], roomMax[2]]) {
        corners.push(viconToThree([dx, dy, dz]));
      }
    }
  }
  const edges = [
    [0,1],[2,3],[4,5],[6,7],
    [0,2],[1,3],[4,6],[5,7],
    [0,4],[1,5],[2,6],[3,7],
  ];
  const geo = new THREE.BufferGeometry();
  const positions = [];
  for (const [a, b] of edges) {
    positions.push(corners[a].x, corners[a].y, corners[a].z);
    positions.push(corners[b].x, corners[b].y, corners[b].z);
  }
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  const mat = new THREE.LineBasicMaterial({ color: 0x3a6ea5 });
  roomWireframe = new THREE.LineSegments(geo, mat);
  scene.add(roomWireframe);
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

// ============================================================
// PLOT CON GRID + ASSE Y NUMERATO E UNITA' DI MISURA
// ============================================================
// ============================================================
// PLOT CON GRID + ASSE Y NUMERATO E UNITA' DI MISURA
// Canvas ad alta risoluzione: il buffer interno viene sincronizzato
// con la dimensione REALE (CSS) del canvas moltiplicata per
// devicePixelRatio, altrimenti il browser stira un'immagine a bassa
// risoluzione producendo linee sfocate/pixelate (bug visibile con
// schermi a scaling >100% o quando il canvas viene ridimensionato via
// flexbox/CSS invece che con attributi width/height fissi).
// ============================================================
// ============================================================
// PLOT CON GRID + ASSE Y NUMERATO
// (unita' di misura NON piu' ripetuta dentro il canvas: e' gia'
// presente nel titolo <h3> sopra ciascun grafico, ripeterla causava
// sovrapposizione con legenda e primo valore dell'asse Y)
// ============================================================
class TimeSeriesPlot {
  constructor(canvasId, labels, colors, windowSize = 150) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.labels = labels;
    this.colors = colors;
    this.windowSize = windowSize;
    this.series = labels.map(() => []);
    this.marginLeft = 40;
    this.marginTop = 24;    // <-- aumentato: spazio per la riga legenda, separata dalla prima griglia
    this.marginBottom = 4;
    this.marginRight = 4;

    this.dpr = window.devicePixelRatio || 1;
    this._resizeToContainer();

    this._resizeObserver = new ResizeObserver(() => {
      this._resizeToContainer();
      this.draw();
    });
    this._resizeObserver.observe(this.canvas);
  }

  _resizeToContainer() {
    const rect = this.canvas.getBoundingClientRect();
    this.cssWidth = Math.max(1, rect.width);
    this.cssHeight = Math.max(1, rect.height);
    this.canvas.width = Math.round(this.cssWidth * this.dpr);
    this.canvas.height = Math.round(this.cssHeight * this.dpr);
    this.ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
  }

  push(values) {
    values.forEach((v, i) => {
      this.series[i].push(v);
      if (this.series[i].length > this.windowSize) this.series[i].shift();
    });
    this.draw();
  }

  draw() {
    const { ctx } = this;
    const w = this.cssWidth, h = this.cssHeight;
    ctx.clearRect(0, 0, w, h);

    const plotW = w - this.marginLeft - this.marginRight;
    const plotH = h - this.marginTop - this.marginBottom;

    let allVals = this.series.flat();
    if (allVals.length === 0) allVals = [0];
    let vmin = Math.min(...allVals, -0.1);
    let vmax = Math.max(...allVals, 0.1);
    if (vmax - vmin < 1e-3) { vmax += 0.5; vmin -= 0.5; }
    const pad = (vmax - vmin) * 0.12;
    vmin -= pad;
    vmax += pad;

    const yFor = (v) => this.marginTop + plotH - ((v - vmin) / (vmax - vmin)) * plotH;
    const xFor = (idx) => this.marginLeft + (idx / (this.windowSize - 1)) * plotW;

    // -- riga legenda, in cima, ben separata dalla griglia sottostante --
    ctx.font = '10px sans-serif';
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'left';
    this.labels.forEach((label, i) => {
      ctx.fillStyle = this.colors[i];
      ctx.fillText(label, this.marginLeft + i * 34, 10);
    });

    // -- griglia orizzontale + etichette asse Y (parte SOTTO la legenda) --
    const nGrid = 4;
    ctx.lineWidth = 1;
    for (let i = 0; i <= nGrid; i++) {
      const v = vmin + (vmax - vmin) * (i / nGrid);
      const y = Math.round(yFor(v)) + 0.5;
      ctx.strokeStyle = '#22262e';
      ctx.beginPath();
      ctx.moveTo(this.marginLeft, y);
      ctx.lineTo(w, y);
      ctx.stroke();
      ctx.fillStyle = '#888';
      ctx.textAlign = 'right';
      ctx.fillText(v.toFixed(2), this.marginLeft - 4, y);
    }

    if (vmin < 0 && vmax > 0) {
      ctx.strokeStyle = '#4a4f5a';
      ctx.beginPath();
      const y0 = Math.round(yFor(0)) + 0.5;
      ctx.moveTo(this.marginLeft, y0);
      ctx.lineTo(w, y0);
      ctx.stroke();
    }

    // -- serie --
    this.series.forEach((serie, i) => {
      if (serie.length < 2) return;
      ctx.strokeStyle = this.colors[i];
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      serie.forEach((v, idx) => {
        const x = xFor(idx);
        const y = yFor(v);
        idx === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.stroke();
    });
  }
}

const plotLinVel = new TimeSeriesPlot('plot-linvel', ['vx', 'vy', 'vz'], ['#4da6ff', '#5cd65c', '#ff9d4d']);
const plotAngVel = new TimeSeriesPlot('plot-angvel', ['wx', 'wy', 'wz'], ['#4da6ff', '#5cd65c', '#ff9d4d']);
const plotRPY = new TimeSeriesPlot('plot-rpy', ['roll', 'pitch', 'yaw'], ['#4da6ff', '#5cd65c', '#ff9d4d']);

// ============================================================
// AGGIORNAMENTO UI DA STATO RICEVUTO
// ============================================================
function setIndicator(id, ok, textOn, textOff) {
  const el = document.getElementById(id);
  el.className = 'indicator ' + (ok ? 'on' : 'off');
  el.textContent = ok ? textOn : textOff;
}

function updateUI(state) {
  setIndicator('ind-vicon', state.vicon_connected, 'VICON OK', 'VICON --');
  setIndicator('ind-tello', state.tello_connected, 'TELLO OK', 'TELLO --');
  setIndicator('ind-battery', state.battery !== null && state.battery > 20,
    `BATT ${state.battery}%`, `BATT ${state.battery ?? '--'}%`);
  setIndicator('ind-session', state.session_state === 'flying', state.session_state.toUpperCase(), state.session_state.toUpperCase());

  document.getElementById('s-session').textContent = state.session_state;
  document.getElementById('s-queue').textContent =
    `${state.queues_completed} completed (limit: ${state.num_queues > 0 ? state.num_queues : 'infinite'})`;
  document.getElementById('s-wp').textContent = `${state.wp_idx + 1}/${state.n_waypoints}`;
  document.getElementById('s-pos').textContent = state.pos.map(v => v.toFixed(2)).join(', ');
  document.getElementById('s-rpy').textContent =
    `${state.roll_deg.toFixed(1)} / ${state.pitch_deg.toFixed(1)} / ${state.yaw_deg.toFixed(1)}`;

  setRoomBounds(state.room_min, state.room_max);

  const dronePos = viconToThree(state.pos);
  droneMesh.position.copy(dronePos);
  setYawArrowDirection(droneArrow, state.yaw_deg);

  if (state.target) {
    targetMesh.visible = true;
    targetMesh.position.copy(viconToThree(state.target));
    if (state.target_yaw_deg !== null && state.target_yaw_deg !== undefined) {
      setYawArrowDirection(targetArrow, state.target_yaw_deg);
    }
  } else {
    targetMesh.visible = false;
  }

  if (state.aruco_pos) {
    arucoMesh.visible = true;
    arucoMesh.position.copy(viconToThree(state.aruco_pos));
    document.getElementById('s-aruco').textContent = state.aruco_pos.map(v => v.toFixed(2)).join(', ');
  } else {
    arucoMesh.visible = false;
    document.getElementById('s-aruco').textContent = '--';
  }

  document.getElementById('s-custom-target').textContent =
    state.custom_target ? state.custom_target.map(v => v.toFixed(2)).join(', ') : '--';

  const customPanel = document.getElementById('custom-target-panel');
  customPanel.hidden = state.target_mode !== 'custom';
  document.getElementById('three-container').classList.toggle('placement-mode', state.target_mode === 'custom');

  plotLinVel.push(state.lin_vel_b);
  plotAngVel.push(state.ang_vel_b);
  plotRPY.push([state.roll_deg, state.pitch_deg, state.yaw_deg]);

  const algoBusy = state.session_state !== 'idle';
  document.getElementById('btn-start').disabled = algoBusy;
}

// ============================================================
// WEBSOCKET
// ============================================================
function connectWebSocket() {
  const ws = new WebSocket(WS_URL);
  ws.onmessage = (ev) => updateUI(JSON.parse(ev.data));
  ws.onclose = () => setTimeout(connectWebSocket, 1000);
  ws.onerror = () => ws.close();
}

// ============================================================
// FORM E BOTTONI
// ============================================================
document.getElementById('params-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const payload = {
    dof_mask_mode: fd.get('dof_mask_mode'),
    target_mode: fd.get('target_mode'),
    advance_mode: fd.get('advance_mode'),
    num_queues: parseInt(fd.get('num_queues'), 10),
  };
  const res = await fetch('/api/params', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  document.getElementById('params-msg').textContent = data.message;
});

document.getElementById('select-target-mode').addEventListener('change', (e) => {
  document.getElementById('custom-target-panel').hidden = e.target.value !== 'custom';
});

document.getElementById('custom-target-z').addEventListener('input', (e) => {
  document.getElementById('custom-target-z-val').textContent = `${parseFloat(e.target.value).toFixed(2)} m`;
});

document.getElementById('btn-start').addEventListener('click', () => fetch('/api/start', { method: 'POST' }));
document.getElementById('btn-land').addEventListener('click', () => fetch('/api/land', { method: 'POST' }));
document.getElementById('btn-advance').addEventListener('click', () => fetch('/api/advance', { method: 'POST' }));

// ============================================================
// AVVIO
// ============================================================
initScene();
connectWebSocket();