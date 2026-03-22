/* ═══════════════════════════════════════════════════════════
   FraudSentinel — Three.js Transaction Simulation
   Cinematic 3D network of nodes with animated transaction beams
   ═══════════════════════════════════════════════════════════ */

(function () {
  'use strict';

  const API_BASE = 'http://localhost:5000';

  // ─── CONFIG ───────────────────────────────────────────────────────────────
  const CONFIG = {
    NODE_COUNT:        18,
    NODE_RADIUS_RANGE: [0.18, 0.38],
    ORBIT_RADIUS:      8.5,
    BEAM_SPEED:        0.018,
    BEAM_DURATION_MS:  1800,
    INTERVAL_MS:       2200,
    PARTICLE_COUNT:    120,
    STAR_COUNT:        600,
    EXPLOSION_PARTS:   24
  };

  // ─── SCENE STATE ─────────────────────────────────────────────────────────
  let scene, camera, renderer, clock;
  let nodes          = [];
  let activeBeams    = [];
  let explosions     = [];
  let starField;
  let gridMesh;
  let simRunning     = true;
  let simTimer       = null;
  let txnCount       = 0;
  let fraudCount     = 0;
  let legitCount     = 0;
  let animFrameId    = null;

  // DOM refs
  let canvas, wrap;

  // ─── INIT ─────────────────────────────────────────────────────────────────
  function init() {
    canvas = document.getElementById('simCanvas');
    wrap   = document.getElementById('simCanvasWrap');
    if (!canvas || !wrap) return;

    const W = wrap.clientWidth;
    const H = wrap.clientHeight;

    // Renderer
    renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha:     true
    });
    renderer.setSize(W, H);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x050810, 1);

    // Scene
    scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x050810, 0.025);

    // Camera
    camera = new THREE.PerspectiveCamera(60, W / H, 0.1, 200);
    camera.position.set(0, 4, 18);
    camera.lookAt(0, 0, 0);

    // Clock
    clock = new THREE.Clock();

    // Lighting
    setupLights();

    // Background
    buildStarField();
    buildGrid();

    // Network nodes
    buildNodes();

    // Resize handler
    window.addEventListener('resize', onResize);

    // Controls
    document.getElementById('simToggleBtn').addEventListener('click', toggleSim);
    document.getElementById('simResetBtn').addEventListener('click', resetSim);

    // Start loop
    animate();
    scheduleTransaction();
  }

  // ─── LIGHTING ─────────────────────────────────────────────────────────────
  function setupLights() {
    const ambient = new THREE.AmbientLight(0x0a1428, 1.0);
    scene.add(ambient);

    const blue = new THREE.PointLight(0x00d4ff, 1.5, 30);
    blue.position.set(-8, 6, 4);
    scene.add(blue);

    const red = new THREE.PointLight(0xff4b5c, 0.8, 20);
    red.position.set(8, -4, -2);
    scene.add(red);

    const center = new THREE.PointLight(0x4466ff, 0.6, 25);
    center.position.set(0, 0, 0);
    scene.add(center);
  }

  // ─── STAR FIELD ───────────────────────────────────────────────────────────
  function buildStarField() {
    const geo  = new THREE.BufferGeometry();
    const pos  = new Float32Array(CONFIG.STAR_COUNT * 3);
    const cols = new Float32Array(CONFIG.STAR_COUNT * 3);

    for (let i = 0; i < CONFIG.STAR_COUNT; i++) {
      const r     = 30 + Math.random() * 70;
      const theta = Math.random() * Math.PI * 2;
      const phi   = Math.acos(2 * Math.random() - 1);

      pos[i * 3]     = r * Math.sin(phi) * Math.cos(theta);
      pos[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      pos[i * 3 + 2] = r * Math.cos(phi);

      // Mix blue and white stars
      const t = Math.random();
      cols[i * 3]     = t < 0.7 ? 0.6 : 0.0;
      cols[i * 3 + 1] = t < 0.7 ? 0.7 : 0.8;
      cols[i * 3 + 2] = 1.0;
    }

    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('color',    new THREE.BufferAttribute(cols, 3));

    const mat = new THREE.PointsMaterial({
      size:         0.12,
      vertexColors: true,
      transparent:  true,
      opacity:      0.7,
      blending:     THREE.AdditiveBlending,
      depthWrite:   false
    });

    starField = new THREE.Points(geo, mat);
    scene.add(starField);
  }

  // ─── GRID ─────────────────────────────────────────────────────────────────
  function buildGrid() {
    const geo  = new THREE.PlaneGeometry(40, 40, 20, 20);
    const mat  = new THREE.MeshBasicMaterial({
      color:       0x00d4ff,
      wireframe:   true,
      transparent: true,
      opacity:     0.04,
      depthWrite:  false
    });
    gridMesh = new THREE.Mesh(geo, mat);
    gridMesh.rotation.x = -Math.PI / 2;
    gridMesh.position.y = -5;
    scene.add(gridMesh);
  }

  // ─── NODES ────────────────────────────────────────────────────────────────
  function buildNodes() {
    for (let i = 0; i < CONFIG.NODE_COUNT; i++) {
      const node = createNode(i);
      nodes.push(node);
      scene.add(node.group);
    }

    // Connect some nodes with static dim lines
    for (let i = 0; i < nodes.length; i++) {
      const connections = Math.floor(1 + Math.random() * 2);
      for (let c = 0; c < connections; c++) {
        const j = Math.floor(Math.random() * nodes.length);
        if (j !== i) createStaticEdge(nodes[i], nodes[j]);
      }
    }
  }

  function createNode(index) {
    const group = new THREE.Group();

    // Distribute nodes in a spherical shell
    const phi   = Math.acos(-1 + (2 * index) / CONFIG.NODE_COUNT);
    const theta = Math.sqrt(CONFIG.NODE_COUNT * Math.PI) * phi;
    const r     = CONFIG.ORBIT_RADIUS * (0.7 + Math.random() * 0.5);

    const x = r * Math.sin(phi) * Math.cos(theta) + (Math.random() - 0.5) * 2;
    const y = r * Math.cos(phi) * 0.5              + (Math.random() - 0.5) * 1.5;
    const z = r * Math.sin(phi) * Math.sin(theta)  + (Math.random() - 0.5) * 2;

    group.position.set(x, y, z);

    const radius = CONFIG.NODE_RADIUS_RANGE[0] +
                   Math.random() * (CONFIG.NODE_RADIUS_RANGE[1] - CONFIG.NODE_RADIUS_RANGE[0]);

    // Core sphere
    const geo = new THREE.SphereGeometry(radius, 16, 16);
    const mat = new THREE.MeshStandardMaterial({
      color:     0x00aaff,
      emissive:  0x002244,
      metalness: 0.4,
      roughness: 0.3
    });
    const mesh = new THREE.Mesh(geo, mat);
    group.add(mesh);

    // Outer glow ring
    const ringGeo = new THREE.RingGeometry(radius * 1.4, radius * 1.8, 24);
    const ringMat = new THREE.MeshBasicMaterial({
      color:       0x00d4ff,
      transparent: true,
      opacity:     0.15,
      side:        THREE.DoubleSide,
      blending:    THREE.AdditiveBlending,
      depthWrite:  false
    });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    group.add(ring);

    // Halo glow sphere
    const haloGeo = new THREE.SphereGeometry(radius * 2.2, 8, 8);
    const haloMat = new THREE.MeshBasicMaterial({
      color:       0x0066ff,
      transparent: true,
      opacity:     0.04,
      blending:    THREE.AdditiveBlending,
      depthWrite:  false,
      side:        THREE.BackSide
    });
    const halo = new THREE.Mesh(haloGeo, haloMat);
    group.add(halo);

    return {
      group,
      mesh,
      ring,
      halo,
      radius,
      basePos: new THREE.Vector3(x, y, z),
      phase:   Math.random() * Math.PI * 2,
      speed:   0.3 + Math.random() * 0.5,
      active:  false
    };
  }

  function createStaticEdge(nodeA, nodeB) {
    const points = [nodeA.group.position, nodeB.group.position];
    const geo    = new THREE.BufferGeometry().setFromPoints(points);
    const mat    = new THREE.LineBasicMaterial({
      color:       0x1a3050,
      transparent: true,
      opacity:     0.3,
      blending:    THREE.AdditiveBlending,
      depthWrite:  false
    });
    const line = new THREE.Line(geo, mat);
    scene.add(line);
  }

  // ─── BEAM ─────────────────────────────────────────────────────────────────
  function createBeam(fromNode, toNode, isFraud) {
    const color     = isFraud ? 0xff2233 : 0x00d4ff;
    const emissive  = isFraud ? 0xff0011 : 0x003344;

    const start = fromNode.group.position.clone();
    const end   = toNode.group.position.clone();

    // Create curve for the beam
    const mid = new THREE.Vector3()
      .addVectors(start, end)
      .multiplyScalar(0.5);
    mid.y += 1.5 + Math.random() * 2;

    const curve = new THREE.QuadraticBezierCurve3(start, mid, end);
    const points = curve.getPoints(60);

    // Main beam line
    const lineGeo = new THREE.BufferGeometry().setFromPoints(points);
    const lineMat = new THREE.LineBasicMaterial({
      color:       color,
      transparent: true,
      opacity:     0.8,
      blending:    THREE.AdditiveBlending,
      depthWrite:  false
    });
    const line = new THREE.Line(lineGeo, lineMat);
    scene.add(line);

    // Traveling orb
    const orbGeo = new THREE.SphereGeometry(isFraud ? 0.14 : 0.10, 8, 8);
    const orbMat = new THREE.MeshBasicMaterial({
      color:    color,
      transparent: true,
      opacity:  0.95,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });
    const orb = new THREE.Mesh(orbGeo, orbMat);
    orb.position.copy(start);
    scene.add(orb);

    // Orb glow
    const glowGeo = new THREE.SphereGeometry(isFraud ? 0.35 : 0.25, 8, 8);
    const glowMat = new THREE.MeshBasicMaterial({
      color:    color,
      transparent: true,
      opacity:  0.12,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      side: THREE.BackSide
    });
    const glow = new THREE.Mesh(glowGeo, glowMat);
    orb.add(glow);

    // Trail particles
    const trailGeo  = new THREE.BufferGeometry();
    const trailPos  = new Float32Array(30 * 3);
    trailGeo.setAttribute('position', new THREE.BufferAttribute(trailPos, 3));
    const trailMat = new THREE.PointsMaterial({
      color:       color,
      size:        0.06,
      transparent: true,
      opacity:     0.5,
      blending:    THREE.AdditiveBlending,
      depthWrite:  false
    });
    const trail = new THREE.Points(trailGeo, trailMat);
    scene.add(trail);

    const beam = {
      line, orb, trail,
      curve, points,
      progress: 0,
      isFraud,
      fromNode,
      toNode,
      trailHistory: [],
      done: false
    };

    activeBeams.push(beam);

    // Highlight source node
    setNodeActive(fromNode, true, isFraud);

    return beam;
  }

  function setNodeActive(node, active, isFraud = false) {
    const color = isFraud ? 0xff2233 : 0x00d4ff;
    const emiss = isFraud ? 0x550011 : 0x001133;

    node.mesh.material.color.setHex(active ? color : 0x00aaff);
    node.mesh.material.emissive.setHex(active ? emiss : 0x002244);
    node.ring.material.opacity = active ? 0.5 : 0.15;
    node.ring.material.color.setHex(active ? color : 0x00d4ff);
    node.active = active;
  }

  function removeBeam(beam) {
    scene.remove(beam.line);
    scene.remove(beam.orb);
    scene.remove(beam.trail);
    beam.line.geometry.dispose();
    beam.orb.geometry.dispose();
    beam.trail.geometry.dispose();
  }

  // ─── EXPLOSION ────────────────────────────────────────────────────────────
  function createExplosion(position, isFraud) {
    const color    = isFraud ? 0xff2233 : 0x00e896;
    const count    = CONFIG.EXPLOSION_PARTS;
    const particles = [];

    for (let i = 0; i < count; i++) {
      const geo = new THREE.SphereGeometry(0.04 + Math.random() * 0.06, 4, 4);
      const mat = new THREE.MeshBasicMaterial({
        color:    color,
        transparent: true,
        opacity:  0.9,
        blending: THREE.AdditiveBlending,
        depthWrite: false
      });
      const p = new THREE.Mesh(geo, mat);
      p.position.copy(position);

      const speed = 0.04 + Math.random() * 0.08;
      const theta = Math.random() * Math.PI * 2;
      const phi   = Math.random() * Math.PI;

      particles.push({
        mesh: p,
        vel: new THREE.Vector3(
          speed * Math.sin(phi) * Math.cos(theta),
          speed * Math.sin(phi) * Math.sin(theta),
          speed * Math.cos(phi)
        ),
        life: 1.0,
        decay: 0.025 + Math.random() * 0.03
      });
      scene.add(p);
    }

    // Center flash
    const flashGeo = new THREE.SphereGeometry(0.4, 8, 8);
    const flashMat = new THREE.MeshBasicMaterial({
      color:    color,
      transparent: true,
      opacity:  0.7,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });
    const flash = new THREE.Mesh(flashGeo, flashMat);
    flash.position.copy(position);
    scene.add(flash);

    explosions.push({
      particles,
      flash,
      flashLife: 1.0,
      isFraud,
      done: false
    });
  }

  // ─── UPDATE EXPLOSIONS ────────────────────────────────────────────────────
  function updateExplosions() {
    for (let i = explosions.length - 1; i >= 0; i--) {
      const expl = explosions[i];
      expl.flashLife -= 0.06;
      expl.flash.material.opacity = Math.max(0, expl.flashLife * 0.7);
      expl.flash.scale.setScalar(1 + (1 - expl.flashLife) * 3);

      let allDead = true;
      expl.particles.forEach(p => {
        if (p.life > 0) {
          allDead = false;
          p.life  -= p.decay;
          p.mesh.position.add(p.vel);
          p.vel.multiplyScalar(0.92);
          p.mesh.material.opacity = Math.max(0, p.life);
        } else {
          p.mesh.visible = false;
        }
      });

      if (allDead && expl.flashLife <= 0) {
        expl.particles.forEach(p => {
          scene.remove(p.mesh);
          p.mesh.geometry.dispose();
        });
        scene.remove(expl.flash);
        expl.flash.geometry.dispose();
        expl.done = true;
        explosions.splice(i, 1);
      }
    }
  }

  // ─── SIMULATE TRANSACTION ─────────────────────────────────────────────────
  async function fireTransaction() {
    if (!simRunning || nodes.length < 2) return;

    // Pick two distinct random nodes
    const fromIdx = Math.floor(Math.random() * nodes.length);
    let toIdx     = Math.floor(Math.random() * nodes.length);
    while (toIdx === fromIdx) toIdx = Math.floor(Math.random() * nodes.length);

    let isFraud   = false;
    let txnData   = null;

    try {
      const res = await fetch(API_BASE + '/api/simulate_transaction');
      if (res.ok) {
        txnData = await res.json();
        isFraud = txnData.verdict === 'FRAUD';
      }
    } catch {
      // Fallback: random 12% fraud rate
      isFraud = Math.random() < 0.12;
      txnData = {
        transaction_id: `TXN-${String(++txnCount).padStart(6,'0')}`,
        amount: (10 + Math.random() * 1000).toFixed(2),
        verdict: isFraud ? 'FRAUD' : 'LEGITIMATE',
        flagged_by: isFraud ? ['Neural Network'] : [],
        predictions: {
          'Logistic Regression': { probability: isFraud ? 0.82 : 0.03 },
          'SVM':                 { probability: isFraud ? 0.78 : 0.02 },
          'Decision Tree':       { probability: isFraud ? 0.91 : 0.05 },
          'Neural Network':      { probability: isFraud ? 0.95 : 0.01 }
        }
      };
    }

    createBeam(nodes[fromIdx], nodes[toIdx], isFraud);
    updateFeedPanel(txnData, isFraud);
    updateCounters(isFraud);
  }

  // ─── FEED PANEL ───────────────────────────────────────────────────────────
  let feedTxns = [];

  function updateFeedPanel(data, isFraud) {
    const rows    = document.getElementById('feedRows');
    const counter = document.getElementById('feedCount');
    if (!rows) return;

    const txnId  = data ? data.transaction_id : `TXN-${String(++txnCount).padStart(6,'0')}`;
    const amount = data ? `$${parseFloat(data.amount).toFixed(2)}` : '$???';
    const flaggedBy = (data && data.flagged_by && data.flagged_by.length > 0)
      ? data.flagged_by.join(', ')
      : (isFraud ? 'Neural Net' : '—');

    feedTxns.unshift({ txnId, amount, flaggedBy, isFraud });
    if (feedTxns.length > 12) feedTxns.pop();

    const empty = rows.querySelector('.feed-empty');
    if (empty) empty.remove();

    // Prepend new row
    const row = document.createElement('div');
    row.className = 'feed-row' + (isFraud ? ' fraud' : '');
    row.innerHTML = `
      <div class="feed-id">${txnId}</div>
      <div class="feed-amount">${amount}</div>
      <div class="feed-models" title="${flaggedBy}">${flaggedBy}</div>
      <div class="feed-status ${isFraud ? 'fraud' : 'legit'}">${isFraud ? '🚨 FRAUD' : '✅ OK'}</div>
    `;

    rows.insertBefore(row, rows.firstChild);

    // Trim excess rows
    while (rows.children.length > 12) {
      rows.removeChild(rows.lastChild);
    }

    // Update count
    if (counter) counter.textContent = `${feedTxns.length} transactions`;
  }

  function updateCounters(isFraud) {
    if (isFraud) fraudCount++;
    else         legitCount++;

    const lEl = document.querySelector('#simLegitCount .sim-stat-val');
    const fEl = document.querySelector('#simFraudCount .sim-stat-val');
    if (lEl) lEl.textContent = legitCount;
    if (fEl) fEl.textContent = fraudCount;
  }

  // ─── CONTROLS ─────────────────────────────────────────────────────────────
  function toggleSim() {
    simRunning = !simRunning;
    const btn = document.getElementById('simToggleBtn');
    if (simRunning) {
      btn.textContent = '⏸ Pause';
      scheduleTransaction();
    } else {
      btn.textContent = '▶ Resume';
      if (simTimer) { clearTimeout(simTimer); simTimer = null; }
    }
  }

  function resetSim() {
    // Clear beams
    activeBeams.forEach(b => removeBeam(b));
    activeBeams = [];

    // Clear explosions
    explosions.forEach(expl => {
      expl.particles.forEach(p => { scene.remove(p.mesh); p.mesh.geometry.dispose(); });
      scene.remove(expl.flash);
    });
    explosions = [];

    // Reset nodes
    nodes.forEach(n => setNodeActive(n, false, false));

    // Reset counters
    fraudCount = 0;
    legitCount = 0;
    feedTxns   = [];

    const lEl = document.querySelector('#simLegitCount .sim-stat-val');
    const fEl = document.querySelector('#simFraudCount .sim-stat-val');
    if (lEl) lEl.textContent = '0';
    if (fEl) fEl.textContent = '0';

    const rows = document.getElementById('feedRows');
    if (rows) rows.innerHTML = '<div class="feed-empty">Waiting for transactions...</div>';

    const counter = document.getElementById('feedCount');
    if (counter) counter.textContent = '0 transactions';
  }

  function scheduleTransaction() {
    if (!simRunning) return;
    const jitter = (Math.random() - 0.5) * 800;
    simTimer = setTimeout(() => {
      fireTransaction();
      scheduleTransaction();
    }, CONFIG.INTERVAL_MS + jitter);
  }

  // ─── ANIMATION LOOP ───────────────────────────────────────────────────────
  function animate() {
    animFrameId = requestAnimationFrame(animate);
    const delta = clock.getDelta();
    const time  = clock.getElapsedTime();

    // Rotate starfield slowly
    if (starField) {
      starField.rotation.y = time * 0.008;
      starField.rotation.x = time * 0.003;
    }

    // Gentle camera orbit
    camera.position.x = Math.sin(time * 0.05) * 18;
    camera.position.z = Math.cos(time * 0.05) * 18;
    camera.position.y = 4 + Math.sin(time * 0.08) * 2;
    camera.lookAt(0, 0, 0);

    // Animate nodes
    nodes.forEach(node => {
      const pulse = 1 + Math.sin(time * node.speed + node.phase) * 0.08;
      node.group.scale.setScalar(pulse);

      // Rotate ring
      node.ring.rotation.z = time * node.speed * 0.5;
      node.ring.rotation.x = Math.sin(time * 0.3 + node.phase) * 0.4;
    });

    // Animate beams
    for (let i = activeBeams.length - 1; i >= 0; i--) {
      const beam = activeBeams[i];
      beam.progress = Math.min(beam.progress + CONFIG.BEAM_SPEED, 1);

      const pt = beam.curve.getPointAt(beam.progress);
      beam.orb.position.copy(pt);

      // Update trail
      beam.trailHistory.unshift(pt.clone());
      if (beam.trailHistory.length > 15) beam.trailHistory.pop();

      const trailPositions = beam.trail.geometry.attributes.position.array;
      for (let t = 0; t < 10; t++) {
        const src = beam.trailHistory[t] || pt;
        trailPositions[t * 3]     = src.x;
        trailPositions[t * 3 + 1] = src.y;
        trailPositions[t * 3 + 2] = src.z;
      }
      beam.trail.geometry.attributes.position.needsUpdate = true;

      // Pulse orb while traveling
      const pulseFactor = 1 + Math.sin(time * 20) * (beam.isFraud ? 0.3 : 0.15);
      beam.orb.scale.setScalar(pulseFactor);

      // Fade in beam line
      const fadeProgress = Math.min(beam.progress * 4, 1);
      beam.line.material.opacity = 0.5 * fadeProgress;

      // Arrival
      if (beam.progress >= 1 && !beam.done) {
        beam.done = true;
        createExplosion(beam.toNode.group.position.clone(), beam.isFraud);
        setNodeActive(beam.toNode, true, beam.isFraud);

        // Deactivate after a moment
        setTimeout(() => {
          setNodeActive(beam.fromNode, false, false);
          setNodeActive(beam.toNode, false, false);
        }, 800);

        // Remove beam after fade
        setTimeout(() => {
          removeBeam(beam);
          const idx = activeBeams.indexOf(beam);
          if (idx > -1) activeBeams.splice(idx, 1);
        }, 400);
      }
    }

    // Update explosions
    updateExplosions();

    renderer.render(scene, camera);
  }

  // ─── RESIZE ───────────────────────────────────────────────────────────────
  function onResize() {
    if (!wrap || !renderer || !camera) return;
    const W = wrap.clientWidth;
    const H = wrap.clientHeight;
    camera.aspect = W / H;
    camera.updateProjectionMatrix();
    renderer.setSize(W, H);
  }

  // ─── BOOT ─────────────────────────────────────────────────────────────────
  window.addEventListener('DOMContentLoaded', () => {
    // Slight delay to let layout settle
    setTimeout(init, 400);
  });

})();
