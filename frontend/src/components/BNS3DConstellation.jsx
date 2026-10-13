import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';

export default function BNS3DConstellation({ isProcessing = false }) {
  const mountRef = useRef(null);

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const width = container.clientWidth || 400;
    const height = container.clientHeight || 260;

    // Scene & Camera
    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x060713, 0.002);

    const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 1000);
    camera.position.z = 170;

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(renderer.domElement);

    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
    scene.add(ambientLight);

    const pointLight1 = new THREE.PointLight(0x6366f1, 2.5, 300);
    pointLight1.position.set(100, 100, 120);
    scene.add(pointLight1);

    const pointLight2 = new THREE.PointLight(0x06b6d4, 2.5, 300);
    pointLight2.position.set(-100, -80, 100);
    scene.add(pointLight2);

    // Master Group
    const masterGroup = new THREE.Group();
    scene.add(masterGroup);

    // 1. Central Codex Globe (Wireframe Icosahedron)
    const globeGeo = new THREE.IcosahedronGeometry(44, 2);
    const globeMat = new THREE.MeshStandardMaterial({
      color: 0x6366f1,
      wireframe: true,
      transparent: true,
      opacity: 0.35,
      roughness: 0.2,
    });
    const globeMesh = new THREE.Mesh(globeGeo, globeMat);
    masterGroup.add(globeMesh);

    // 2. Inner Glowing Core
    const coreGeo = new THREE.IcosahedronGeometry(22, 1);
    const coreMat = new THREE.MeshBasicMaterial({
      color: 0x06b6d4,
      wireframe: true,
      transparent: true,
      opacity: 0.6,
    });
    const coreMesh = new THREE.Mesh(coreGeo, coreMat);
    masterGroup.add(coreMesh);

    // 3. Orbiting Knowledge Nodes (Sections + Verification Agents)
    const nodeCount = 38;
    const nodes = [];
    const nodeGroup = new THREE.Group();
    masterGroup.add(nodeGroup);

    const nodeGeo = new THREE.SphereGeometry(2.2, 12, 12);
    const palette = [0x6366f1, 0x06b6d4, 0x10b981, 0xf43f5e, 0x8b5cf6, 0xf59e0b];

    for (let i = 0; i < nodeCount; i++) {
      const phi = Math.acos(-1 + (2 * i) / nodeCount);
      const theta = Math.sqrt(nodeCount * Math.PI) * phi;
      const r = 66 + (i % 4) * 9;

      const x = r * Math.cos(theta) * Math.sin(phi);
      const y = r * Math.sin(theta) * Math.sin(phi);
      const z = r * Math.cos(phi);

      const color = palette[i % palette.length];
      const mat = new THREE.MeshBasicMaterial({ color });
      const mesh = new THREE.Mesh(nodeGeo, mat);
      mesh.position.set(x, y, z);
      nodeGroup.add(mesh);
      nodes.push({ mesh, basePos: new THREE.Vector3(x, y, z), phase: i * 0.35 });
    }

    // 4. Neural Connective Lines
    const linePositions = [];
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const dist = nodes[i].basePos.distanceTo(nodes[j].basePos);
        if (dist < 40) {
          linePositions.push(
            nodes[i].basePos.x, nodes[i].basePos.y, nodes[i].basePos.z,
            nodes[j].basePos.x, nodes[j].basePos.y, nodes[j].basePos.z
          );
        }
      }
    }
    const lineGeo = new THREE.BufferGeometry();
    lineGeo.setAttribute('position', new THREE.Float32BufferAttribute(linePositions, 3));
    const lineMat = new THREE.LineBasicMaterial({
      color: 0x6366f1,
      transparent: true,
      opacity: 0.28,
    });
    const linesMesh = new THREE.LineSegments(lineGeo, lineMat);
    masterGroup.add(linesMesh);

    // 5. Star / Knowledge Dust Particles
    const particleCount = 180;
    const pPositions = new Float32Array(particleCount * 3);
    for (let i = 0; i < particleCount * 3; i += 3) {
      pPositions[i] = (Math.random() - 0.5) * 320;
      pPositions[i + 1] = (Math.random() - 0.5) * 220;
      pPositions[i + 2] = (Math.random() - 0.5) * 260;
    }
    const pGeo = new THREE.BufferGeometry();
    pGeo.setAttribute('position', new THREE.BufferAttribute(pPositions, 3));
    const pMat = new THREE.PointsMaterial({
      color: 0x94a3b8,
      size: 1.4,
      transparent: true,
      opacity: 0.45,
    });
    const particles = new THREE.Points(pGeo, pMat);
    scene.add(particles);

    // Mouse Drag & Parallax Handling
    let isDragging = false;
    let prevMouseX = 0;
    let prevMouseY = 0;
    let targetRotX = 0;
    let targetRotY = 0;

    const onMouseDown = (e) => {
      isDragging = true;
      prevMouseX = e.clientX;
      prevMouseY = e.clientY;
    };

    const onMouseUp = () => {
      isDragging = false;
    };

    const onMouseMove = (e) => {
      if (isDragging) {
        const dx = e.clientX - prevMouseX;
        const dy = e.clientY - prevMouseY;
        masterGroup.rotation.y += dx * 0.008;
        masterGroup.rotation.x += dy * 0.008;
        prevMouseX = e.clientX;
        prevMouseY = e.clientY;
      } else {
        const rect = container.getBoundingClientRect();
        const mx = ((e.clientX - rect.left) / container.clientWidth) * 2 - 1;
        const my = -((e.clientY - rect.top) / container.clientHeight) * 2 + 1;
        targetRotY = mx * 0.35;
        targetRotX = -my * 0.25;
      }
    };

    container.addEventListener('mousedown', onMouseDown);
    window.addEventListener('mouseup', onMouseUp);
    container.addEventListener('mousemove', onMouseMove);

    // Handle Resize
    const handleResize = () => {
      if (!container) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', handleResize);

    // Animation Loop
    let animId;
    const clock = new THREE.Clock();

    const animate = () => {
      animId = requestAnimationFrame(animate);
      const elapsed = clock.getElapsedTime();
      const speedMultiplier = isProcessing ? 2.5 : 1.0;

      if (!isDragging) {
        masterGroup.rotation.y += 0.004 * speedMultiplier;
        masterGroup.rotation.x += (targetRotX - masterGroup.rotation.x) * 0.04;
      }

      globeMesh.rotation.x += 0.002 * speedMultiplier;
      globeMesh.rotation.z += 0.001 * speedMultiplier;
      coreMesh.rotation.y -= 0.008 * speedMultiplier;
      particles.rotation.y += 0.0006 * speedMultiplier;

      // Pulsate nodes
      for (let i = 0; i < nodes.length; i++) {
        const s = 1 + 0.25 * Math.sin(elapsed * 3.5 + nodes[i].phase);
        nodes[i].mesh.scale.set(s, s, s);
      }

      renderer.render(scene, camera);
    };
    animate();

    return () => {
      cancelAnimationFrame(animId);
      container.removeEventListener('mousedown', onMouseDown);
      window.removeEventListener('mouseup', onMouseUp);
      container.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('resize', handleResize);
      if (renderer.domElement && container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, [isProcessing]);

  return (
    <div className="canvas-3d-wrapper" style={{ height: '280px', width: '100%', cursor: 'grab' }}>
      <div className="canvas-hud">
        <span className="hud-pill">✦ 3D BNS CODEX · 1,388 Chunks</span>
        <span className="hud-hint">Drag / Hover to rotate</span>
      </div>
      <div ref={mountRef} style={{ width: '100%', height: '100%' }} />
    </div>
  );
}
