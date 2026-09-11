import { useEffect, useRef } from "react";
import * as THREE from "three";
import type { Stage } from "../types";

type Mood = "idle" | "working" | "auto" | "escalate";

const MOODS: Record<Mood, { a: THREE.Color; b: THREE.Color; speed: number; spread: number }> = {
  idle: { a: new THREE.Color("#6366f1"), b: new THREE.Color("#93c5fd"), speed: 0.12, spread: 1 },
  working: { a: new THREE.Color("#8b5cf6"), b: new THREE.Color("#d946ef"), speed: 0.55, spread: 1.5 },
  auto: { a: new THREE.Color("#10b981"), b: new THREE.Color("#3b82f6"), speed: 0.24, spread: 1.2 },
  escalate: { a: new THREE.Color("#f59e0b"), b: new THREE.Color("#ef4444"), speed: 0.3, spread: 1.1 },
};

function moodFor(stage: Stage, action?: string): Mood {
  if (stage === "retrieving" || stage === "retrieved" || stage === "generating") return "working";
  if (stage === "done") return action === "auto_handle" ? "auto" : "escalate";
  return "idle";
}

/**
 * Animated interactive 3D particle field for the Light Theme.
 */
export default function Background3D({ stage, action }: { stage: Stage; action?: string }) {
  const host = useRef<HTMLDivElement>(null);
  const target = useRef<Mood>("idle");
  target.current = moodFor(stage, action);

  useEffect(() => {
    const el = host.current;
    if (!el) return;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "low-power" });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.75));
    renderer.setSize(el.clientWidth, el.clientHeight);
    el.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2("#f8fafc", 0.08);
    const camera = new THREE.PerspectiveCamera(60, el.clientWidth / el.clientHeight, 0.1, 100);
    camera.position.set(0, 1.6, 6.2);

    const COLS = 100;
    const ROWS = 60;
    const count = COLS * ROWS;
    const positions = new Float32Array(count * 3);
    const base = new Float32Array(count * 2);
    let i = 0;
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        const x = (c / (COLS - 1) - 0.5) * 20;
        const z = (r / (ROWS - 1) - 0.5) * 16;
        positions[i * 3] = x;
        positions[i * 3 + 1] = 0;
        positions[i * 3 + 2] = z;
        base[i * 2] = x;
        base[i * 2 + 1] = z;
        i++;
      }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));

    const mat = new THREE.PointsMaterial({
      size: 0.08,
      transparent: true,
      opacity: 0.7,
      depthWrite: false,
      blending: THREE.NormalBlending,
    });
    mat.color = new THREE.Color("#6366f1");
    const points = new THREE.Points(geo, mat);
    scene.add(points);

    const cur = { a: MOODS.idle.a.clone(), b: MOODS.idle.b.clone(), speed: MOODS.idle.speed, spread: 1 };
    let t = 0;
    let raf = 0;
    let running = true;
    const clock = new THREE.Clock();

    let mouseX = 0;
    let mouseY = 0;
    let targetX = 0;
    let targetY = 0;

    const onMouseMove = (e: MouseEvent) => {
      mouseX = (e.clientX - window.innerWidth / 2) * 0.002;
      mouseY = (e.clientY - window.innerHeight / 2) * 0.002;
    };
    window.addEventListener("mousemove", onMouseMove);

    const onResize = () => {
      if (!el) return;
      renderer.setSize(el.clientWidth, el.clientHeight);
      camera.aspect = el.clientWidth / el.clientHeight;
      camera.updateProjectionMatrix();
    };
    window.addEventListener("resize", onResize);
    
    const onVis = () => {
      running = !document.hidden;
      if (running) {
        clock.getDelta();
        loop();
      }
    };
    document.addEventListener("visibilitychange", onVis);

    const tmp = new THREE.Color();
    function loop() {
      if (!running) return;
      raf = requestAnimationFrame(loop);
      const dt = Math.min(clock.getDelta(), 0.05);
      const m = MOODS[target.current];
      cur.a.lerp(m.a, 0.05);
      cur.b.lerp(m.b, 0.05);
      cur.speed += (m.speed - cur.speed) * 0.05;
      cur.spread += (m.spread - cur.spread) * 0.05;
      t += dt * (reduce ? 0 : 1);

      targetX = mouseX * 2;
      targetY = mouseY * 2;
      
      const arr = geo.attributes.position.array as Float32Array;
      for (let k = 0; k < count; k++) {
        const x = base[k * 2];
        const z = base[k * 2 + 1];
        const wave =
          Math.sin(x * 0.5 + t * cur.speed * 3) * 0.4 +
          Math.cos(z * 0.6 - t * cur.speed * 2.4) * 0.35 +
          Math.sin((x + z) * 0.3 + t * cur.speed) * 0.3;
        arr[k * 3 + 1] = wave * cur.spread;
      }
      geo.attributes.position.needsUpdate = true;

      const mid = tmp.copy(cur.a).lerp(cur.b, 0.5 + Math.sin(t * 0.4) * 0.4);
      mat.color.copy(mid);
      
      points.rotation.y = Math.sin(t * 0.05) * 0.08 + targetX * 0.5;
      points.rotation.x = targetY * 0.3;

      camera.position.x += (targetX - camera.position.x) * 0.05;
      camera.position.y += (1.6 - targetY - camera.position.y) * 0.05;
      camera.lookAt(0, 0, 0);
      
      renderer.render(scene, camera);
    }
    loop();

    return () => {
      running = false;
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
      document.removeEventListener("visibilitychange", onVis);
      window.removeEventListener("mousemove", onMouseMove);
      geo.dispose();
      mat.dispose();
      renderer.dispose();
      el.removeChild(renderer.domElement);
    };
  }, []);

  return <div ref={host} className="bg3d" aria-hidden="true" />;
}
