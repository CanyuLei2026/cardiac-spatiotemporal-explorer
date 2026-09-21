import { useEffect, useMemo, useRef } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import * as THREE from 'three'
import { signalAt } from '../data.js'

function heatColor(value, min, max, target) {
  const span = Math.max(max - min, 1e-6)
  const t = THREE.MathUtils.clamp((value - min) / span, 0, 1)
  if (t < 0.25) {
    const u = t * 4
    target.setRGB(0.02 + u * 0.02, 0.12 + u * 0.36, 0.30 + u * 0.50)
  } else if (t < 0.5) {
    const u = (t - 0.25) * 4
    target.setRGB(0.04 + u * 0.18, 0.48 + u * 0.38, 0.80 + u * 0.12)
  } else if (t < 0.75) {
    const u = (t - 0.5) * 4
    target.setRGB(0.22 + u * 0.76, 0.86 + u * 0.02, 0.92 - u * 0.72)
  } else {
    const u = (t - 0.75) * 4
    target.setRGB(0.98, 0.88 - u * 0.70, 0.20 - u * 0.12)
  }
  return target
}

function normalizePoint(data, index) {
  const min = data.metadata.coordinateMin
  const max = data.metadata.coordinateMax
  const center = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2]
  const scale = 2.7 / Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2])
  return [
    (data.points[index * 3] - center[0]) * scale,
    (data.points[index * 3 + 1] - center[1]) * scale,
    (data.points[index * 3 + 2] - center[2]) * scale,
  ]
}

function Scene({ data, frame, filters, selected, onSelect, annotations, resetVersion, colorRange }) {
  const controls = useRef()
  const geometry = useRef()

  const activeIndices = useMemo(() => {
    const result = []
    for (let i = 0; i < data.metadata.pointCount; i += 1) {
      const regionMatch = filters.region === 'all' || data.regions[i] === Number(filters.region)
      const typeMatch = filters.pointType === 'all' || data.pointTypes[i] === Number(filters.pointType)
      const componentMatch = filters.component === 'all' || data.components[i] === Number(filters.component)
      if (regionMatch && typeMatch && componentMatch) result.push(i)
    }
    return result
  }, [data, filters])

  const { positions, colors } = useMemo(() => {
    const p = new Float32Array(activeIndices.length * 3)
    const c = new Float32Array(activeIndices.length * 3)
    const color = new THREE.Color()
    activeIndices.forEach((sourceIndex, displayIndex) => {
      const normalized = normalizePoint(data, sourceIndex)
      p.set(normalized, displayIndex * 3)
      heatColor(signalAt(data, sourceIndex, frame), colorRange[0], colorRange[1], color)
      c.set([color.r, color.g, color.b], displayIndex * 3)
    })
    return { positions: p, colors: c }
  }, [activeIndices, data, frame, colorRange])

  useEffect(() => {
    if (!geometry.current) return
    geometry.current.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    geometry.current.setAttribute('color', new THREE.BufferAttribute(colors, 3))
    geometry.current.computeBoundingSphere()
  }, [positions, colors])

  useEffect(() => {
    controls.current?.reset()
  }, [resetVersion])

  const annotationColors = { Normal: '#4ee6a8', Abnormal: '#ff6b5f', Landmark: '#f8d66d' }

  return (
    <>
      <color attach="background" args={['#06111b']} />
      <fog attach="fog" args={['#06111b', 5, 9]} />
      <ambientLight intensity={1.4} />
      <directionalLight position={[3, 4, 4]} intensity={2.2} />
      <points
        onClick={(event) => {
          event.stopPropagation()
          if (Number.isInteger(event.index)) onSelect(activeIndices[event.index])
        }}
      >
        <bufferGeometry ref={geometry} />
        <pointsMaterial vertexColors size={0.038} sizeAttenuation transparent opacity={0.94} />
      </points>

      {selected !== null && (
        <mesh position={normalizePoint(data, selected)}>
          <sphereGeometry args={[0.055, 18, 18]} />
          <meshStandardMaterial color="#ffffff" emissive="#70e1f5" emissiveIntensity={1.6} />
        </mesh>
      )}

      {annotations.map((annotation) => (
        <mesh key={annotation.pointIndex} position={normalizePoint(data, annotation.pointIndex)}>
          <sphereGeometry args={[0.07, 14, 14]} />
          <meshStandardMaterial
            color={annotationColors[annotation.label] || '#f8d66d'}
            emissive={annotationColors[annotation.label] || '#f8d66d'}
            emissiveIntensity={0.75}
          />
        </mesh>
      ))}

      <gridHelper args={[5, 10, '#173a4f', '#102736']} position={[0, -1.7, 0]} />
      <OrbitControls ref={controls} makeDefault enableDamping dampingFactor={0.08} />
    </>
  )
}

export default function HeartViewer(props) {
  const mobile = typeof window !== 'undefined' && window.matchMedia('(max-width: 760px)').matches
  return (
    <Canvas camera={{ position: [0.1, 0.15, mobile ? 6.1 : 4.3], fov: 42 }} dpr={[1, 2]} gl={{ antialias: true }}>
      <Scene {...props} />
    </Canvas>
  )
}
