/// <reference types="vite/client" />
// Pulls in @react-three/fiber's global JSX.IntrinsicElements augmentation, so <mesh>,
// <points>, <bufferGeometry> etc. type-check. Without this, tsc --noEmit fails at build
// time even though the dev server runs fine.
/// <reference types="@react-three/fiber" />
