import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import ArchitectureDiagram from '../architecture/ArchitectureDiagram'

const root = document.getElementById('root')
if (!root) throw new Error('Missing #root element')

createRoot(root).render(
  <StrictMode>
    <ArchitectureDiagram onNodeSelect={(node) => console.log('selected:', node?.id ?? null)} />
  </StrictMode>,
)
