# Architecture diagram (React)

An interactive, dependency-free React component that renders the
[production-rag-eval](../../README.md) architecture: offline indexing, the online
query → answer path with a conformal abstention gate, and the evaluation layer.

It is the same diagram shipped as static images under
[`docs/assets`](../../docs/assets), rebuilt as a component so it can live in a
portfolio site, docs page, or slide deck.

## Files

- `ArchitectureDiagram.tsx` — the component (data-driven; the diagram is defined
  by the `PHASES`, `LEGEND`, and `STATS` constants at the top).
- `ArchitectureDiagram.css` — design tokens + styles. Light and dark themes,
  responsive (columns collapse on narrow containers), compositor-friendly motion,
  and `prefers-reduced-motion` support.

## Requirements

React 18+ (uses `useId`). No other runtime dependencies.

## Run the demo

A minimal Vite + TypeScript demo lives in the parent [`web/`](..) folder:

```bash
cd web
npm install
npm run dev      # open the printed localhost URL
npm run build    # type-check + production build
```

## Usage

```tsx
import ArchitectureDiagram from './web/architecture/ArchitectureDiagram'

export default function Page() {
  return (
    <main>
      <ArchitectureDiagram onNodeSelect={(node) => console.log(node?.id ?? 'cleared')} />
    </main>
  )
}
```

### Props

| Prop | Type | Description |
|---|---|---|
| `onNodeSelect` | `(node \| null) => void` | Fired when a node is clicked (or cleared). |
| `className` | `string` | Extra class on the root `<section>`. |

## Interaction

- **Hover** a node to lift it.
- **Click** (or focus + `Enter`/`Space`) a node to open a detail panel describing it.
- Click the same node again, or the panel's close button, to clear.

## Customizing

Everything visual is driven by CSS custom properties on `.rag-arch` (palette,
spacing, radii, motion) and the data constants in the component. Edit the
`PHASES` array to change nodes, the per-kind colors in the CSS to re-theme, or
the `STATS` array to swap the headline metrics.
