import { Fragment, useId, useState, type KeyboardEvent } from 'react'
import './ArchitectureDiagram.css'

type NodeKind = 'store' | 'proc' | 'gate' | 'good' | 'neutral'
type ConnectorVariant = 'down' | 'split' | 'merge' | 'branch'

interface DiagramNode {
  id: string
  title: string
  subtitle?: string
  kind: NodeKind
  detail: string
}

interface Phase {
  id: string
  index: string
  title: string
  flex: number
  /** Each row renders horizontally; rows stack with a connector between them. */
  rows: DiagramNode[][]
}

interface Stat {
  figure: string
  label: string
  kind: Extract<NodeKind, 'gate' | 'good' | 'proc'>
}

const PHASES: Phase[] = [
  {
    id: 'index',
    index: '①',
    title: 'Offline · indexing (run once)',
    flex: 0.85,
    rows: [
      [{ id: 'financebench', title: 'FinanceBench', subtitle: '150 Q&A · 10-K chunks', kind: 'store', detail: '150 question/answer pairs over real SEC 10-K filings, each paired with its gold evidence passage.' }],
      [{ id: 'chunker', title: 'Chunker', subtitle: '512-token recursive', kind: 'proc', detail: 'Recursive 512-token splitter that preserves character offsets so citations can map back to source spans.' }],
      [{ id: 'context', title: 'Contextual prefix', subtitle: 'Claude Haiku per chunk', kind: 'proc', detail: 'Anthropic Contextual Retrieval: Claude Haiku writes a short situating prefix for every chunk. Cached in SQLite.' }],
      [{ id: 'hybrid-index', title: 'Hybrid index', subtitle: 'Qdrant + BM25', kind: 'store', detail: 'voyage-3-large dense vectors (dim=256) in Qdrant alongside a rank-bm25 lexical index — built once, served to every query.' }],
    ],
  },
  {
    id: 'serve',
    index: '②',
    title: 'Online · query → answer',
    flex: 1.3,
    rows: [
      [{ id: 'query', title: 'Query', kind: 'neutral', detail: 'A natural-language question about a filing — revenue, ratios, segment data, and so on.' }],
      [
        { id: 'bm25', title: 'BM25 retrieve', subtitle: 'top-50 lexical', kind: 'proc', detail: 'Sparse lexical retrieval over the rank-bm25 index — strong on exact terms and figures.' },
        { id: 'dense', title: 'Dense retrieve', subtitle: 'voyage-3-large · top-50', kind: 'proc', detail: 'Dense semantic retrieval over Qdrant — strong on paraphrase and conceptual matches.' },
      ],
      [{ id: 'rrf', title: 'RRF fusion', subtitle: 'k = 60', kind: 'proc', detail: 'Reciprocal Rank Fusion merges the two ranked lists into one without tuning per-retriever weights.' }],
      [{ id: 'rerank', title: 'Cohere Rerank 3.5', subtitle: 'top-10 passages', kind: 'proc', detail: 'A cross-encoder reranks the fused candidates down to the 10 passages handed to generation.' }],
      [{ id: 'gate', title: 'Conformal gate', subtitle: 'τ · α = 0.10', kind: 'gate', detail: 'Split conformal prediction calibrates a confidence threshold τ. Below it the system abstains instead of guessing.' }],
      [
        { id: 'answer', title: 'Claude Sonnet 4.6', subtitle: 'Citations API · grounded', kind: 'good', detail: 'Generation via the Anthropic Citations API — every answer cites the source passages it used.' },
        { id: 'abstain', title: 'Abstain', subtitle: 'insufficient confidence', kind: 'neutral', detail: 'When retrieval confidence is below τ, the system returns "I don\'t know" rather than a likely hallucination.' },
      ],
    ],
  },
  {
    id: 'eval',
    index: '③',
    title: 'Evaluation',
    flex: 0.85,
    rows: [
      [{ id: 'ragas', title: 'RAGAS', subtitle: 'faithfulness', kind: 'neutral', detail: 'Faithfulness, answer relevancy, and context precision, judged by a Claude Sonnet evaluator.' }],
      [{ id: 'hhem', title: 'Vectara HHEM', subtitle: 'hallucination', kind: 'neutral', detail: 'HHEM-2.1-Open cross-encoder scores how well the answer is supported by the retrieved context.' }],
      [{ id: 'deepeval', title: 'DeepEval', subtitle: 'G-Eval financial', kind: 'neutral', detail: 'A G-Eval rubric tuned for financial correctness, judged by Claude.' }],
      [{ id: 'citation', title: 'Citation coverage', subtitle: '≥ 1 per answer', kind: 'good', detail: 'A hard gate: every answered question must cite at least one source passage. Measured at 1.00.' }],
    ],
  },
]

const PHASE_LINKS = ['serves', 'scored'] as const

const LEGEND: { kind: NodeKind; label: string }[] = [
  { kind: 'store', label: 'Store / index' },
  { kind: 'proc', label: 'Processing' },
  { kind: 'gate', label: 'Decision gate' },
  { kind: 'good', label: 'Grounded answer' },
  { kind: 'neutral', label: 'Neutral / abstain' },
]

const STATS: Stat[] = [
  { figure: '0.83', label: 'RAGAS faithfulness', kind: 'gate' },
  { figure: '1.00', label: 'citation coverage', kind: 'good' },
  { figure: '7%', label: 'abstention rate', kind: 'proc' },
  { figure: '$0.011', label: 'cost per question', kind: 'good' },
  { figure: '0.91', label: 'dense recall@10', kind: 'good' },
]

function connectorVariant(prev: DiagramNode[], current: DiagramNode[]): ConnectorVariant {
  if (prev.length === 1 && current.length === 2) return prev[0].id === 'gate' ? 'branch' : 'split'
  if (prev.length === 2 && current.length === 1) return 'merge'
  return 'down'
}

export interface ArchitectureDiagramProps {
  /** Notified whenever the selected node changes (null when cleared). */
  onNodeSelect?: (node: DiagramNode | null) => void
  className?: string
}

export default function ArchitectureDiagram({ onNodeSelect, className }: ArchitectureDiagramProps) {
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const headingId = useId()

  const allNodes = PHASES.flatMap((p) => p.rows.flat())
  const selected = allNodes.find((n) => n.id === selectedId) ?? null

  const select = (node: DiagramNode) => {
    const next = node.id === selectedId ? null : node
    setSelectedId(next?.id ?? null)
    onNodeSelect?.(next)
  }

  return (
    <section className={['rag-arch', className].filter(Boolean).join(' ')} aria-labelledby={headingId}>
      <header className="rag-arch__head">
        <h2 id={headingId} className="rag-arch__title">
          Production RAG · FinanceBench
        </h2>
        <p className="rag-arch__subtitle">
          Hybrid retrieval · contextual chunks · conformal abstention · grounded generation · triple eval
        </p>
        <ul className="rag-arch__legend" aria-label="Node types">
          {LEGEND.map((item) => (
            <li key={item.kind} className="rag-arch__legend-item">
              <span className={`rag-arch__swatch rag-arch__swatch--${item.kind}`} aria-hidden="true" />
              {item.label}
            </li>
          ))}
        </ul>
      </header>

      <div className="rag-arch__phases">
        {PHASES.map((phase, phaseIndex) => (
          <Fragment key={phase.id}>
            <article className="rag-arch__phase" style={{ flex: phase.flex }} aria-label={phase.title}>
              <h3 className="rag-arch__phase-title">
                <span className="rag-arch__phase-index" aria-hidden="true">
                  {phase.index}
                </span>
                {phase.title}
              </h3>

              <div className="rag-arch__flow">
                {phase.rows.map((row, rowIndex) => (
                  <div key={rowIndex} className="rag-arch__row-group">
                    {rowIndex > 0 && <Connector variant={connectorVariant(phase.rows[rowIndex - 1], row)} />}
                    <div className="rag-arch__row" data-count={row.length}>
                      {row.map((node) => (
                        <Node key={node.id} node={node} selected={node.id === selectedId} onSelect={select} />
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </article>

            {phaseIndex < PHASES.length - 1 && <PhaseArrow label={PHASE_LINKS[phaseIndex]} />}
          </Fragment>
        ))}
      </div>

      <footer className="rag-arch__footer">
        <p className="rag-arch__footnote">
          Measured on FinanceBench test split (n=30) — reproduced from the harness, not hand-picked
        </p>
        <ul className="rag-arch__stats">
          {STATS.map((stat) => (
            <li key={stat.label} className="rag-arch__stat">
              <span className={`rag-arch__stat-figure rag-arch__stat-figure--${stat.kind}`}>{stat.figure}</span>
              <span className="rag-arch__stat-label">{stat.label}</span>
            </li>
          ))}
        </ul>
      </footer>

      {selected && (
        <aside className="rag-arch__detail" role="status" aria-live="polite">
          <div className="rag-arch__detail-body">
            <p className="rag-arch__detail-title">{selected.title}</p>
            <p className="rag-arch__detail-text">{selected.detail}</p>
          </div>
          <button type="button" className="rag-arch__detail-close" onClick={() => select(selected)} aria-label="Close detail">
            ×
          </button>
        </aside>
      )}
    </section>
  )
}

function Node({
  node,
  selected,
  onSelect,
}: {
  node: DiagramNode
  selected: boolean
  onSelect: (node: DiagramNode) => void
}) {
  const onKeyDown = (event: KeyboardEvent<HTMLButtonElement>) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault()
      onSelect(node)
    }
  }

  return (
    <button
      type="button"
      className={`rag-arch__node rag-arch__node--${node.kind}${selected ? ' is-selected' : ''}`}
      aria-pressed={selected}
      onClick={() => onSelect(node)}
      onKeyDown={onKeyDown}
    >
      <span className="rag-arch__node-title">{node.title}</span>
      {node.subtitle && <span className="rag-arch__node-subtitle">{node.subtitle}</span>}
    </button>
  )
}

function Connector({ variant }: { variant: ConnectorVariant }) {
  if (variant === 'down') {
    return (
      <span className="rag-arch__conn rag-arch__conn--down" aria-hidden="true">
        <span className="rag-arch__stem" />
        <span className="rag-arch__head" />
      </span>
    )
  }

  if (variant === 'merge') {
    return (
      <span className="rag-arch__conn rag-arch__conn--merge" aria-hidden="true">
        <span className="rag-arch__drop rag-arch__drop--l" />
        <span className="rag-arch__drop rag-arch__drop--r" />
        <span className="rag-arch__bar" />
        <span className="rag-arch__stem" />
        <span className="rag-arch__head" />
      </span>
    )
  }

  // split or branch (split + answer/abstain labels)
  const isBranch = variant === 'branch'
  return (
    <span
      className={`rag-arch__conn rag-arch__conn--split${isBranch ? ' rag-arch__conn--branch' : ''}`}
      aria-hidden="true"
    >
      <span className="rag-arch__stem" />
      <span className="rag-arch__bar" />
      <span className="rag-arch__drop rag-arch__drop--l" />
      <span className="rag-arch__drop rag-arch__drop--r" />
      <span className="rag-arch__head rag-arch__head--l" />
      <span className="rag-arch__head rag-arch__head--r" />
      {isBranch && (
        <>
          <span className="rag-arch__branch-label rag-arch__branch-label--l">answer</span>
          <span className="rag-arch__branch-label rag-arch__branch-label--r">abstain</span>
        </>
      )}
    </span>
  )
}

function PhaseArrow({ label }: { label: string }) {
  return (
    <span className="rag-arch__phase-arrow" aria-hidden="true">
      <span className="rag-arch__phase-arrow-label">{label}</span>
      <span className="rag-arch__phase-arrow-line" />
      <span className="rag-arch__phase-arrow-head" />
    </span>
  )
}
