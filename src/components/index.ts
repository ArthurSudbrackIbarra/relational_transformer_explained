/**
 * Shared UI + visualization components (BUILD_PLAN §5). Chapters import from
 * here: `import { Section, Figure, Cite, TokenStrip } from '@/components'`.
 *
 * Visualization components take plain props (arrays, callbacks, highlight
 * sets) and never import the engine (only types from src/engine/types.ts), so
 * they are reusable and demoed with fake data at /#/dev/components.
 */

// shell
export { Layout, ContentColumn } from './shell/Layout';
export { ChapterPage, ChapterHeader, Section, Figure, ChapterNav } from './shell/Chapter';
export type { ChapterHeaderProps, SectionProps, FigureProps } from './shell/Chapter';
export { ChapterPlaceholder } from './shell/ChapterPlaceholder';
export { SimIndicator } from './shell/SimIndicator';
export { ThemeToggle } from './shell/ThemeToggle';
export { Logo } from './shell/Logo';

// prose
export { Callout } from './prose/Callout';
export type { CalloutProps, CalloutVariant } from './prose/Callout';
export { Cite } from './prose/Cite';
export type { CiteProps } from './prose/Cite';
export { CodeRef } from './prose/CodeRef';
export type { CodeRefProps } from './prose/CodeRef';
export { Math, TeX, InlineMath, MathBlock } from './prose/Math';
export { ScaleNote, SCALE_ROWS } from './prose/ScaleNote';
export { Term } from './prose/Term';
export { Quiz } from './prose/Quiz';
export type { QuizQuestion, QuizChoice } from './prose/Quiz';
export { PAPER, CODE, codeUrl, paperUrl } from './prose/sources';
export { GLOSSARY, getGlossaryEntry } from './glossary';
export type { GlossaryEntry, GlossaryId } from './glossary';

// data views
export { SchemaDiagram } from './viz/SchemaDiagram';
export type { SchemaDiagramProps, SchemaTableSpec } from './viz/SchemaDiagram';
export { TableView } from './viz/TableView';
export type { TableViewProps } from './viz/TableView';
export { EntityGraph, edgeKey } from './viz/EntityGraph';
export type { EntityGraphProps, GraphNodeSpec, GraphEdgeSpec, NodeId, RingStyle } from './viz/EntityGraph';
export { TokenStrip, tokenValueText, tokenStoredValue } from './viz/TokenStrip';
export type { TokenStripProps } from './viz/TokenStrip';
export { TokenTable, ALL_TOKEN_FIELDS, FIELD_USED_BY_MODEL } from './viz/TokenTable';
export type { TokenTableProps, TokenField } from './viz/TokenTable';
export { ShapeTag } from './viz/ShapeTag';
export type { Shape, ShapeTagProps } from './viz/ShapeTag';
export { VectorStrip } from './viz/VectorStrip';
export type { VectorStripProps } from './viz/VectorStrip';
export { Heatmap, ScaleLegend, HEATMAP_GEOMETRY } from './viz/Heatmap';
export type { HeatmapProps, HeatmapCell } from './viz/Heatmap';
export { MaskMatrix } from './viz/MaskMatrix';
export type { MaskMatrixProps } from './viz/MaskMatrix';
export { StepPlayer, useStepPlayer, SPEEDS } from './viz/StepPlayer';
export type { StepPlayerProps, StepPlayerState, UseStepPlayerOptions } from './viz/StepPlayer';
export { LineChart } from './viz/LineChart';
export type { LineChartProps, LineSeries, LinePoint } from './viz/LineChart';
export { BarChart } from './viz/BarChart';
export type { BarChartProps, BarSeries } from './viz/BarChart';
export { Stat } from './viz/Stat';
export type { StatProps } from './viz/Stat';
export { SemBadge } from './viz/SemBadge';
export { TableSwatch } from './viz/TableSwatch';
export { LAYER_INFO, LayerLabel } from './viz/layers';
export { Legend, ChartFrame, DataTable } from './viz/chartParts';
export { ShapeGlyph, shapeOfSlot, shapePath, SLOT_SHAPES } from './viz/shapes';
export * from './viz/format';
export { SVG_TYPE, textWidth, maxTextWidth, fitText } from './viz/text';
export { sequentialCss, divergingCss, sequentialRgb, divergingRgb, inkOn, maxAbs, extent } from './viz/scales';

// ui primitives
export { Button, IconButton, Segmented, Select, Slider, NumberField, Switch, Kbd, ControlRow } from './ui/controls';
export { StatusBadge } from './ui/StatusBadge';
export type { Status } from './ui/StatusBadge';
export { ErrorBoundary } from './ui/ErrorBoundary';
export { HoverCard, Popover, Portal } from './ui/floating';
export { cx } from './ui/cx';
