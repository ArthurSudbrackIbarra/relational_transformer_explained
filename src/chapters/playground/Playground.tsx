/**
 * The playground: seed picker (left), prediction + context (center), ablation switches (right).
 *  - ≥ 1400px: three columns;
 *  - 1200-1399px: the seed picker spans the top (laid out in two columns by a container query), the
 *    prediction and the switches sit side by side under it;
 *  - md: seed picker and switches side by side, the prediction under them;
 *  - phones: one column (seed, prediction + context, switches).
 * Below 1200px the switch panel carries a compact live readout, so the prediction is never out of
 * sight while you flip a switch.
 */
import { AblationPanel } from './Ablations';
import { ContextView } from './ContextView';
import { History, PaperSideBySide, PredictionCard } from './Prediction';
import { SetupPanel } from './Setup';

export function Playground() {
  return (
    <div className="wide my-8">
      <div
        className="grid gap-4 [grid-template-areas:'setup'_'center'_'ablate'] md:grid-cols-2 md:[grid-template-areas:'setup_ablate'_'center_center'] min-[75rem]:grid-cols-[minmax(0,1fr)_292px] min-[75rem]:[grid-template-areas:'setup_setup'_'center_ablate'] min-[87.5rem]:grid-cols-[272px_minmax(0,1fr)_292px] min-[87.5rem]:[grid-template-areas:'setup_center_ablate']"
        role="region"
        aria-label="Zero-shot playground"
      >
        <div className="card min-w-0 px-4 py-4 [grid-area:setup]">
          <SetupPanel />
        </div>
        <div className="card flex min-w-0 flex-col gap-5 px-4 py-4 [grid-area:center] sm:px-5">
          <PredictionCard />
          <PaperSideBySide />
          <ContextView />
          <History />
        </div>
        <div className="card min-w-0 px-4 py-4 [grid-area:ablate]">
          <AblationPanel />
        </div>
      </div>
    </div>
  );
}
