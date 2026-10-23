/**
 * The strip above each figure that says WHICH prediction it shows: the task, the seed entity,
 * the checkpoint, and whether this is the store's live simulation or a local preview of the other
 * target type (see live.ts). A local preview can be promoted to the global task.
 */
import { Button, StatusBadge } from '@/components';
import { useSim } from '@/state/store';
import { seedEntity, taskLabel, type Lane } from './live';

export function LaneHeader({ lane, className }: { lane: Lane; className?: string }) {
  const setTask = useSim((s) => s.setTask);
  const ent = seedEntity(lane);
  const split = lane.task.kind === 'autocomplete' ? 'all rows' : lane.split;
  return (
    <div className={`mb-4 flex flex-wrap items-center gap-x-3 gap-y-2 rounded-lg bg-surface-2 px-3 py-2 text-[15px] text-ink-2 ${className ?? ''}`}>
      <span className="font-semibold text-ink">{lane.isGlobal ? 'Live context' : 'Local preview'}</span>
      <span>
        task <span className="text-ink">{taskLabel(lane.task)}</span> ({lane.task.targetType})
      </span>
      <span>
        seed: <span className="text-ink">{ent.label}</span>
        {ent.name ? ` (${ent.name})` : ''}, {split}
      </span>
      <span>
        checkpoint <span className="font-mono text-ink">{lane.ckpt}</span>
      </span>
      {lane.assets.weightsKind === 'init' && <StatusBadge status="warning">untrained weights</StatusBadge>}
      {lane.disabled.length > 0 && <StatusBadge status="warning">ablation on: {lane.disabled.join(', ')} switched off</StatusBadge>}
      {!lane.isGlobal && (
        <Button size="sm" variant="ghost" className="ml-auto" onClick={() => setTask(lane.task.id)}>
          Make it the global task
        </Button>
      )}
    </div>
  );
}
