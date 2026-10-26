/**
 * Form controls in the site's quiet style. Chapters use these for figure
 * controls so every interactive figure looks and behaves the same.
 * All are plain, accessible HTML controls (native semantics, visible focus).
 */
import { useId, type ButtonHTMLAttributes, type ReactNode, type SelectHTMLAttributes } from 'react';
import { cx } from './cx';

// ─────────────────────────────────────────────────────────────────────────────
// Button
// ─────────────────────────────────────────────────────────────────────────────

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  /** primary = accent fill (one per view), secondary = outlined, ghost = text only. */
  variant?: 'primary' | 'secondary' | 'ghost';
  size?: 'sm' | 'md';
  icon?: ReactNode;
  iconRight?: ReactNode;
}

/** Button with optional leading/trailing icon. */
export function Button({ variant = 'secondary', size = 'md', icon, iconRight, className, children, type, ...rest }: ButtonProps) {
  return (
    <button
      type={type ?? 'button'}
      className={cx(
        'inline-flex select-none items-center justify-center gap-1.5 rounded-lg font-medium whitespace-nowrap transition-colors disabled:opacity-45',
        size === 'sm' ? 'h-8 px-2.5 text-[15.5px]' : 'h-10 px-4 text-[17px]',
        variant === 'primary' && 'bg-accent text-accent-ink hover:brightness-110 active:brightness-95',
        variant === 'secondary' &&
          'border border-border-strong bg-surface text-ink hover:bg-surface-2 active:bg-surface-3',
        variant === 'ghost' && 'text-ink-2 hover:bg-surface-2 hover:text-ink active:bg-surface-3',
        className,
      )}
      {...rest}
    >
      {icon}
      {children}
      {iconRight}
    </button>
  );
}

export interface IconButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  /** Required accessible name (also the native tooltip). */
  label: string;
  size?: 'sm' | 'md';
  active?: boolean;
}

/** Square icon-only button. `label` becomes aria-label and title. */
export function IconButton({ label, size = 'md', active, className, children, type, ...rest }: IconButtonProps) {
  return (
    <button
      type={type ?? 'button'}
      aria-label={label}
      title={label}
      className={cx(
        'inline-flex shrink-0 items-center justify-center rounded-lg text-ink-2 transition-colors hover:bg-surface-2 hover:text-ink active:bg-surface-3 disabled:opacity-40',
        size === 'sm' ? 'h-8 w-8' : 'h-9 w-9',
        active && 'bg-accent-soft text-accent-text',
        className,
      )}
      {...rest}
    >
      {children}
    </button>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Segmented control (radio group)
// ─────────────────────────────────────────────────────────────────────────────

export interface SegmentedOption<T extends string | number> {
  value: T;
  label: ReactNode;
  disabled?: boolean;
  /** Tooltip / accessible description. */
  title?: string;
}

export interface SegmentedProps<T extends string | number> {
  value: T;
  onChange: (value: T) => void;
  options: SegmentedOption<T>[];
  /** Accessible name of the group. */
  label: string;
  size?: 'sm' | 'md';
  className?: string;
  disabled?: boolean;
}

/** A row of mutually exclusive options (role="radiogroup"; arrow keys move). */
export function Segmented<T extends string | number>({
  value,
  onChange,
  options,
  label,
  size = 'sm',
  className,
  disabled,
}: SegmentedProps<T>) {
  const name = useId();
  return (
    <div
      role="radiogroup"
      aria-label={label}
      className={cx(
        'inline-flex max-w-full flex-wrap items-center gap-0.5 rounded-lg border border-border bg-surface-2 p-0.5',
        disabled && 'opacity-50',
        className,
      )}
    >
      {options.map((o) => {
        const checked = o.value === value;
        return (
          <label
            key={String(o.value)}
            title={o.title}
            className={cx(
              'relative inline-flex cursor-pointer items-center justify-center rounded-md font-medium transition-colors has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-accent',
              size === 'sm' ? 'h-8 px-2.5 text-[15px]' : 'h-9 px-3 text-[16px]',
              checked ? 'bg-surface text-ink shadow-[0_1px_2px_rgba(0,0,0,0.08)]' : 'text-ink-2 hover:text-ink',
              (o.disabled || disabled) && 'cursor-not-allowed opacity-50',
            )}
          >
            <input
              type="radio"
              className="sr-only"
              name={name}
              checked={checked}
              disabled={o.disabled || disabled}
              onChange={() => onChange(o.value)}
            />
            {o.label}
          </label>
        );
      })}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Select (native)
// ─────────────────────────────────────────────────────────────────────────────

export interface SelectProps extends Omit<SelectHTMLAttributes<HTMLSelectElement>, 'size'> {
  label?: ReactNode;
  /** Render the label visually hidden (still accessible). */
  hideLabel?: boolean;
  size?: 'sm' | 'md';
  hint?: ReactNode;
}

/** Native <select> with a label above it. */
export function Select({ label, hideLabel, size = 'sm', hint, className, id, children, ...rest }: SelectProps) {
  const auto = useId();
  const sid = id ?? auto;
  return (
    <div className={cx('flex min-w-0 flex-col gap-1', className)}>
      {label && (
        <label htmlFor={sid} className={cx('text-[15px] font-medium text-ink-2', hideLabel && 'sr-only')}>
          {label}
        </label>
      )}
      <div className="relative">
        <select
          id={sid}
          className={cx(
            'w-full min-w-0 appearance-none rounded-lg border border-border-strong bg-surface pr-8 pl-2.5 text-ink transition-colors hover:bg-surface-2 disabled:opacity-50',
            size === 'sm' ? 'h-8 text-[15.5px]' : 'h-10 text-[17px]',
          )}
          {...rest}
        >
          {children}
        </select>
        <svg
          aria-hidden="true"
          viewBox="0 0 16 16"
          className="pointer-events-none absolute top-1/2 right-2.5 h-3.5 w-3.5 -translate-y-1/2 text-muted"
        >
          <path d="M4 6l4 4 4-4" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </div>
      {hint && <p className="text-[14.5px] leading-snug text-muted">{hint}</p>}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Slider
// ─────────────────────────────────────────────────────────────────────────────

export interface SliderProps {
  label: ReactNode;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (value: number) => void;
  /** Formatter for the value readout. */
  format?: (v: number) => ReactNode;
  hint?: ReactNode;
  className?: string;
  disabled?: boolean;
}

/** Range slider with label and live value readout. */
export function Slider({ label, value, min, max, step = 1, onChange, format, hint, className, disabled }: SliderProps) {
  const id = useId();
  return (
    <div className={cx('flex min-w-0 flex-col gap-1', className)}>
      <div className="flex items-baseline justify-between gap-2">
        <label htmlFor={id} className="text-[15px] font-medium text-ink-2">
          {label}
        </label>
        <output htmlFor={id} className="num text-[15px] text-ink">
          {format ? format(value) : value}
        </output>
      </div>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
        className="h-5 w-full cursor-pointer disabled:cursor-not-allowed"
      />
      {hint && <p className="text-[14.5px] leading-snug text-muted">{hint}</p>}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// NumberField
// ─────────────────────────────────────────────────────────────────────────────

export interface NumberFieldProps {
  label: ReactNode;
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  hint?: ReactNode;
  className?: string;
  disabled?: boolean;
  /** Shown while the field is empty (pass value = NaN for an empty field). */
  placeholder?: string;
}

/** Integer/float input; commits on change when the value parses. */
export function NumberField({ label, value, onChange, min, max, step = 1, hint, className, disabled, placeholder }: NumberFieldProps) {
  const id = useId();
  return (
    <div className={cx('flex min-w-0 flex-col gap-1', className)}>
      <label htmlFor={id} className="text-[15px] font-medium text-ink-2">
        {label}
      </label>
      <input
        id={id}
        type="number"
        inputMode="numeric"
        value={Number.isFinite(value) ? value : ''}
        placeholder={placeholder}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        onChange={(e) => {
          const v = Number(e.target.value);
          if (e.target.value !== '' && Number.isFinite(v)) onChange(v);
        }}
        className="num h-8 w-full min-w-0 rounded-lg border border-border-strong bg-surface px-2.5 text-[15.5px] text-ink disabled:opacity-50"
      />
      {hint && <p className="text-[14.5px] leading-snug text-muted">{hint}</p>}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Switch
// ─────────────────────────────────────────────────────────────────────────────

export interface SwitchProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label: ReactNode;
  className?: string;
  disabled?: boolean;
}

/** On/off switch (role="switch"). */
export function Switch({ checked, onChange, label, className, disabled }: SwitchProps) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={cx('group inline-flex items-center gap-2 text-[15.5px] text-ink-2 disabled:opacity-50', className)}
    >
      <span
        aria-hidden="true"
        className={cx(
          'relative inline-block h-[18px] w-[30px] shrink-0 rounded-full border transition-colors',
          checked ? 'border-accent bg-accent' : 'border-border-strong bg-surface-3',
        )}
      >
        <span
          className={cx(
            'absolute top-[2px] h-3 w-3 rounded-full transition-[left]',
            checked ? 'left-[14px] bg-accent-ink' : 'left-[2px] bg-surface',
          )}
        />
      </span>
      <span className="group-hover:text-ink">{label}</span>
    </button>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Kbd
// ─────────────────────────────────────────────────────────────────────────────

/** Keyboard key hint (the text font: a key is not code). */
export function Kbd({ children }: { children: ReactNode }) {
  return (
    <kbd className="inline-flex h-6 min-w-6 items-center justify-center rounded border border-border-strong bg-surface-2 px-1.5 font-sans text-[14px] text-ink-2">
      {children}
    </kbd>
  );
}

/** A labeled row of figure controls (sits above the figure it scopes; dataviz "filters in one row"). */
export function ControlRow({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={cx('flex flex-wrap items-end gap-x-4 gap-y-3', className)}>{children}</div>;
}
