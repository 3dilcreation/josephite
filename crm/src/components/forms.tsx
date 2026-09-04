import { humanise } from "@/lib/format";

export function Field({
  label,
  name,
  type = "text",
  required,
  defaultValue,
  placeholder,
  step,
  className = "",
}: {
  label: string;
  name: string;
  type?: string;
  required?: boolean;
  defaultValue?: string | number | null;
  placeholder?: string;
  step?: string;
  className?: string;
}) {
  return (
    <div className={className}>
      <label className="label" htmlFor={name}>
        {label}
      </label>
      <input
        id={name}
        name={name}
        type={type}
        step={step}
        required={required}
        placeholder={placeholder}
        defaultValue={defaultValue ?? undefined}
        className="input"
      />
    </div>
  );
}

export function TextArea({
  label,
  name,
  rows = 3,
  defaultValue,
  placeholder,
  className = "",
}: {
  label: string;
  name: string;
  rows?: number;
  defaultValue?: string | null;
  placeholder?: string;
  className?: string;
}) {
  return (
    <div className={className}>
      <label className="label" htmlFor={name}>
        {label}
      </label>
      <textarea
        id={name}
        name={name}
        rows={rows}
        placeholder={placeholder}
        defaultValue={defaultValue ?? undefined}
        className="input"
      />
    </div>
  );
}

export function Select({
  label,
  name,
  options,
  defaultValue,
  includeBlank,
  blankLabel = "— none —",
  className = "",
}: {
  label: string;
  name: string;
  options: { value: string; label: string }[];
  defaultValue?: string | null;
  includeBlank?: boolean;
  blankLabel?: string;
  className?: string;
}) {
  return (
    <div className={className}>
      <label className="label" htmlFor={name}>
        {label}
      </label>
      <select id={name} name={name} defaultValue={defaultValue ?? ""} className="input">
        {includeBlank ? <option value="">{blankLabel}</option> : null}
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </div>
  );
}

/** Turn a Prisma enum object into <Select> options with readable labels. */
export function enumOptions(enumObject: Record<string, string>) {
  return Object.values(enumObject).map((value) => ({ value, label: humanise(value) }));
}
