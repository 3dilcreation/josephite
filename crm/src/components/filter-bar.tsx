"use client";

import { usePathname, useRouter, useSearchParams } from "next/navigation";

type FilterDef = { name: string; label: string; options: { value: string; label: string }[] };

/**
 * Filters live in the URL, not in component state, so a filtered view can be
 * bookmarked, shared in WhatsApp, and survives a refresh.
 */
export function FilterBar({ filters, searchPlaceholder }: { filters: FilterDef[]; searchPlaceholder?: string }) {
  const router = useRouter();
  const pathname = usePathname();
  const params = useSearchParams();

  function apply(name: string, value: string) {
    const next = new URLSearchParams(params.toString());
    if (value) next.set(name, value);
    else next.delete(name);
    next.delete("page");
    router.push(`${pathname}?${next.toString()}`);
  }

  return (
    <form
      className="mb-4 flex flex-wrap items-end gap-2"
      onSubmit={(event) => {
        event.preventDefault();
        const value = new FormData(event.currentTarget).get("q");
        apply("q", typeof value === "string" ? value : "");
      }}
    >
      <div className="min-w-[12rem] flex-1">
        <label className="label" htmlFor="q">
          Search
        </label>
        <input
          id="q"
          name="q"
          defaultValue={params.get("q") ?? ""}
          placeholder={searchPlaceholder ?? "Name, phone, email…"}
          className="input"
        />
      </div>

      {filters.map((filter) => (
        <div key={filter.name} className="min-w-[9rem]">
          <label className="label" htmlFor={filter.name}>
            {filter.label}
          </label>
          <select
            id={filter.name}
            name={filter.name}
            value={params.get(filter.name) ?? ""}
            onChange={(event) => apply(filter.name, event.target.value)}
            className="input"
          >
            <option value="">All</option>
            {filter.options.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>
      ))}

      <div className="min-w-[9rem]">
        <label className="label" htmlFor="from">
          From
        </label>
        <input
          id="from"
          type="date"
          value={params.get("from") ?? ""}
          onChange={(event) => apply("from", event.target.value)}
          className="input"
        />
      </div>
      <div className="min-w-[9rem]">
        <label className="label" htmlFor="to">
          To
        </label>
        <input
          id="to"
          type="date"
          value={params.get("to") ?? ""}
          onChange={(event) => apply("to", event.target.value)}
          className="input"
        />
      </div>

      <button type="submit" className="btn btn-primary">
        Apply
      </button>
      <button type="button" className="btn btn-ghost" onClick={() => router.push(pathname)}>
        Reset
      </button>
    </form>
  );
}
