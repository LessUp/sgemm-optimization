## Why

The current Pages site is structurally sound, but it still reads like an incremental documentation polish instead of a decisive technical whitepaper. The repository now needs a more opinionated public surface that upgrades the reading path, fixes theme-fragile figures, and strengthens the academic and comparative framing expected by senior GitHub readers.

## What Changes

- Rebuild the bilingual Pages information architecture around a stronger whitepaper flow: guide, system, algorithm ladder, evidence, and references.
- Replace theme-fragile light/dark figure handling with a site-aware shared figure system that follows manual theme switching reliably.
- Redesign the public theme, homepage, and section index pages to feel more deliberate, technical, and publication-grade.
- Add deeper whitepaper surfaces for system blueprint, performance framing, and reference-driven reading paths.
- Tighten docs validation so route structure and theme-aware figure usage are checked in CI-safe tests.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `project-presentation`: strengthen the public Pages information architecture, whitepaper hierarchy, theme-safe figure behavior, and research framing.
- `testing`: extend CI-safe documentation validation to cover the new mirrored routes and theme-aware figure conventions.

## Impact

- Affected code: `docs/.vitepress/*`, `docs/en/**`, `docs/zh/**`, `docs/public/figures/*`, and `docs/tests/*`
- Affected systems: GitHub Pages build, local docs test/build workflow, OpenSpec alignment for public presentation
- Breaking surface changes: public navigation labels, section emphasis, and some public route inventory will be reorganized for the rebuilt whitepaper flow
