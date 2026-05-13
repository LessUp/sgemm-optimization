## Why

The project already has solid CUDA implementation depth, but its public presentation still undersells that depth for two key audiences: interviewers and advanced community readers. The repository and Pages surfaces also need tighter narrative alignment so visitors can quickly understand both technical value and engineering rigor.

## What Changes

- Restructure GitHub Pages information architecture around three high-signal tracks: project highlights, interview playbook, and references.
- Rewrite bilingual home pages to combine visual clarity, benchmark credibility, and direct learning paths.
- Add mirrored English/Chinese pages for highlights, interview storytelling, and paper/repo references.
- Refresh VitePress theme styling with a lighter NVIDIA-inspired visual language while preserving readability and mobile usability.
- Align README and README.zh-CN with the new Pages narrative and fix stale documentation links.
- Add an OpenSpec delta to capture interview-readiness and citation-quality expectations in project presentation requirements.

## Capabilities

### New Capabilities

- None.

### Modified Capabilities

- `project-presentation`: Tighten requirements so README, Pages structure, bilingual mirrors, interview-readiness, and public references remain consistent and evidence-backed.

## Impact

- Affected documentation surfaces: `docs/.vitepress/*`, `docs/en/*`, `docs/zh/*`, `README.md`, `README.zh-CN.md`
- Affected governance artifacts: `openspec/changes/revamp-public-presentation/specs/project-presentation/spec.md`
- Validation impact: `npm --prefix docs run build` and `openspec validate --all` become mandatory closeout checks for this change.
