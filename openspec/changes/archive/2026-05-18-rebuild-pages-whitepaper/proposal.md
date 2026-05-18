## Why

The current Pages site already has solid technical content, but it still reads like a polished documentation set rather than a decisive whitepaper-grade project narrative. This change rebuilds the public docs experience so interviewers and advanced GitHub readers can immediately see the repository as a rigorously argued CUDA SGEMM case study, with stronger information architecture, clearer visual hierarchy, and traceable evidence.

## What Changes

- Rebuild the bilingual GitHub Pages information architecture around a whitepaper-and-academy model, with clearer progression from executive orientation to deep technical study.
- Realign the VitePress docs stack and shared site structure with the `kimi-cli` docs baseline where that baseline improves maintainability, navigation consistency, and long-term clarity.
- Replace the current theme treatment with a more deliberate visual system that improves typography, spacing, contrast, and dark/light parity without depending on fragile one-off styling.
- Rework homepage and major entry pages so architecture, methodology, validation, and learning routes read as one coherent technical argument rather than adjacent sections.
- Upgrade diagrams, citations, and reference surfaces so visual assets remain legible in both themes and the content presents stronger academic and comparative context.
- Tighten README-to-Pages alignment so repository entry points, Pages navigation, and evidence framing tell the same public story.

## Capabilities

### New Capabilities

- None.

### Modified Capabilities

- `project-presentation`: Strengthen requirements for whitepaper-grade Pages structure, mirrored bilingual academy navigation, theme-safe technical visuals, and evidence-backed public references.

## Impact

- Affected docs stack: `docs/package.json`, `docs/.vitepress/**/*`, `docs/public/**/*`
- Affected public content: `docs/en/**/*`, `docs/zh/**/*`, `README.md`, `README.zh-CN.md`
- Affected governance: `openspec/changes/rebuild-pages-whitepaper/specs/project-presentation/spec.md`
- Validation impact: docs route tests, docs build, and `openspec validate --all` remain required for completion
