## Context

The repository already uses VitePress and has recently improved its public presentation, but the current site still behaves like a good project manual rather than a top-tier architecture whitepaper. The strongest content is present, yet it is distributed across route groups that do not fully match the intended reader journey: first orientation, then system framing, then optimization study, then validation, then research lineage.

The local `kimi-cli` docs stack is the desired engineering baseline. It uses the same VitePress generation family already present here, but with a cleaner config posture: centralized locale navigation, simpler theme bootstrapping, and clearer separation between content structure and styling. This repository should adopt that baseline where it improves maintainability, while still keeping SGEMM-specific visual language, bilingual mirroring, and closeout-oriented consolidation.

The current visual risk is concentrated in technical figures. Most diagrams are Mermaid-first and tuned through runtime theme re-initialization, while SVG/icon assets remain sparse. That keeps iteration easy, but it also makes the site look less curated and causes weak light/dark mode parity. The redesign therefore needs both structural and visual changes: route architecture, reusable presentation primitives, and a theme-safe figure strategy.

## Goals / Non-Goals

**Goals:**

- Rebuild the Pages site into a bilingual whitepaper-and-academy experience with a clearer progression from project thesis to deep kernel study.
- Align the docs engineering baseline with the `kimi-cli` VitePress setup where that improves config clarity, navigation consistency, and future maintainability.
- Introduce a production-ready visual system for cards, metrics, figures, citations, and diagrams that remains legible in both light and dark themes.
- Strengthen technical depth through new or expanded modules for architectural walkthroughs, related projects, references, and evolution analysis.
- Keep README, README.zh-CN, and Pages entry routes aligned so repository visitors see one consistent public narrative.

**Non-Goals:**

- No changes to CUDA kernel logic, benchmark harnesses, validation tolerances, or GPU runtime behavior.
- No hosted GPU execution changes, workflow topology expansion, or new deployment platform.
- No obligation to preserve the current route hierarchy or internal content grouping when a cleaner information architecture is better.
- No attempt to mimic `kimi-cli` branding or content; only its engineering posture and documentation-site baseline are reused.

## Decisions

### Decision 1: Adopt the `kimi-cli` docs baseline for configuration and theme entry

- **Choice**: Keep VitePress and the current plugin family, but refactor `docs/package.json`, `.vitepress/config.ts`, and `.vitepress/theme/index.ts` toward the cleaner `kimi-cli` baseline: centralized locale definitions, simple theme extension, and minimal runtime logic in the theme bootstrap.
- **Why**: The current stack already shares the same major dependencies, so the highest-return change is not a framework swap but a convergence toward a cleaner baseline that is easier to reason about during closeout.
- **Alternative considered**: Preserve the current docs configuration and only rewrite page content.
- **Rejected because**: The existing route layout, runtime Mermaid theme handling, and ad hoc theme responsibilities would continue to constrain the redesign and make future consolidation harder.

### Decision 2: Replace the current route taxonomy with a whitepaper-and-academy map

- **Choice**: Reorganize bilingual docs into mirrored route groups with a clearer reader journey: overview/orientation, architecture, academy, validation, and research.
- **Why**: The target audience is not browsing randomly; it is evaluating the project as an engineering artifact. The route map therefore needs to behave like a curated reading sequence rather than a loose collection of topical pages.
- **Alternative considered**: Keep the existing `architecture`, `methodology`, `resources`, `validation`, and `support` groups and only improve page copy.
- **Rejected because**: That model is informative but not forceful enough for a whitepaper-grade presentation, and it hides the “academy” progression the user explicitly wants.

### Decision 3: Use a theme-safe figure system instead of Mermaid-only presentation

- **Choice**: Reserve Mermaid for simple process diagrams, but move homepage-grade and architecture-grade figures to curated SVG assets and structured figure wrappers that use explicit design tokens for light/dark mode.
- **Why**: High-signal diagrams need better composition control, better typography, and deterministic theme behavior. SVG assets also solve the current “dark mode unreadable / light mode washed out” failure mode more directly than repeated runtime Mermaid reconfiguration.
- **Alternative considered**: Keep all diagrams in Mermaid and continue improving theme variables.
- **Rejected because**: Mermaid is efficient for documentation flowcharts, but it is not the right default for polished whitepaper hero figures, comparison diagrams, or dual-theme visual art direction.

### Decision 4: Expand public content with explicit citation, comparison, and evolution surfaces

- **Choice**: Add or strengthen pages for references, related open-source projects, and architectural evolution analysis, and connect them directly from the main navigation.
- **Why**: Whitepaper-grade credibility depends on traceable lineage. Advanced readers and interviewers need to see not only what the project implements, but also what ideas it inherits, where it differs, and how its design choices evolved.
- **Alternative considered**: Keep a single references page as a terminal appendix.
- **Rejected because**: A lone references appendix is easy to ignore and does not create the academic depth or comparative framing required by the target positioning.

### Decision 5: Treat README as the executive summary and Pages as the deep document

- **Choice**: Keep `README.md` and `README.zh-CN.md` concise, evidence-first, and navigational, while moving the long-form architecture and academy narrative into Pages.
- **Why**: This preserves distinct roles across entry surfaces and avoids duplicating long-form content across GitHub and Pages.
- **Alternative considered**: Promote the README into a near-full whitepaper mirror.
- **Rejected because**: That would violate the repository’s consolidation posture and make bilingual maintenance harder.

## Risks / Trade-offs

- **[Risk]** Aggressive route restructuring can create broken internal links or route-test drift.  
  **Mitigation**: Update route tests together with nav/sidebar changes and run the docs build after each large content batch.

- **[Risk]** A larger visual system can become brittle if it relies on page-specific styling.  
  **Mitigation**: Build around shared tokens, a small reusable component set, and a documented figure/card vocabulary.

- **[Risk]** Bilingual parity can regress when pages are renamed or split.  
  **Mitigation**: Mirror route trees exactly and move English/Chinese pages in paired edits.

- **[Risk]** Replacing diagrams with SVGs can create manual-asset maintenance overhead.  
  **Mitigation**: Limit custom SVG work to high-value figures and keep diagram subjects stable and reusable across pages.

- **[Risk]** The site may become more ambitious than the repository’s closeout posture allows.  
  **Mitigation**: Prefer consolidation and strong entry pages over expanding into a large number of low-value new leaves.

## Migration Plan

1. Refactor docs configuration and route tests to the new mirrored navigation model.
2. Introduce the shared visual primitives and theme-safe figure assets required by the new homepage and architecture surfaces.
3. Migrate content into the new bilingual route map, rewriting pages in pairs and removing superseded structures.
4. Align README entry surfaces and public metadata-facing copy with the new Pages narrative.
5. Validate docs tests, docs build, and OpenSpec consistency before finalizing the change.

Rollback strategy: revert the docs route map, theme changes, and content moves together so the site returns to the previous route architecture in one step.

## Open Questions

- None blocking. The user’s preference for an aggressive, non-backward-compatible rebuild resolves the main scope ambiguity.
