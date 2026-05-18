## Context

The repository already has a bilingual VitePress site with clear technical intent, but the current presentation still relies on an incremental layer of cards, ad hoc figure embedding, and route names inherited from earlier iterations. Two issues are coupled: the information architecture does not yet present the public story as a tightly defended whitepaper, and the high-signal SVG figures rely on `prefers-color-scheme` asset swapping, which can drift from the site’s manual theme toggle.

The rebuild is intentionally aggressive. Backward compatibility is not a design goal. The project posture favors long-term clarity, one authoritative public story, and stronger differentiation over preserving previous route naming or visual patterns.

## Goals / Non-Goals

**Goals:**

- Make the Pages home and section indexes read like a coherent technical whitepaper for advanced readers.
- Rebuild the visual system so the site feels deliberate, high-signal, and theme-safe in both light and dark mode.
- Introduce a shared figure component that swaps paired SVG assets from VitePress theme state instead of browser media preference alone.
- Expand the public content with deeper system-blueprint, performance-model, and reference-oriented wayfinding pages in both languages.
- Encode the new IA and figure rules in docs tests so future edits do not regress silently.

**Non-Goals:**

- Rewriting the CUDA kernel implementation or benchmark engine itself.
- Adding a second docs framework, client-side visualization stack, or hosted runtime dependency.
- Preserving old visual patterns, route names, or page emphasis when they conflict with the rebuilt whitepaper direction.

## Decisions

### 1. Keep the existing VitePress stack and rebuild inside it

The site already has mirrored locales, tested routing, and a working GitHub Pages pipeline. Replacing the framework would create migration noise without improving the core problem. The better trade-off is to keep VitePress and rebuild the theme, shared components, and curated content surfaces inside the existing pipeline.

**Alternative considered:** migrate to a different static-site framework. Rejected because the current need is stronger IA and design discipline, not framework capability.

### 2. Replace `<picture media="prefers-color-scheme">` with a shared theme-aware figure component

The current paired SVG approach is valid, but the switching mechanism is wrong for a manual site theme toggle. A shared component registered in the VitePress theme can read `isDark` from site state and select the correct asset explicitly. This keeps the paired-asset model, which is good for high-craft SVGs, while fixing the manual-toggle bug.

**Alternative considered:** merge both themes into a single SVG that depends on CSS inheritance. Rejected because the current figures are curated illustrations, and paired assets give more control over palette and contrast.

### 3. Reframe the public IA as a whitepaper reading spine, not a loose doc tree

The top-level navigation and section landing pages will guide readers through five jobs: orientation, system design, algorithm ladder, evidence boundary, and reference context. Existing deep-dive pages remain valuable, but section indexes become stronger editorial surfaces that route readers by intent and depth.

**Alternative considered:** keep current section emphasis and only polish the homepage. Rejected because the weakness is structural, not only cosmetic.

### 4. Prefer a technical publication visual language over generic doc-card patterns

The redesign will use stronger contrast, more deliberate spacing rhythm, shared whitepaper panels, and fewer template-like cards or side-stripe callouts. This aligns the site with a “lab-grade whitepaper” voice and avoids the current middle ground between docs theme defaults and a custom publication surface.

**Alternative considered:** stay close to the default VitePress look with light token tweaks. Rejected because it would not produce the step-change the public positioning now needs.

### 5. Treat docs tests as structural contracts for the public story

The existing Node test already guards route conventions. The rebuild will extend it to assert mirrored new routes, shared figure component usage, and the absence of theme-fragile `<picture>` usage on curated whitepaper figures. That gives a low-cost CI-safe contract for presentation integrity.

**Alternative considered:** rely only on manual preview checks. Rejected because the site is now part of the public technical claim and needs repeatable safeguards.

## Risks / Trade-offs

- **Large docs diff** → Mitigation: keep all changes inside the existing docs stack, limit new primitives to a small shared component set, and mirror EN/ZH changes together.
- **Route and nav churn** → Mitigation: update config, section indexes, and tests in the same pass so the new IA is internally consistent.
- **Visual overreach could harm readability** → Mitigation: keep prose measure conservative, use paired SVG assets for contrast control, and preserve semantic VitePress content flows.
- **Bilingual drift** → Mitigation: add mirrored pages in the same change and keep sidebars symmetric across locales.
