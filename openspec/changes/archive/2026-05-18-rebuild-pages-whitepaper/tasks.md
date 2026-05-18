## 1. Docs infrastructure realignment

- [x] 1.1 Align `docs/package.json`, `docs/.vitepress/config.ts`, and `docs/.vitepress/theme/index.ts` with the cleaner `kimi-cli` VitePress baseline while preserving SGEMM-specific plugins and deployment base handling
- [x] 1.2 Replace the current public route taxonomy with a mirrored bilingual whitepaper-and-academy navigation model and update any route tests that encode the old structure
- [x] 1.3 Introduce the shared presentation primitives, tokens, and stylesheet structure needed for the rebuilt home, architecture, academy, validation, and research surfaces

## 2. Theme-safe figures and assets

- [x] 2.1 Create curated SVG or equivalent theme-safe figure assets for homepage-grade and architecture-grade visuals
- [x] 2.2 Replace fragile light/dark rendering behavior for technical visuals so diagrams, labels, and icons remain legible in both themes
- [x] 2.3 Apply the new figure system to the highest-signal pages before removing superseded visual treatments

## 3. Content architecture rebuild

- [x] 3.1 Rewrite the bilingual home pages to present the project as an evidence-backed whitepaper entry instead of a generic documentation landing page
- [x] 3.2 Reorganize existing architecture, methodology, learning, and validation content into the new mirrored route groups and add any missing academy transition pages
- [x] 3.3 Add or expand bilingual research-oriented pages for references, related open-source projects, and architectural evolution analysis
- [x] 3.4 Remove or merge superseded pages and update internal cross-links so the retained document set has one clear responsibility per page

## 4. Repository entry alignment and validation

- [x] 4.1 Rewrite `README.md` and `README.zh-CN.md` so they operate as concise executive summaries aligned with the rebuilt Pages structure
- [x] 4.2 Run docs route tests, docs build, and `openspec validate --all`, fixing any broken links, route drift, or spec inconsistencies
- [x] 4.3 Review the finished change for bilingual parity, closeout-stage consolidation, and public narrative coherence before handoff
