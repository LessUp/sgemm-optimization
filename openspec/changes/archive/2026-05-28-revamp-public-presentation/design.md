## Context

The repository is in closeout mode, which means public quality now depends more on coherence than on adding new kernel features. Current docs already cover the technical ladder, but entry surfaces are still distributed across generic pages that do not explicitly answer two high-value questions:

1. Why this project is different from many SGEMM learning repos.
2. How a candidate should present this work in interview settings with evidence.

At the same time, OpenSpec requires bilingual mirroring and single-source responsibilities, so the redesign must improve impact without introducing duplicated, drifting documentation.

## Goals / Non-Goals

**Goals:**

- Build an interview-grade public narrative that starts from proof (benchmarks, verification, constraints) rather than slogans.
- Introduce explicit citation and repository-reference structure so readers can trace each major claim to authoritative sources.
- Keep English/Chinese pages mirrored by structure and intent, while allowing language-native wording.
- Keep visual style recognizably technical with light NVIDIA-inspired cues (green accents, structured cards, metric blocks) without over-branding.
- Align README and Pages so users get one consistent story regardless of entry point.

**Non-Goals:**

- No changes to CUDA kernels, benchmark logic, or test tolerances.
- No replacement of VitePress stack or workflow topology.
- No attempt to clone NVIDIA brand assets, typography, or proprietary visual materials.
- No broad governance rewrite outside project-presentation deltas required by this change.

## Decisions

### Decision 1: Add three presentation rails as first-class pages

- **Choice**: Add `project-highlights`, `interview-playbook`, and `references` in both languages.
- **Why**: Existing pages teach optimization steps well but do not package the project as a differentiated engineering artifact.
- **Alternative considered**: Embedding all content into home page sections.
- **Rejected because**: Home page would become too long, and high-value interview/reference content would be harder to discover and maintain.

### Decision 2: Keep mirrored bilingual architecture at the navigation level

- **Choice**: Update navbar/sidebar so EN and ZH contain one-to-one page counterparts.
- **Why**: This directly satisfies the stable bilingual requirements and prevents orphaned content.
- **Alternative considered**: EN-first structure with selective ZH translation.
- **Rejected because**: Violates current project-presentation expectations for mirrored public surfaces.

### Decision 3: Use “evidence-first” storytelling blocks on homepage

- **Choice**: Home page structure emphasizes benchmark context, correctness boundary, and optimization progression.
- **Why**: Interviewers and advanced readers trust evidence framing more than generic feature claims.
- **Alternative considered**: Marketing-style hero with minimal technical details.
- **Rejected because**: Undermines the project’s engineering credibility.

### Decision 4: Lightly evolve existing theme rather than replace it

- **Choice**: Refine current CSS variable system and components; introduce stronger hierarchy, card polish, and mobile-friendly readability.
- **Why**: Lower risk and faster convergence while preserving existing VitePress compatibility.
- **Alternative considered**: Full theme rewrite with custom Vue components.
- **Rejected because**: Higher maintenance and unnecessary complexity for closeout-stage goals.

## Risks / Trade-offs

- **[Risk]** More pages can increase maintenance burden for bilingual parity.  
  **Mitigation**: Keep each page scoped with single responsibility and mirrored sections.

- **[Risk]** Stronger visual styling might reduce content density.  
  **Mitigation**: Keep typography conservative in article pages; concentrate visual emphasis on home and summary cards.

- **[Risk]** Spec delta and implementation may drift if done in separate passes.  
  **Mitigation**: Implement documentation and spec delta in one apply session, then run `openspec validate --all`.

- **[Risk]** External reference links may break over time.  
  **Mitigation**: Prefer canonical URLs (NVIDIA docs, official papers, mature repos) and keep links grouped for periodic audits.

## Migration Plan

1. Update VitePress config for new nav/sidebar architecture.
2. Add mirrored EN/ZH pages (highlights/interview/references).
3. Rewrite EN/ZH home pages to evidence-first layout.
4. Refresh theme styles for light NVIDIA-inspired visual consistency.
5. Update README and README.zh-CN to match new entry narrative and corrected internal links.
6. Add OpenSpec project-presentation delta for new normative expectations.
7. Validate with docs build and OpenSpec checks before final handoff.

Rollback strategy: revert documentation and OpenSpec delta files in a single commit if build/validation cannot be stabilized.

## Open Questions

- None blocking. The requested style level (“light inspiration”) and scope (“whole-repo aligned redesign”) are explicit.
