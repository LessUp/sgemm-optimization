## 1. Information Architecture and Theme Baseline

- [x] 1.1 Update VitePress locale nav and sidebar to include mirrored EN/ZH routes for highlights, interview playbook, and references.
- [x] 1.2 Refresh shared theme styling with light NVIDIA-inspired visual hierarchy while preserving readability and mobile usability.
- [x] 1.3 Verify that all newly linked routes exist in both language trees before content rewrites.

## 2. Bilingual Content Rewrite and New Public Pages

- [x] 2.1 Rewrite `docs/en/index.md` and `docs/zh/index.md` to an evidence-first home layout with benchmark, correctness, and learning-path entry blocks.
- [x] 2.2 Add mirrored `project-highlights` pages in English and Chinese with differentiated project strengths and engineering conventions.
- [x] 2.3 Add mirrored `interview-playbook` pages in English and Chinese with presentation storyline, deep-dive questions, and answer framing.
- [x] 2.4 Add mirrored `references` pages in English and Chinese with structured paper/doc/repo citations tied to project decisions.

## 3. Repository Entry Alignment

- [x] 3.1 Rewrite `README.md` as an entry surface aligned with the new Pages architecture and corrected internal links.
- [x] 3.2 Rewrite `README.zh-CN.md` with the same structure, mirrored navigation, and corrected Chinese link targets.
- [x] 3.3 Ensure README and Pages claims use consistent wording for validation boundaries and benchmark scope.

## 4. Validation and OpenSpec Consistency

- [x] 4.1 Run `npm --prefix docs run build` and fix broken links, missing pages, or front-matter inconsistencies.
- [x] 4.2 Run `openspec validate --all` and resolve any spec-format or delta-merge issues.
- [x] 4.3 Mark all completed tasks and produce a concise change summary with affected files and verification outputs.
