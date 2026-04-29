# Tasks: Bilingual Documentation and Pages Closeout

## Task 1: OpenSpec Change and Completion Baseline

- [x] Create change directory: `openspec/changes/bilingual-docs-closeout/`
- [x] Write proposal.md covering four deliverables
- [x] Write design.md with route model and component specs
- [x] Add project-presentation delta scenarios
- [x] Run `openspec validate --all` to verify

## Task 2: Bilingual Information Architecture and Switcher Contract

- [x] Add page-pair metadata (`lang`, `page_key`, `lang_ref`) to index.md and specs.md
- [x] Update `_config.yml` for bilingual labels and navigation
- [x] Create `_includes/nav_footer_custom.html` with switcher markup
- [x] Create `assets/js/language-toggle.js` for route switching
- [x] Add switcher styles to `_sass/custom/custom.scss`
- [x] Create `_includes/head_custom.html` to load JS

## Task 3: README Pair Refresh and Repository Metadata Alignment

- [x] Restructure both READMEs to identical outline
- [x] Add "where to start" navigation tables
- [x] Align badges, links, and terminology
- [x] Update GitHub repository metadata via `gh repo edit`

## Task 4: Pages Home and Theme Refresh

- [x] Update index.md with bilingual-aware hero and CTAs
- [x] Create zh/index.md as Chinese counterpart
- [x] Polish theme details (spacing, hierarchy, contrast)
- [x] Verify dark/light mode support preserved

## Task 5: Mirror and Enrich Tutorial Set

- [x] Normalize front matter for all docs/*.md pages
- [x] Expand onboarding pages (getting-started, learning-path, architecture)
- [x] Add reader guidance to kernel pages
- [x] Create all zh/docs/*.md counterparts with content parity
- [x] Create optional pages only if they close real gaps (not needed)

## Task 6: Public Spec/Navigation Surfaces and Closeout Audit

- [x] Create zh/specs.md as bilingual specs landing
- [x] Audit CHANGELOG.md and CONTRIBUTING.md for bilingual needs (decided: keep as-is, low-frequency files)
- [x] Verify workflow triggers cover new file layout
- [x] Record and resolve remaining closeout gaps

## Task 7: Validation and Final Polish

- [x] Run `openspec validate --all` — PASSED
- [x] Run available repository baseline checks (OpenSpec passed; CUDA build/runtime blocked in this environment because `nvcc` is unavailable)
- [x] Review published-surface checklist
- [x] Archive OpenSpec change when complete

---

## Progress Tracking

| Task | Status | Notes |
|------|--------|-------|
| 1. OpenSpec Change | ✅ complete | |
| 2. IA & Switcher | ✅ complete | |
| 3. README Refresh | ✅ complete | |
| 4. Pages Home | ✅ complete | |
| 5. Tutorial Mirror | ✅ complete | |
| 6. Specs & Audit | ✅ complete | |
| 7. Validation | ✅ complete | OpenSpec validated; CUDA tests require GPU env |

---

## Remaining Closeout Items

After merge to master:
1. Verify Pages build succeeds on CI
2. Test language switcher on deployed site

## Files Created/Modified Summary

### New Files
- `_includes/head_custom.html`
- `_includes/nav_footer_custom.html`
- `assets/js/language-toggle.js`
- `zh/index.md`
- `zh/specs.md`
- `zh/docs/getting-started.md`
- `zh/docs/learning-path.md`
- `zh/docs/architecture.md`
- `zh/docs/benchmark-results.md`
- `zh/docs/kernel-naive.md`
- `zh/docs/kernel-tiled.md`
- `zh/docs/kernel-bank-free.md`
- `zh/docs/kernel-double-buffer.md`
- `zh/docs/kernel-tensor-core.md`

### Modified Files
- `README.md` — restructured with unified outline
- `README.zh-CN.md` — restructured with unified outline
- `_config.yml` — bilingual description and aux_links
- `_sass/custom/custom.scss` — language switcher styles
- `index.md` — page-pair metadata
- `specs.md` — page-pair metadata
- `docs/*.md` (9 files) — page-pair metadata
- `.github/workflows/pages.yml` — added zh/** and _includes/** triggers
