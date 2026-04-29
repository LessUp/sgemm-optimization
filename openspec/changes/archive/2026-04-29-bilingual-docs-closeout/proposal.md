# Proposal: Bilingual Documentation and Pages Closeout

## Summary

Turn the repository into an archive-ready bilingual project surface by refreshing both READMEs, adding a real English/Chinese switch to the GitHub Pages site, expanding the tutorial experience, and auditing overall project completion.

## Motivation

The repository has reached a stable implementation state. All kernel implementations are complete, tests pass, and OpenSpec validation succeeds. However, the public-facing documentation surfaces are not yet ready for archival:

1. **README parity gap**: English and Chinese READMEs exist but have drifted in structure and content
2. **No language switching**: GitHub Pages lacks a functional way to switch between English and Chinese
3. **Missing Chinese tutorials**: The docs/ directory has detailed English tutorials but no Chinese counterparts
4. **Incomplete public navigation**: specs.md and the public-facing spec landing need bilingual exposure

## Deliverables

### 1. README Refresh and Parity

- Restructure both READMEs to share the same outline
- Add explicit "where to start" navigation tables
- Align badges, links, and terminology across languages
- Update GitHub repository metadata to match bilingual positioning

### 2. GitHub Pages Theme/Navigation Improvements

- Make homepage bilingual-aware with hero, CTA, and learning entry cards
- Polish theme details (hero spacing, CTA hierarchy, table/card contrast)
- Preserve existing dark/light mode support

### 3. English/Chinese Switching for Public Teaching Surfaces

- Define page-pair metadata model (`lang`, `page_key`, `lang_ref`)
- Create lightweight switcher include for sidebar/footer
- Implement route switching with preference persistence
- Style switcher to match NVIDIA-inspired theme

### 4. Repository Completion Audit and Closeout Polish

- Audit CHANGELOG.md, CONTRIBUTING.md for bilingual exposure needs
- Verify workflow triggers cover new file layout
- Record and resolve remaining presentation gaps
- Archive the OpenSpec change when truly complete

## Scope

### In Scope

- Root README pair (`README.md`, `README.zh-CN.md`)
- GitHub Pages home (`index.md`, `zh/index.md`)
- All tutorial pages (`docs/*.md`, `zh/docs/*.md`)
- Public specs landing (`specs.md`, `zh/specs.md`)
- Language switcher infrastructure
- GitHub repository metadata

### Out of Scope

- Full mirror of normative OpenSpec specs (governance files remain English-only)
- Separate site deployments or complex frontend frameworks
- New kernel implementations or code changes

## Success Criteria

1. OpenSpec validation passes for all stable specs plus the new change
2. Both READMEs share the same structure with aligned content
3. GitHub Pages has functional language switching on all public pages
4. Every English tutorial page has a Chinese counterpart
5. GitHub metadata matches the refreshed positioning
6. Repository feels complete as a public teaching project

## Dependencies

- Existing `just-the-docs` theme and custom SCSS
- OpenSpec validation tooling
- GitHub Actions Pages workflow
- `gh` CLI for metadata updates
