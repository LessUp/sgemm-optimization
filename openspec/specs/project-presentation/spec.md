# Project Presentation Specification

> **Version**: 1.0.0 | **Last Updated**: 2026-04-23 | **Status**: Complete

## Purpose

Define how the repository presents the project through README, GitHub Pages, and public repository metadata so each entry surface stays distinct, current, and aligned with the SGEMM optimization focus.

## Requirements

### Requirement: Repository entry points have distinct roles
The project MUST present itself through distinct entry surfaces so that each one serves a clear audience and does not duplicate another surface unnecessarily.

#### Scenario: New visitor lands on the repository or project site
- **WHEN** a user opens `README.md` or the GitHub Pages site
- **THEN** the repository MUST provide a concise explanation of the project's value, core technical highlights, and clear navigation to deeper documentation without mirroring the same long-form content across both surfaces

### Requirement: Documentation inventory is intentionally consolidated
The project MUST aggressively merge or remove stale, low-value, or overlapping documentation so that each retained document has a single clear responsibility.

#### Scenario: Maintainer audits repository documents
- **WHEN** a maintainer reviews root docs, Pages content, and supporting documentation
- **THEN** duplicate release-history material, overlapping landing pages, and non-authoritative explanatory files MUST be merged or removed until each retained document has a clear purpose and ownership

### Requirement: GitHub metadata matches project positioning
The public repository metadata MUST reinforce the same positioning used by the cleaned documentation and Pages experience.

#### Scenario: Visitor views the GitHub repository summary
- **WHEN** a visitor sees the repository About panel, homepage URL, or topic tags
- **THEN** the description, homepage, and topics MUST accurately reflect the project's CUDA SGEMM optimization focus and link users toward the maintained project entry points

### Requirement: Repository entry surfaces support bilingual navigation
The repository MUST provide clear entry points for both English and Chinese readers with consistent structure across both README files.

#### Scenario: Bilingual Repository Entry
- **WHEN** a visitor lands on the GitHub repository root
- **THEN** the repository MUST present `README.md` as the default English entry
- **AND** the repository MUST show a prominent link to `README.zh-CN.md` for Chinese readers
- **AND** both files MUST have consistent structure and navigation tables

### Requirement: All public pages offer language switching
Every public documentation page MUST provide a way to switch to its counterpart in the other language.

#### Scenario: Language Switching Between Paired Pages
- **WHEN** a visitor reading a tutorial page in English clicks the language switcher for Chinese
- **THEN** the system MUST redirect to the Chinese counterpart page
- **AND** the Chinese page MUST offer a switch back to English

#### Scenario: Switcher Mobile Accessible
- **WHEN** a visitor on a mobile device opens the sidebar
- **THEN** the language switcher MUST be visible and functional
- **AND** buttons MUST be large enough for touch interaction

### Requirement: Tutorial pages have mirrored counterparts
Every tutorial page in English MUST have a Chinese counterpart with equivalent content structure.

#### Scenario: Tutorial Pages Stay Mirrored
- **WHEN** a tutorial page exists in English at `/docs/kernel-naive/` with a Chinese counterpart at `/zh/docs/kernel-naive/`
- **THEN** both pages MUST have equivalent technical content
- **AND** both pages MUST have cross-references to the same related pages
- **AND** both pages MUST have matching front matter except for language-pair metadata

### Requirement: Workflow triggers cover bilingual public surfaces
The GitHub Pages workflow MUST rebuild when public bilingual content changes.

#### Scenario: Workflow Triggers Updated
- **WHEN** any `docs/**`, `zh/**`, `_includes/**`, `_sass/**`, `assets/**`, root landing page, or specs landing file changes
- **THEN** the GitHub Pages workflow MUST trigger a rebuild
- **AND** the published site MUST reflect changes

### Requirement: No orphaned public content exists after closeout
All linked public content MUST resolve correctly with no missing counterparts.

#### Scenario: No Orphaned Content
- **WHEN** someone browses the closed-out site
- **THEN** every linked public page MUST exist
- **AND** no page MUST reference a missing counterpart
- **AND** all cross-references MUST resolve correctly
