# Project Presentation Specification

> **Version**: 1.2.0 | **Last Updated**: 2026-05-28 | **Status**: Complete

## Purpose

Define how the repository presents the project through README, GitHub Pages, and public repository metadata so each entry surface stays distinct, current, and aligned with the SGEMM optimization focus.

## Requirements

### Requirement: Repository entry points have distinct roles
The project MUST present itself through distinct entry surfaces so that each one serves a clear audience and does not duplicate another surface unnecessarily.

#### Scenario: New visitor lands on the repository or project site
- **WHEN** a user opens `README.md` or the GitHub Pages site
- **THEN** the repository MUST provide a concise explanation of the project's value, core technical highlights, and clear navigation to deeper documentation without mirroring the same long-form content across both surfaces
- **AND** the Pages home experience MUST behave as the long-form whitepaper entry point rather than as a second copy of the README
- **AND** the Pages home experience MUST prioritize quick orientation, evidence summary, and clear paths to detailed technical pages

### Requirement: Documentation inventory is intentionally consolidated
The project MUST aggressively merge or remove stale, low-value, or overlapping documentation so that each retained document has a single clear responsibility.

#### Scenario: Maintainer audits repository documents
- **WHEN** a maintainer reviews root docs, Pages content, and supporting documentation
- **THEN** duplicate release-history material, overlapping landing pages, and non-authoritative explanatory files MUST be merged or removed until each retained document has a clear purpose and ownership
- **AND** the public Pages structure MUST avoid parallel sections that explain the same concept with only cosmetic differences

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
- **AND** both files MUST link to equivalent English and Chinese documentation routes on the published site

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
Every public academy or tutorial page in English MUST have a Chinese counterpart with equivalent content structure.

#### Scenario: Academy Pages Stay Mirrored
- **WHEN** a public learning or deep-dive page exists in English under the published documentation tree
- **THEN** a Chinese counterpart MUST exist at the mirrored route
- **AND** both pages MUST have equivalent technical content structure
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
- **AND** every linked English public route that belongs to the mirrored documentation tree MUST have a Chinese counterpart and vice versa

### Requirement: Public whitepaper information architecture is curated by reader depth
The GitHub Pages site MUST present the project through a mirrored bilingual information architecture that guides readers from orientation to system design, optimization study, evidence review, and research context instead of exposing a loosely grouped set of topical pages.

#### Scenario: Reader lands on the home page without repository context
- **WHEN** a visitor opens the English or Chinese home page
- **THEN** the page MUST summarize the project thesis, system blueprint, optimization ladder, and evidence model
- **AND** the page MUST provide direct paths into deeper guide, architecture, academy, validation, and research sections

#### Scenario: Reader follows the intended reading spine
- **WHEN** a visitor uses top-level navigation, sidebars, or section landing pages
- **THEN** the documentation structure MUST separate orientation, system architecture, algorithm-study surfaces, evidence framing, and reference context into distinct mirrored route groups
- **AND** each section landing page MUST clarify who it is for, what it answers, and what page should be opened next

### Requirement: Technical figures remain legible across themes
All high-signal public technical figures MUST remain readable, visually coherent, and semantically equivalent in both light mode and dark mode, including when the visitor changes theme from the site's manual theme switcher.

#### Scenario: Visitor switches between light and dark mode
- **WHEN** the site theme changes through system preference or the VitePress theme toggle
- **THEN** diagrams, SVG illustrations, icons, and figure labels MUST preserve sufficient contrast for text, boundaries, and emphasis
- **AND** the rendered figure variant MUST follow the active site theme instead of relying only on browser-level `prefers-color-scheme`

#### Scenario: Architecture page uses a curated figure
- **WHEN** a page renders a custom architecture or whitepaper figure
- **THEN** the figure MUST use a shared site component or equivalent shared convention for paired theme-safe assets
- **AND** the figure MUST remain understandable without assuming one background color or one contrast profile

### Requirement: Public research context is explicit and traceable
The documentation site MUST provide academic-style context that links project claims to references, related open-source implementations, comparative reading paths, and architectural evolution notes.

#### Scenario: Reader investigates technical lineage
- **WHEN** a reader opens the research-oriented section from site navigation
- **THEN** the site MUST include structured references grouped by purpose
- **AND** the site MUST include at least one related-project or comparative-analysis surface
- **AND** the site MUST include at least one evolution-oriented explanation of why the project architecture is organized as it is

#### Scenario: Reader wants a guided reference path
- **WHEN** a reader approaches the research section from the home page or a section index
- **THEN** the site MUST provide a curated path that distinguishes canonical references, comparative open-source context, and further reading routes
