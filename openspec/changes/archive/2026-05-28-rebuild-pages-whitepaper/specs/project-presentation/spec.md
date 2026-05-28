## MODIFIED Requirements

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
All high-signal public technical figures MUST remain readable, visually coherent, and semantically equivalent in both light mode and dark mode, including when the visitor changes theme from the site’s manual theme switcher.

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
