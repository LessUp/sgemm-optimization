## ADDED Requirements

### Requirement: Public project narrative is interview-ready and evidence-backed
Public entry surfaces MUST communicate the project as an engineering case study that can be defended in technical interviews with concrete evidence.

#### Scenario: Interviewer scans repository entry quickly
- **WHEN** an interviewer reads `README.md` or `README.zh-CN.md`
- **THEN** the entry MUST summarize the optimization ladder, correctness boundary, and validation model in concise technical language
- **AND** the entry MUST provide direct links to Pages sections that contain deeper proof and implementation context

#### Scenario: Visitor asks how claims are substantiated
- **WHEN** a visitor follows public documentation links from README or home pages
- **THEN** the linked pages MUST include benchmark framing, verification constraints, and explicit scope notes
- **AND** performance claims MUST avoid implying universal guarantees across all hardware and workloads

### Requirement: Public references are structured and traceable
The documentation site MUST provide a dedicated references surface that maps project ideas to authoritative papers, official documentation, and high-signal open-source repositories.

#### Scenario: Reader validates technical lineage
- **WHEN** a reader opens the references page from navigation
- **THEN** references MUST be grouped by purpose (for example: architecture background, kernel optimization practice, Tensor Core programming)
- **AND** each group MUST include at least one canonical source URL
- **AND** references MUST be presented in both English and Chinese public surfaces

## MODIFIED Requirements

### Requirement: Repository entry points have distinct roles
The project MUST present itself through distinct entry surfaces so that each one serves a clear audience and does not duplicate another surface unnecessarily.

#### Scenario: New visitor lands on the repository or project site
- **WHEN** a user opens `README.md` or the GitHub Pages site
- **THEN** the repository MUST provide a concise explanation of the project's value, core technical highlights, and clear navigation to deeper documentation without mirroring the same long-form content across both surfaces
- **AND** the Pages home experience MUST prioritize quick orientation, evidence summary, and clear paths to detailed technical pages

### Requirement: Repository entry surfaces support bilingual navigation
The repository MUST provide clear entry points for both English and Chinese readers with consistent structure across both README files.

#### Scenario: Bilingual Repository Entry
- **WHEN** a visitor lands on the GitHub repository root
- **THEN** the repository MUST present `README.md` as the default English entry
- **AND** the repository MUST show a prominent link to `README.zh-CN.md` for Chinese readers
- **AND** both files MUST have consistent structure and navigation tables
- **AND** both files MUST link to equivalent English and Chinese documentation routes on the published site

### Requirement: Tutorial pages have mirrored counterparts
Every tutorial page in English MUST have a Chinese counterpart with equivalent content structure.

#### Scenario: Tutorial Pages Stay Mirrored
- **WHEN** a tutorial page exists in English at `/en/kernel-naive` with a Chinese counterpart at `/zh/kernel-naive`
- **THEN** both pages MUST have equivalent technical content
- **AND** both pages MUST have cross-references to the same related pages
- **AND** both pages MUST have matching front matter except for language-pair metadata

### Requirement: Workflow triggers cover bilingual public surfaces
The GitHub Pages workflow MUST rebuild when public bilingual content changes.

#### Scenario: Workflow Triggers Updated
- **WHEN** any `docs/**`, `zh/**`, `_includes/**`, `_sass/**`, `assets/**`, root landing page, or specs landing file changes
- **THEN** the GitHub Pages workflow MUST trigger a rebuild
- **AND** the published site MUST reflect changes
