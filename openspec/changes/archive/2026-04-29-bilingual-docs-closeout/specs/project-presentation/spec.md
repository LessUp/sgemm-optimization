# Spec: Project Presentation (Delta)

> This spec captures the bilingual presentation requirements for the closeout phase.

## ADDED Requirements

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

#### Scenario: Switcher Remembers Preference

- **WHEN** a visitor who previously switched to Chinese visits another English page
- **THEN** the switcher MUST still show Chinese as active
- **AND** clicking the switcher MUST return them to Chinese content

#### Scenario: Switcher Mobile Accessible

- **WHEN** a visitor on a mobile device opens the sidebar
- **THEN** the language switcher MUST be visible and functional
- **AND** buttons MUST be large enough for touch interaction

### Requirement: Tutorial pages have mirrored counterparts

Every tutorial page in English MUST have a Chinese counterpart with equivalent content structure.

#### Scenario: Tutorial Pages Stay Mirrored

- **WHEN** a tutorial page exists in English at `/docs/kernel-naive/` with a Chinese counterpart at `/zh/docs/kernel-naive/`
- **THEN** both pages MUST have identical section structure
- **AND** both pages MUST have equivalent technical content
- **AND** both pages MUST have cross-references to the same related pages
- **AND** both pages MUST have matching front matter (except `lang`, `page_key`, `lang_ref`)

### Requirement: GitHub metadata aligns with bilingual positioning

Repository metadata on GitHub MUST reflect the bilingual nature of the project.

#### Scenario: Public Metadata Stays Aligned

- **WHEN** the READMEs are updated with bilingual content
- **THEN** the GitHub description and homepage MUST match the bilingual positioning
- **AND** topics MUST remain relevant to both language audiences

### Requirement: All public surfaces are reviewed before archive

The closeout audit MUST verify all bilingual surfaces are complete and consistent.

#### Scenario: All Public Surfaces Reviewed

- **WHEN** the closeout audit runs after bilingual implementation is complete
- **THEN** README pair MUST be verified
- **AND** Pages home pair MUST be verified
- **AND** all tutorial page pairs MUST be verified
- **AND** specs landing pair MUST be verified
- **AND** GitHub metadata MUST be verified

### Requirement: Workflow triggers cover new file layout

The GitHub Pages workflow MUST rebuild when any bilingual content changes.

#### Scenario: Workflow Triggers Updated

- **WHEN** any `zh/**/*.md` file changes
- **THEN** the GitHub Pages workflow MUST trigger a rebuild
- **AND** the published site MUST reflect changes

### Requirement: No orphaned content exists after archive

All linked content MUST resolve correctly with no missing counterparts.

#### Scenario: No Orphaned Content

- **WHEN** someone browses the archived site
- **THEN** every linked page MUST exist
- **AND** no page MUST reference a missing counterpart
- **AND** all cross-references MUST resolve correctly