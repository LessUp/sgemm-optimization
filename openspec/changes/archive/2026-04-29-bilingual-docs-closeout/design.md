# Design: Bilingual Documentation and Pages Closeout

## Architecture Overview

Keep one GitHub Pages deployment with mirrored English and Chinese routes:

```
/                    → English home
/zh/                 → Chinese home
/docs/*              → English tutorials
/zh/docs/*           → Chinese tutorials
/specs               → English specs landing
/zh/specs            → Chinese specs landing
```

## Content Model

### Public Surface Policy

```yaml
public_surface_policy:
  normative_source: openspec/specs/*
  bilingual_public_surfaces:
    - README.md
    - README.zh-CN.md
    - index.md
    - specs.md
    - docs/*
    - zh/index.md
    - zh/specs.md
    - zh/docs/*
  route_model:
    english_root: /
    chinese_root: /zh/
  page_pairing_keys:
    - lang          # "en" or "zh-CN"
    - page_key      # unique identifier for the content
    - lang_ref      # page_key of the counterpart page
```

### Page Front Matter Template

English page:
```yaml
---
lang: en
page_key: kernel-naive
lang_ref: zh-kernel-naive
permalink: /docs/kernel-naive/
---
```

Chinese counterpart:
```yaml
---
lang: zh-CN
page_key: zh-kernel-naive
lang_ref: kernel-naive
permalink: /zh/docs/kernel-naive/
---
```

## Components

### 1. Language Switcher Include

Location: `_includes/nav_footer_custom.html`

```html
<div class="language-switcher" 
     data-page-key="{{ page.page_key }}" 
     data-lang="{{ page.lang }}" 
     data-lang-ref="{{ page.lang_ref }}">
  <button type="button" data-lang-choice="en" 
          class="{% if page.lang == 'en' %}is-active{% endif %}">
    English
  </button>
  <button type="button" data-lang-choice="zh-CN"
          class="{% if page.lang == 'zh-CN' %}is-active{% endif %}">
    简体中文
  </button>
</div>
```

### 2. Route Switching Logic

Location: `assets/js/language-toggle.js`

Responsibilities:
- Read current page metadata from DOM
- On language button click, find paired page path
- Store preference in localStorage
- Redirect to counterpart or language home

```js
// Route mapping for page pairs
const pagePairs = {
  'home': { en: '/', 'zh-CN': '/zh/' },
  'specs': { en: '/specs/', 'zh-CN': '/zh/specs/' },
  'getting-started': { en: '/docs/getting-started/', 'zh-CN': '/zh/docs/getting-started/' },
  // ... additional mappings
};

function findPairedPath(pageKey, targetLang) {
  const pair = pagePairs[pageKey];
  if (pair && pair[targetLang]) {
    return pair[targetLang];
  }
  // Fallback to language home
  return targetLang === 'zh-CN' ? '/zh/' : '/';
}
```

### 3. SCSS Styling

Location: `_sass/custom/custom.scss`

```scss
.language-switcher {
  display: flex;
  gap: 0.5rem;
  margin: 0.5rem 0;
}

.language-switcher button {
  padding: 0.25rem 0.75rem;
  border: 1px solid var(--border-color, #ddd);
  border-radius: 4px;
  background: transparent;
  cursor: pointer;
  font-size: 0.875rem;
}

.language-switcher button.is-active {
  border-color: var(--nvidia-green, #76b900);
  color: var(--nvidia-green, #76b900);
  font-weight: 500;
}
```

## Navigation Strategy

### Sidebar Organization

The just-the-docs sidebar uses `nav_order` for sequencing. We have two options:

**Option A: Language-filtered navigation (preferred)**
- Use front matter `lang` to filter sidebar via custom include
- Each language sees only its own pages in navigation

**Option B: Unified navigation with language sections**
- Group pages by language with section headers
- Simpler to implement but mixes languages in sidebar

Default to Option B for simplicity unless Option A proves necessary.

### Cross-Language Linking

Each page's switcher provides the primary cross-language link. Additionally:
- README files cross-link to each other
- Home pages mention the alternate language version
- Tutorial pages include "Also available in: X" notes

## File Layout

### New Directories

```
zh/
├── index.md
├── specs.md
└── docs/
    ├── getting-started.md
    ├── learning-path.md
    ├── architecture.md
    ├── benchmark-results.md
    ├── kernel-naive.md
    ├── kernel-tiled.md
    ├── kernel-bank-free.md
    ├── kernel-double-buffer.md
    └── kernel-tensor-core.md
```

### New Includes

```
_includes/
├── head_custom.html      # Load language-toggle.js
└── nav_footer_custom.html # Switcher markup

assets/js/
└── language-toggle.js    # Switching logic
```

## README Structure

Both READMEs follow identical outline:

1. **Value Proposition** — One-sentence project description
2. **Optimization Ladder** — Brief table of kernel progression
3. **Quick Start** — Minimal commands to build and run
4. **Learning Route** — Table mapping goals to entry points
5. **Validation Boundary** — What's tested, what requires CUDA
6. **Repository Layout** — Directory overview
7. **Project Status** — Archive/completion notes

## Metadata Alignment

### GitHub Repository

```bash
gh repo edit LessUp/sgemm-optimization \
  --description "Bilingual CUDA SGEMM optimization tutorial and reference implementation, from naive kernels to Tensor Core WMMA." \
  --homepage "https://lessup.github.io/sgemm-optimization/"
```

### _config.yml

```yaml
title: SGEMM Optimization
description: >-
  Bilingual CUDA SGEMM optimization tutorial and reference implementation,
  from naive kernels to Tensor Core WMMA.
aux_links:
  "GitHub":
    - "//github.com/LessUp/sgemm-optimization"
  "English README":
    - "//github.com/LessUp/sgemm-optimization/blob/master/README.md"
  "中文 README":
    - "//github.com/LessUp/sgemm-optimization/blob/master/README.zh-CN.md"
```

## Edge Cases

### Pages Without Counterparts

If a page exists in one language only:
- Still render switcher but show "Not available in [lang]" for missing language
- Or redirect to language home with a notice

### Search Behavior

Just-the-docs search indexes all pages. This is acceptable — readers can find content in either language.

### Mobile View

Switcher should remain accessible in mobile sidebar; test at common breakpoints.

## Rollback Plan

All changes are additive (new files, new includes, new directory). Rollback is straightforward:
1. Remove `zh/` directory
2. Remove `_includes/*.html` and `assets/js/language-toggle.js`
3. Revert modifications to `index.md`, `specs.md`, `docs/*.md`, `_config.yml`, `_sass/custom/custom.scss`

No code changes required — this is documentation and presentation work only.
