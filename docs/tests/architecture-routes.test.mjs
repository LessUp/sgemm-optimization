import test from 'node:test'
import assert from 'node:assert/strict'
import { existsSync, readFileSync, readdirSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const testFile = fileURLToPath(import.meta.url)
const docsRoot = path.resolve(path.dirname(testFile), '..')

for (const locale of ['en', 'zh']) {
  test(`${locale} overview and academy sections use canonical directory routes`, () => {
    assert.equal(existsSync(path.join(docsRoot, locale, 'overview', 'index.md')), true)
    assert.equal(existsSync(path.join(docsRoot, locale, 'academy', 'index.md')), true)
    assert.equal(existsSync(path.join(docsRoot, locale, 'research', 'index.md')), true)

    assert.equal(existsSync(path.join(docsRoot, locale, 'methodology', 'index.md')), false)
    assert.equal(existsSync(path.join(docsRoot, locale, 'resources', 'index.md')), false)
    assert.equal(existsSync(path.join(docsRoot, locale, 'learning-path.md')), false)
    assert.equal(existsSync(path.join(docsRoot, locale, 'references.md')), false)
  })

  test(`${locale} architecture overview uses only the directory route source`, () => {
    assert.equal(existsSync(path.join(docsRoot, locale, 'architecture', 'index.md')), true)
    assert.equal(existsSync(path.join(docsRoot, locale, 'architecture.md')), false)
  })

  test(`${locale} validation remains a directory route`, () => {
    assert.equal(existsSync(path.join(docsRoot, locale, 'validation', 'index.md')), true)
    assert.equal(existsSync(path.join(docsRoot, locale, 'validation.md')), false)
  })

  test(`${locale} homepage avoids root-absolute internal links`, () => {
    const homepage = readFileSync(path.join(docsRoot, locale, 'index.md'), 'utf8')
    assert.equal(/href="\/(?:en|zh)\//.test(homepage), false)
    assert.equal(homepage.includes('./overview/'), true)
    assert.equal(homepage.includes('./academy/'), true)
    assert.equal(homepage.includes('./research/'), true)
  })

  test(`${locale} whitepaper route additions exist`, () => {
    assert.equal(existsSync(path.join(docsRoot, locale, 'overview', 'reader-map.md')), true)
    assert.equal(existsSync(path.join(docsRoot, locale, 'architecture', 'system-blueprint.md')), true)
    assert.equal(existsSync(path.join(docsRoot, locale, 'validation', 'performance-model.md')), true)
    assert.equal(existsSync(path.join(docsRoot, locale, 'research', 'reference-map.md')), true)
  })

  test(`${locale} curated figures use the shared theme-aware component`, () => {
    const homepage = readFileSync(path.join(docsRoot, locale, 'index.md'), 'utf8')
    const architecture = readFileSync(path.join(docsRoot, locale, 'architecture', 'index.md'), 'utf8')
    const blueprint = readFileSync(path.join(docsRoot, locale, 'architecture', 'system-blueprint.md'), 'utf8')

    assert.equal(homepage.includes('<ThemedFigure'), true)
    assert.equal(architecture.includes('<ThemedFigure'), true)
    assert.equal(blueprint.includes('<ThemedFigure'), true)
    assert.equal(homepage.includes('<picture>'), false)
    assert.equal(architecture.includes('<picture>'), false)
    assert.equal(blueprint.includes('<picture>'), false)
  })
}

test('whitepaper figure assets exist for both light and dark themes', () => {
  assert.equal(existsSync(path.join(docsRoot, 'public', 'figures', 'whitepaper-system-light.svg')), true)
  assert.equal(existsSync(path.join(docsRoot, 'public', 'figures', 'whitepaper-system-dark.svg')), true)
  assert.equal(existsSync(path.join(docsRoot, 'public', 'figures', 'kernel-ladder-light.svg')), true)
  assert.equal(existsSync(path.join(docsRoot, 'public', 'figures', 'kernel-ladder-dark.svg')), true)
})

test('shared theme-aware figure component exists', () => {
  assert.equal(existsSync(path.join(docsRoot, '.vitepress', 'components', 'ThemedFigure.vue')), true)
})

test('figure SVGs do not contain malformed spaced hex colors', () => {
  const figureDir = path.join(docsRoot, 'public', 'figures')

  for (const file of readdirSync(figureDir)) {
    if (!file.endsWith('.svg'))
      continue

    const svg = readFileSync(path.join(figureDir, file), 'utf8')
    assert.equal(/#[0-9A-Fa-f]{2,}\s+[0-9A-Fa-f]+/.test(svg), false, `${file} contains a malformed hex color`)
  }
})
