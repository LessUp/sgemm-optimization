import test from 'node:test'
import assert from 'node:assert/strict'
import { existsSync, readFileSync } from 'node:fs'
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
}

test('whitepaper figure assets exist for both light and dark themes', () => {
  assert.equal(existsSync(path.join(docsRoot, 'public', 'figures', 'whitepaper-system-light.svg')), true)
  assert.equal(existsSync(path.join(docsRoot, 'public', 'figures', 'whitepaper-system-dark.svg')), true)
  assert.equal(existsSync(path.join(docsRoot, 'public', 'figures', 'kernel-ladder-light.svg')), true)
  assert.equal(existsSync(path.join(docsRoot, 'public', 'figures', 'kernel-ladder-dark.svg')), true)
})
