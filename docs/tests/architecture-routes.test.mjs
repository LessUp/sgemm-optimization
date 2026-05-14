import test from 'node:test'
import assert from 'node:assert/strict'
import { existsSync, readFileSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const testFile = fileURLToPath(import.meta.url)
const docsRoot = path.resolve(path.dirname(testFile), '..')

for (const locale of ['en', 'zh']) {
  test(`${locale} architecture overview uses only the directory route source`, () => {
    assert.equal(existsSync(path.join(docsRoot, locale, 'architecture', 'index.md')), true)
    assert.equal(existsSync(path.join(docsRoot, locale, 'architecture.md')), false)
  })

  test(`${locale} methodology and validation sections use canonical directory routes`, () => {
    assert.equal(existsSync(path.join(docsRoot, locale, 'methodology', 'index.md')), true)
    assert.equal(existsSync(path.join(docsRoot, locale, 'methodology.md')), false)
    assert.equal(existsSync(path.join(docsRoot, locale, 'validation', 'index.md')), true)
    assert.equal(existsSync(path.join(docsRoot, locale, 'validation.md')), false)
  })

  test(`${locale} resources section uses the canonical directory route`, () => {
    assert.equal(existsSync(path.join(docsRoot, locale, 'resources', 'index.md')), true)
    assert.equal(existsSync(path.join(docsRoot, locale, 'resources.md')), false)
  })

  test(`${locale} homepage avoids root-absolute internal links`, () => {
    const homepage = readFileSync(path.join(docsRoot, locale, 'index.md'), 'utf8')
    assert.equal(/href="\/(?:en|zh)\//.test(homepage), false)
  })
}
