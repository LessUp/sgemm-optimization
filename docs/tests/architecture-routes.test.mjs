import test from 'node:test'
import assert from 'node:assert/strict'
import { existsSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const testFile = fileURLToPath(import.meta.url)
const docsRoot = path.resolve(path.dirname(testFile), '..')

for (const locale of ['en', 'zh']) {
  test(`${locale} architecture overview uses only the directory route source`, () => {
    assert.equal(existsSync(path.join(docsRoot, locale, 'architecture', 'index.md')), true)
    assert.equal(existsSync(path.join(docsRoot, locale, 'architecture.md')), false)
  })
}
