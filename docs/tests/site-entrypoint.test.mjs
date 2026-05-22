import test from 'node:test'
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const docsRoot = path.resolve(__dirname, '..')
const entrypointPath = path.join(docsRoot, 'index.md')

test('docs root is a stable landing page, not a redirect shim', async () => {
  const entrypoint = await readFile(entrypointPath, 'utf8')

  assert.doesNotMatch(entrypoint, /navigator\.language|window\.location\.href|redirecting|setTimeout\(/)
  assert.doesNotMatch(entrypoint, /\/cuda-kernel-academy\/(?:en|zh)\//)

  assert.match(entrypoint, /\/en\//)
  assert.match(entrypoint, /\/zh\//)
  assert.match(entrypoint, /Reading path|阅读路径/i)
})
