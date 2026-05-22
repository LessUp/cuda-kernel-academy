import test from 'node:test'
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const repoRoot = path.resolve(__dirname, '..', '..')

const canonicalRepoSlug = 'AICL-Lab/cuda-kernel-academy'
const canonicalRepoUrl = `https://github.com/${canonicalRepoSlug}`
const canonicalPagesUrl = 'https://aicl-lab.github.io/cuda-kernel-academy/'

const publicFiles = [
  'README.md',
  'README.zh-CN.md',
  'CONTRIBUTING.md',
  '.github/ISSUE_TEMPLATE/config.yml',
  'docs/.vitepress/config.ts',
  'docs/en/guides/getting-started.md',
  'docs/zh/guides/getting-started.md',
  'docs/zh/reference/integration-examples.md',
  'docs/zh/reference/troubleshooting.md'
]

test('public docs use one canonical repo and Pages identity', async () => {
  const violations = []

  for (const relativePath of publicFiles) {
    const content = await readFile(path.join(repoRoot, relativePath), 'utf8')

    if (content.includes('LessUp/cuda-kernel-academy')) {
      violations.push(`${relativePath}: found stale LessUp repo slug`)
    }

    if (content.includes('https://lessup.github.io/cuda-kernel-academy/')) {
      violations.push(`${relativePath}: found stale LessUp Pages URL`)
    }
  }

  const vitepressConfig = await readFile(path.join(repoRoot, 'docs/.vitepress/config.ts'), 'utf8')
  assert.match(vitepressConfig, new RegExp(canonicalPagesUrl.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')))
  assert.match(vitepressConfig, new RegExp(canonicalRepoUrl.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')))

  assert.deepStrictEqual(
    violations,
    [],
    `Found stale public identity links:\n${violations.join('\n')}`
  )
})
