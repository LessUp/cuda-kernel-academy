import test from 'node:test'
import assert from 'node:assert/strict'
import { readdir, readFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const docsRoot = path.resolve(__dirname, '..')
const contentRoots = [path.join(docsRoot, 'en'), path.join(docsRoot, 'zh')]
const forbiddenPatterns = [
  /^\s*(?:style|classDef)\b.*(?:fill|color)\s*:\s*#(?:[0-9a-f]{3}|[0-9a-f]{4}|[0-9a-f]{6}|[0-9a-f]{8})\b/i
]

function collectMermaidBlocks(markdown) {
  const blocks = []
  let inMermaidBlock = false
  let currentBlock = []

  for (const line of markdown.split('\n')) {
    const fenceMatch = line.match(/^```(\w+)?\s*$/)

    if (fenceMatch) {
      if (inMermaidBlock) {
        blocks.push(currentBlock.join('\n'))
        currentBlock = []
      }

      inMermaidBlock = fenceMatch[1]?.toLowerCase() === 'mermaid'
      continue
    }

    if (inMermaidBlock) {
      currentBlock.push(line)
    }
  }

  return blocks
}

async function collectMarkdownFiles(dir) {
  const entries = await readdir(dir, { withFileTypes: true })
  const nested = await Promise.all(entries.map(async (entry) => {
    const fullPath = path.join(dir, entry.name)
    if (entry.isDirectory()) {
      return collectMarkdownFiles(fullPath)
    }

    return entry.isFile() && entry.name.endsWith('.md') ? [fullPath] : []
  }))

  return nested.flat()
}

test('docs mermaid diagrams avoid hardcoded dark theme colors', async () => {
  const markdownFiles = (await Promise.all(contentRoots.map(collectMarkdownFiles))).flat()
  const violations = []

  for (const markdownFile of markdownFiles) {
    const content = await readFile(markdownFile, 'utf8')
    const mermaidBlocks = collectMermaidBlocks(content)

    for (const token of forbiddenPatterns) {
      mermaidBlocks.forEach((block, blockIndex) => {
        block.split('\n').forEach((line, lineIndex) => {
          if (token.test(line)) {
            violations.push(
              `${path.relative(docsRoot, markdownFile)}:mermaid-block-${blockIndex + 1}:${
                lineIndex + 1
              }: ${line.trim()}`
            )
          }
        })
      })
    }
  }

  assert.deepStrictEqual(
    violations,
    [],
    `Found theme-hostile Mermaid styling:\n${violations.join('\n')}`
  )
})
