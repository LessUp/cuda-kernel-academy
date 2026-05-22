import test from 'node:test'
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const repoRoot = path.resolve(__dirname, '..', '..')
const readmePath = path.join(repoRoot, '04-inference-engine', 'README.md')
const cmakePath = path.join(repoRoot, '04-inference-engine', 'CMakeLists.txt')

test('04-inference-engine README documents real CMake artifacts', async () => {
  const [readme, cmake] = await Promise.all([
    readFile(readmePath, 'utf8'),
    readFile(cmakePath, 'utf8')
  ])

  assert.match(cmake, /add_executable\(mini_inference_benchmark benchmarks\/benchmark\.cpp\)/)
  assert.match(readme, /mini_inference_benchmark/u)
  assert.match(readme, /detailed_benchmark/u)
  assert.match(readme, /mnist_demo/u)
  assert.doesNotMatch(readme, /- `benchmark`：|\.\/benchmark\b/u)
})
