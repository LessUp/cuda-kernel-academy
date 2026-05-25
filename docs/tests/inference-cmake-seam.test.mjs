import test from 'node:test'
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const repoRoot = path.resolve(__dirname, '..', '..')
const cmakePath = path.join(repoRoot, '04-inference-engine', 'CMakeLists.txt')

test('04-inference-engine requires the parent TensorCraft target and links through it', async () => {
  const cmake = await readFile(cmakePath, 'utf8')

  assert.match(
    cmake,
    /message\(\s*FATAL_ERROR[\s\S]*TensorCraft::tensorcraft target[\s\S]*repository root/,
    'Expected 04-inference-engine to fail clearly when TensorCraft is not provided by the parent build'
  )

  assert.match(
    cmake,
    /target_link_libraries\(mini_inference[\s\S]*TensorCraft::tensorcraft/,
    'Expected mini_inference to link through the stable TensorCraft::tensorcraft target'
  )

  assert.doesNotMatch(
    cmake,
    /MINI_INFERENCE_STANDALONE|TENSORCRAFT_ROOT|USE_TENSORCRAFT/,
    'Expected the standalone fallback seam and toggle option to be removed from 04-inference-engine/CMakeLists.txt'
  )
})
