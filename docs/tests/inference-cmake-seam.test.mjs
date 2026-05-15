import test from 'node:test'
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const repoRoot = path.resolve(__dirname, '..', '..')
const cmakePath = path.join(repoRoot, '04-inference-engine', 'CMakeLists.txt')

test('04-inference-engine only injects sibling TensorCraft in standalone builds', async () => {
  const cmake = await readFile(cmakePath, 'utf8')

  assert.match(
    cmake,
    /if\(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR\)[\s\S]*set\(MINI_INFERENCE_STANDALONE ON\)/,
    'Expected standalone-build detection guard in 04-inference-engine/CMakeLists.txt'
  )

  assert.match(
    cmake,
    /elseif\(MINI_INFERENCE_STANDALONE\)[\s\S]*add_subdirectory\("\$\{TENSORCRAFT_ROOT\}"\s+tensorcraft-from-parent[\s\S]*endif\(\)/,
    'Expected sibling-path TensorCraft fallback to be limited to standalone builds'
  )

  assert.match(
    cmake,
    /message\(\s*FATAL_ERROR[\s\S]*USE_TENSORCRAFT=ON requires an upstream TensorCraft::tensorcraft target/,
    'Expected parent builds to fail clearly when the upstream TensorCraft target is missing'
  )

  assert.match(
    cmake,
    /target_link_libraries\(mini_inference[\s\S]*TensorCraft::tensorcraft/,
    'Expected mini_inference to link through the stable TensorCraft::tensorcraft target'
  )

  assert.doesNotMatch(
    cmake,
    /target_include_directories\(mini_inference[\s\S]*TENSORCRAFT_ROOT\/include/,
    'Expected TensorCraft headers to flow through the linked target instead of manual include injection'
  )
})
