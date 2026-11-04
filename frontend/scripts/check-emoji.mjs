import { readdir, readFile } from 'node:fs/promises'
import { join } from 'node:path'

const roots = [
  'src/components/layout',
  'src/components/screens',
  'src/components/ui',
]
const extensions = new Set(['.js', '.jsx'])
const forbidden = /[\p{Extended_Pictographic}✓✔⚠✕✖]/u

async function sourceFiles(directory) {
  const entries = await readdir(directory, { withFileTypes: true })
  const nested = await Promise.all(entries.map((entry) => {
    const path = join(directory, entry.name)
    if (entry.isDirectory()) return sourceFiles(path)
    return extensions.has(entry.name.slice(entry.name.lastIndexOf('.'))) ? [path] : []
  }))
  return nested.flat()
}

const files = [...(await Promise.all(roots.map(sourceFiles))).flat(), 'src/pages/AppPage.jsx']
const failures = []
for (const file of files) {
  const text = await readFile(file, 'utf8')
  if (forbidden.test(text)) failures.push(file)
}

if (failures.length) {
  console.error(`Emoji glyphs found in app code:\n${failures.join('\n')}`)
  process.exit(1)
}
