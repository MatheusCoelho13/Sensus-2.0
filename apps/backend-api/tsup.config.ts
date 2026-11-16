import { defineConfig } from 'tsup'

// shared-contracts ships raw .ts (extensionless imports) → must be bundled, not left external.
export default defineConfig({
  entry: ['src/server.ts'],
  format: 'esm',
  target: 'node22',
  clean: true,
  noExternal: ['@sensus/shared-contracts'],
})
