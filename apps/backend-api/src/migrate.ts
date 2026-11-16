import { readdir, readFile } from 'node:fs/promises'
import type { Pool } from 'pg'

const DIR = new URL('../migrations/', import.meta.url)
const LOCK_KEY = 727274 // serialize concurrent instances

/**
 * Applies migrations/V<n>__*.sql in order. Versions already recorded by the old
 * Flyway setup (flyway_schema_history) count as applied, so existing DBs aren't re-initialised.
 */
export async function migrate(pool: Pool): Promise<void> {
  const client = await pool.connect()
  try {
    await client.query('SELECT pg_advisory_lock($1)', [LOCK_KEY])
    await client.query('CREATE TABLE IF NOT EXISTS schema_migrations (version INT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())')

    const applied = new Set<number>()
    for (const r of (await client.query<{ version: number }>('SELECT version FROM schema_migrations')).rows) applied.add(r.version)
    const hasFlyway = (await client.query("SELECT to_regclass('flyway_schema_history') AS t")).rows[0].t
    if (hasFlyway) {
      const fw = await client.query<{ version: string }>('SELECT version FROM flyway_schema_history WHERE success AND version IS NOT NULL')
      for (const r of fw.rows) applied.add(Number(r.version))
    }

    const files = (await readdir(DIR))
      .map((f) => ({ f, v: Number(/^V(\d+)__.*\.sql$/.exec(f)?.[1]) }))
      .filter((x) => Number.isInteger(x.v))
      .sort((a, b) => a.v - b.v)

    for (const { f, v } of files) {
      if (applied.has(v)) continue
      const sql = (await readFile(new URL(f, DIR), 'utf8')).replace(/^﻿/, '')
      try {
        await client.query('BEGIN')
        await client.query(sql)
        await client.query('INSERT INTO schema_migrations (version) VALUES ($1) ON CONFLICT DO NOTHING', [v])
        await client.query('COMMIT')
      } catch (e) {
        await client.query('ROLLBACK').catch(() => {})
        throw new Error(`migration ${f} failed: ${(e as Error).message}`)
      }
    }
  } finally {
    await client.query('SELECT pg_advisory_unlock($1)', [LOCK_KEY]).catch(() => {})
    client.release()
  }
}
