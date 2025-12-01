/**
 * SQLite database for storing benchmark metrics and history
 */

import Database from "better-sqlite3";
import { homedir } from "os";
import { join } from "path";
import { mkdirSync, existsSync } from "fs";

let db: Database.Database | null = null;

export async function initDatabase(): Promise<void> {
  const dataDir = join(homedir(), ".horus-mcp");
  if (!existsSync(dataDir)) {
    mkdirSync(dataDir, { recursive: true });
  }

  db = new Database(join(dataDir, "metrics.db"));

  // Create tables
  db.exec(`
    CREATE TABLE IF NOT EXISTS benchmark_runs (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      timestamp TEXT NOT NULL,
      git_commit TEXT,
      git_branch TEXT,
      benchmark_name TEXT NOT NULL,
      metric_name TEXT NOT NULL,
      value REAL NOT NULL,
      unit TEXT,
      metadata TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_benchmark_runs_name
      ON benchmark_runs(benchmark_name, metric_name);

    CREATE INDEX IF NOT EXISTS idx_benchmark_runs_timestamp
      ON benchmark_runs(timestamp);

    CREATE TABLE IF NOT EXISTS validation_runs (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      timestamp TEXT NOT NULL,
      validation_type TEXT NOT NULL,
      passed INTEGER NOT NULL,
      total INTEGER NOT NULL,
      details TEXT
    );
  `);
}

export function getDb(): Database.Database {
  if (!db) {
    throw new Error("Database not initialized. Call initDatabase() first.");
  }
  return db;
}

export interface BenchmarkRecord {
  timestamp: string;
  git_commit: string | null;
  git_branch: string | null;
  benchmark_name: string;
  metric_name: string;
  value: number;
  unit: string | null;
  metadata: string | null;
}

export function insertBenchmark(record: BenchmarkRecord): void {
  const stmt = getDb().prepare(`
    INSERT INTO benchmark_runs
    (timestamp, git_commit, git_branch, benchmark_name, metric_name, value, unit, metadata)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
  `);
  stmt.run(
    record.timestamp,
    record.git_commit,
    record.git_branch,
    record.benchmark_name,
    record.metric_name,
    record.value,
    record.unit,
    record.metadata
  );
}

export function getLatestBenchmarks(benchmarkName?: string, limit = 10): BenchmarkRecord[] {
  let query = `
    SELECT * FROM benchmark_runs
    ${benchmarkName ? "WHERE benchmark_name = ?" : ""}
    ORDER BY timestamp DESC
    LIMIT ?
  `;
  const stmt = getDb().prepare(query);
  return benchmarkName
    ? stmt.all(benchmarkName, limit) as BenchmarkRecord[]
    : stmt.all(limit) as BenchmarkRecord[];
}

export function getBaselineBenchmark(benchmarkName: string, metricName: string): BenchmarkRecord | null {
  const stmt = getDb().prepare(`
    SELECT * FROM benchmark_runs
    WHERE benchmark_name = ? AND metric_name = ?
    ORDER BY timestamp ASC
    LIMIT 1
  `);
  return stmt.get(benchmarkName, metricName) as BenchmarkRecord | null;
}

export function compareTrend(
  benchmarkName: string,
  metricName: string,
  recentCount = 5
): { trend: "improving" | "regressing" | "stable"; percentChange: number } {
  const stmt = getDb().prepare(`
    SELECT value FROM benchmark_runs
    WHERE benchmark_name = ? AND metric_name = ?
    ORDER BY timestamp DESC
    LIMIT ?
  `);
  const results = stmt.all(benchmarkName, metricName, recentCount + 1) as { value: number }[];

  if (results.length < 2) {
    return { trend: "stable", percentChange: 0 };
  }

  const recent = results.slice(0, recentCount).reduce((a, b) => a + b.value, 0) / recentCount;
  const older = results[results.length - 1].value;
  const percentChange = ((recent - older) / older) * 100;

  if (percentChange > 5) {
    return { trend: "regressing", percentChange };
  } else if (percentChange < -5) {
    return { trend: "improving", percentChange };
  }
  return { trend: "stable", percentChange };
}
