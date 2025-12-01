/**
 * Benchmark tracking and regression detection tools
 *
 * These tools help track performance over time and detect regressions:
 * - Run benchmarks and store results
 * - Compare against baselines
 * - Detect performance trends
 */

import { Tool } from "@modelcontextprotocol/sdk/types.js";
import { execSync, spawn } from "child_process";
import { readFileSync, existsSync, writeFileSync } from "fs";
import { join } from "path";
import {
  insertBenchmark,
  getLatestBenchmarks,
  getBaselineBenchmark,
  compareTrend,
  BenchmarkRecord,
} from "../db.js";

function getHorusRoot(): string {
  return process.env.HORUS_ROOT || "/home/lord-patpak/horus/HORUS";
}

function getGitInfo(): { commit: string | null; branch: string | null } {
  const root = getHorusRoot();
  try {
    const commit = execSync("git rev-parse --short HEAD", {
      cwd: root,
      encoding: "utf-8",
    }).trim();
    const branch = execSync("git rev-parse --abbrev-ref HEAD", {
      cwd: root,
      encoding: "utf-8",
    }).trim();
    return { commit, branch };
  } catch (e) {
    return { commit: null, branch: null };
  }
}

export const benchmarkTools: Tool[] = [
  {
    name: "horus_run_benchmark",
    description:
      "Run a HORUS benchmark and store the results. Returns parsed metrics that can be compared to baselines.",
    inputSchema: {
      type: "object",
      properties: {
        benchmark: {
          type: "string",
          enum: ["link_performance", "production_messages", "network_transport", "all"],
          description: "Which benchmark to run",
        },
        save: {
          type: "boolean",
          description: "Whether to save results to the metrics database (default: true)",
        },
      },
      required: ["benchmark"],
    },
  },
  {
    name: "horus_get_benchmark_history",
    description:
      "Get historical benchmark results for trend analysis.",
    inputSchema: {
      type: "object",
      properties: {
        benchmark: {
          type: "string",
          description: "Benchmark name to query",
        },
        limit: {
          type: "number",
          description: "Number of recent results to return (default: 10)",
        },
      },
    },
  },
  {
    name: "horus_compare_benchmark",
    description:
      "Compare current benchmark results against baseline or recent history. Detects regressions.",
    inputSchema: {
      type: "object",
      properties: {
        benchmark: {
          type: "string",
          description: "Benchmark name",
        },
        metric: {
          type: "string",
          description: "Specific metric to compare (e.g., 'Link::send_recv')",
        },
      },
      required: ["benchmark"],
    },
  },
  {
    name: "horus_set_baseline",
    description:
      "Set the current benchmark results as the baseline for future comparisons.",
    inputSchema: {
      type: "object",
      properties: {
        benchmark: {
          type: "string",
          description: "Benchmark name",
        },
      },
      required: ["benchmark"],
    },
  },
  {
    name: "horus_list_benchmarks",
    description:
      "List all available benchmarks in the HORUS benchmark suite.",
    inputSchema: {
      type: "object",
      properties: {},
    },
  },
];

export async function handleBenchmarkTool(
  name: string,
  args: Record<string, unknown>
): Promise<{ content: Array<{ type: string; text: string }> }> {
  switch (name) {
    case "horus_run_benchmark":
      return runBenchmark(args.benchmark as string, args.save as boolean ?? true);
    case "horus_get_benchmark_history":
      return getBenchmarkHistory(args.benchmark as string, args.limit as number);
    case "horus_compare_benchmark":
      return compareBenchmark(args.benchmark as string, args.metric as string);
    case "horus_set_baseline":
      return setBaseline(args.benchmark as string);
    case "horus_list_benchmarks":
      return listBenchmarks();
    default:
      return { content: [{ type: "text", text: `Unknown tool: ${name}` }] };
  }
}

interface ParsedMetric {
  name: string;
  time_ns: number;
  throughput_mibps?: number;
}

function parseCriterionOutput(output: string): ParsedMetric[] {
  const metrics: ParsedMetric[] = [];
  const lines = output.split("\n");

  let currentBenchmark = "";

  for (const line of lines) {
    // Match benchmark name lines like "link_small_16B/Link::send_recv"
    const nameMatch = line.match(/^(\S+\/\S+)$/);
    if (nameMatch) {
      currentBenchmark = nameMatch[1];
      continue;
    }

    // Match time lines like "time:   [247.58 ns 258.97 ns 271.13 ns]"
    const timeMatch = line.match(/time:\s+\[[\d.]+ \w+ ([\d.]+) (ns|µs|ms|us)/);
    if (timeMatch && currentBenchmark) {
      let time_ns = parseFloat(timeMatch[1]);
      const unit = timeMatch[2];

      // Convert to nanoseconds
      if (unit === "µs" || unit === "us") time_ns *= 1000;
      else if (unit === "ms") time_ns *= 1_000_000;

      metrics.push({
        name: currentBenchmark,
        time_ns,
      });
    }

    // Match throughput lines like "thrpt:  [448.94 MiB/s 472.40 MiB/s 496.23 MiB/s]"
    const thrptMatch = line.match(/thrpt:\s+\[[\d.]+ \S+ ([\d.]+) (\S+)/);
    if (thrptMatch && metrics.length > 0) {
      const lastMetric = metrics[metrics.length - 1];
      lastMetric.throughput_mibps = parseFloat(thrptMatch[1]);
    }
  }

  return metrics;
}

async function runBenchmark(benchmark: string, save: boolean) {
  const root = getHorusRoot();
  const benchmarks =
    benchmark === "all"
      ? ["link_performance", "production_messages", "network_transport"]
      : [benchmark];

  const results: Record<string, ParsedMetric[]> = {};
  const gitInfo = getGitInfo();

  for (const bench of benchmarks) {
    try {
      const output = execSync(
        `cargo bench -p horus_benchmarks --bench ${bench} -- --noplot 2>&1`,
        {
          cwd: root,
          encoding: "utf-8",
          maxBuffer: 50 * 1024 * 1024,
          timeout: 600000, // 10 minutes
        }
      );

      const metrics = parseCriterionOutput(output);
      results[bench] = metrics;

      // Save to database
      if (save) {
        const timestamp = new Date().toISOString();
        for (const metric of metrics) {
          insertBenchmark({
            timestamp,
            git_commit: gitInfo.commit,
            git_branch: gitInfo.branch,
            benchmark_name: bench,
            metric_name: metric.name,
            value: metric.time_ns,
            unit: "ns",
            metadata: metric.throughput_mibps
              ? JSON.stringify({ throughput_mibps: metric.throughput_mibps })
              : null,
          });
        }
      }
    } catch (e) {
      results[bench] = [{ name: "error", time_ns: 0 }];
    }
  }

  return {
    content: [
      {
        type: "text",
        text: JSON.stringify(
          {
            benchmarks: results,
            git: gitInfo,
            saved: save,
            timestamp: new Date().toISOString(),
          },
          null,
          2
        ),
      },
    ],
  };
}

function getBenchmarkHistory(benchmark?: string, limit = 10) {
  const records = getLatestBenchmarks(benchmark, limit);

  // Group by metric name
  const grouped: Record<string, BenchmarkRecord[]> = {};
  for (const record of records) {
    const key = `${record.benchmark_name}/${record.metric_name}`;
    if (!grouped[key]) grouped[key] = [];
    grouped[key].push(record);
  }

  return {
    content: [
      {
        type: "text",
        text: JSON.stringify(
          {
            filter: benchmark || "all",
            total_records: records.length,
            grouped,
          },
          null,
          2
        ),
      },
    ],
  };
}

function compareBenchmark(benchmark: string, metric?: string) {
  const records = getLatestBenchmarks(benchmark, 20);

  if (records.length === 0) {
    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            status: "no_data",
            message: `No benchmark data found for '${benchmark}'. Run horus_run_benchmark first.`,
          }, null, 2),
        },
      ],
    };
  }

  // Get unique metrics
  const metrics = [...new Set(records.map((r) => r.metric_name))];
  const metricsToCompare = metric ? [metric] : metrics;

  const comparisons = metricsToCompare.map((metricName) => {
    const baseline = getBaselineBenchmark(benchmark, metricName);
    const trend = compareTrend(benchmark, metricName);
    const latest = records.find((r) => r.metric_name === metricName);

    let regression = false;
    let percentFromBaseline = 0;

    if (baseline && latest) {
      percentFromBaseline = ((latest.value - baseline.value) / baseline.value) * 100;
      regression = percentFromBaseline > 10; // 10% regression threshold
    }

    return {
      metric: metricName,
      latest_value: latest?.value,
      latest_timestamp: latest?.timestamp,
      baseline_value: baseline?.value,
      baseline_commit: baseline?.git_commit,
      percent_from_baseline: percentFromBaseline.toFixed(2),
      trend: trend.trend,
      trend_percent: trend.percentChange.toFixed(2),
      regression,
    };
  });

  const hasRegressions = comparisons.some((c) => c.regression);

  return {
    content: [
      {
        type: "text",
        text: JSON.stringify(
          {
            benchmark,
            has_regressions: hasRegressions,
            summary: hasRegressions
              ? "⚠️ REGRESSION DETECTED - Performance has degraded significantly"
              : "✓ Performance is stable",
            comparisons,
          },
          null,
          2
        ),
      },
    ],
  };
}

function setBaseline(benchmark: string) {
  // The baseline is just the first entry in the database
  // To set a new baseline, we'd need to mark it specially
  // For now, just acknowledge and suggest running fresh benchmarks

  return {
    content: [
      {
        type: "text",
        text: JSON.stringify({
          status: "baseline_note",
          message: `Baselines are the first recorded benchmark for each metric. To reset baselines, delete ~/.horus-mcp/metrics.db and run benchmarks again.`,
          benchmark,
        }, null, 2),
      },
    ],
  };
}

function listBenchmarks() {
  const root = getHorusRoot();
  const benchDir = join(root, "benchmarks", "benches");

  let benchmarks: string[] = [];
  try {
    const files = execSync(`ls ${benchDir}/*.rs 2>/dev/null || true`, {
      encoding: "utf-8",
    });
    benchmarks = files
      .split("\n")
      .filter(Boolean)
      .map((f) => f.replace(benchDir + "/", "").replace(".rs", ""));
  } catch (e) {
    benchmarks = ["link_performance", "production_messages", "network_transport"];
  }

  // Also list available test binaries
  let testBinaries: string[] = [];
  try {
    const cargoToml = readFileSync(join(root, "benchmarks", "Cargo.toml"), "utf-8");
    const binMatches = cargoToml.match(/name = "test_\w+"/g) || [];
    testBinaries = binMatches.map((m) => m.replace('name = "', "").replace('"', ""));
  } catch (e) {
    // Ignore
  }

  return {
    content: [
      {
        type: "text",
        text: JSON.stringify(
          {
            criterion_benchmarks: benchmarks,
            test_binaries: testBinaries,
            run_command: "cargo bench -p horus_benchmarks --bench <name>",
            test_command: "cargo run -p horus_benchmarks --bin <name>",
          },
          null,
          2
        ),
      },
    ],
  };
}
