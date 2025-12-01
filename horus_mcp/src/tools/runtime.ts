/**
 * Runtime debugging tools
 *
 * These tools inspect running HORUS applications:
 * - Active topics and their statistics
 * - Node health and heartbeats
 * - Log inspection
 * - Session management
 */

import { Tool } from "@modelcontextprotocol/sdk/types.js";
import { execSync } from "child_process";
import { readFileSync, existsSync, readdirSync, statSync } from "fs";
import { join, basename } from "path";

// Platform-specific shared memory paths
function getShmBase(): string {
  if (process.platform === "linux") {
    return "/dev/shm/horus";
  } else if (process.platform === "darwin") {
    return "/tmp/horus";
  } else {
    return join(process.env.TEMP || "/tmp", "horus");
  }
}

const SHM_BASE = getShmBase();

export const runtimeTools: Tool[] = [
  {
    name: "horus_list_topics",
    description:
      "List all active HORUS topics in shared memory with their sizes and activity status.",
    inputSchema: {
      type: "object",
      properties: {
        session_id: {
          type: "string",
          description: "Optional session ID to filter topics",
        },
      },
    },
  },
  {
    name: "horus_inspect_topic",
    description:
      "Get detailed information about a specific topic including message type, size, and recent activity.",
    inputSchema: {
      type: "object",
      properties: {
        topic_name: {
          type: "string",
          description: "Name of the topic to inspect",
        },
      },
      required: ["topic_name"],
    },
  },
  {
    name: "horus_list_sessions",
    description:
      "List all active HORUS sessions with their PIDs and status.",
    inputSchema: {
      type: "object",
      properties: {},
    },
  },
  {
    name: "horus_get_node_health",
    description:
      "Get health status of nodes from heartbeat files.",
    inputSchema: {
      type: "object",
      properties: {
        node_name: {
          type: "string",
          description: "Optional node name to filter",
        },
      },
    },
  },
  {
    name: "horus_tail_logs",
    description:
      "Get recent log entries from HORUS shared memory log buffer.",
    inputSchema: {
      type: "object",
      properties: {
        lines: {
          type: "number",
          description: "Number of lines to retrieve (default: 50)",
        },
        filter: {
          type: "string",
          description: "Optional filter pattern (e.g., 'ERROR', node name)",
        },
      },
    },
  },
  {
    name: "horus_check_shm",
    description:
      "Check the overall state of HORUS shared memory including total size and file count.",
    inputSchema: {
      type: "object",
      properties: {},
    },
  },
  {
    name: "horus_kill_session",
    description:
      "Kill a HORUS session by session ID (sends SIGTERM to all PIDs).",
    inputSchema: {
      type: "object",
      properties: {
        session_id: {
          type: "string",
          description: "Session ID to kill",
        },
      },
      required: ["session_id"],
    },
  },
  {
    name: "horus_clean_shm",
    description:
      "Clean up stale shared memory from dead sessions.",
    inputSchema: {
      type: "object",
      properties: {
        dry_run: {
          type: "boolean",
          description: "If true, only report what would be cleaned (default: true)",
        },
      },
    },
  },
];

export async function handleRuntimeTool(
  name: string,
  args: Record<string, unknown>
): Promise<{ content: Array<{ type: string; text: string }> }> {
  switch (name) {
    case "horus_list_topics":
      return listTopics(args.session_id as string | undefined);
    case "horus_inspect_topic":
      return inspectTopic(args.topic_name as string);
    case "horus_list_sessions":
      return listSessions();
    case "horus_get_node_health":
      return getNodeHealth(args.node_name as string | undefined);
    case "horus_tail_logs":
      return tailLogs(args.lines as number | undefined, args.filter as string | undefined);
    case "horus_check_shm":
      return checkShm();
    case "horus_kill_session":
      return killSession(args.session_id as string);
    case "horus_clean_shm":
      return cleanShm(args.dry_run as boolean ?? true);
    default:
      return { content: [{ type: "text", text: `Unknown tool: ${name}` }] };
  }
}

function listTopics(sessionId?: string) {
  const topicsDir = sessionId
    ? join(SHM_BASE, "sessions", sessionId, "topics")
    : join(SHM_BASE, "topics");

  if (!existsSync(topicsDir)) {
    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            status: "no_topics",
            message: `Topics directory not found: ${topicsDir}`,
            hint: "No HORUS application is currently running, or shared memory is not initialized.",
          }, null, 2),
        },
      ],
    };
  }

  try {
    const files = readdirSync(topicsDir);
    const topics = files.map((file) => {
      const filePath = join(topicsDir, file);
      const stats = statSync(filePath);
      return {
        name: file,
        size_bytes: stats.size,
        last_modified: stats.mtime.toISOString(),
        age_ms: Date.now() - stats.mtime.getTime(),
        active: Date.now() - stats.mtime.getTime() < 5000, // Active if modified in last 5s
      };
    });

    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            topics_dir: topicsDir,
            topic_count: topics.length,
            topics,
          }, null, 2),
        },
      ],
    };
  } catch (e) {
    return { content: [{ type: "text", text: `Error listing topics: ${e}` }] };
  }
}

function inspectTopic(topicName: string) {
  // Try multiple locations
  const candidates = [
    join(SHM_BASE, "topics", topicName),
    join(SHM_BASE, "topics", "hubs", topicName),
    join(SHM_BASE, "topics", "links", topicName),
  ];

  let topicPath: string | null = null;
  for (const candidate of candidates) {
    if (existsSync(candidate)) {
      topicPath = candidate;
      break;
    }
  }

  if (!topicPath) {
    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            error: "Topic not found",
            topic: topicName,
            searched: candidates,
          }, null, 2),
        },
      ],
    };
  }

  try {
    const stats = statSync(topicPath);

    // Try to read header info (first 64 bytes for Hub/Link)
    let headerInfo: Record<string, unknown> = {};
    try {
      const fd = readFileSync(topicPath);
      if (fd.length >= 64) {
        // Read sequence number (first 8 bytes as u64)
        const sequence = fd.readBigUInt64LE(0);
        headerInfo = {
          sequence: sequence.toString(),
          header_size: 64,
          data_size: fd.length - 64,
        };
      }
    } catch (e) {
      headerInfo = { error: "Could not read header" };
    }

    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            topic: topicName,
            path: topicPath,
            size_bytes: stats.size,
            last_modified: stats.mtime.toISOString(),
            ...headerInfo,
          }, null, 2),
        },
      ],
    };
  } catch (e) {
    return { content: [{ type: "text", text: `Error inspecting topic: ${e}` }] };
  }
}

function listSessions() {
  const sessionsDir = join(SHM_BASE, "sessions");

  if (!existsSync(sessionsDir)) {
    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            status: "no_sessions",
            message: "No sessions directory found",
          }, null, 2),
        },
      ],
    };
  }

  try {
    const sessionIds = readdirSync(sessionsDir).filter((f) =>
      statSync(join(sessionsDir, f)).isDirectory()
    );

    const sessions = sessionIds.map((sessionId) => {
      const sessionDir = join(sessionsDir, sessionId);
      const pidsDir = join(sessionDir, "pids");

      let pids: number[] = [];
      let alive = false;

      if (existsSync(pidsDir)) {
        pids = readdirSync(pidsDir)
          .map((f) => parseInt(f))
          .filter((n) => !isNaN(n));

        // Check if any PID is alive
        for (const pid of pids) {
          try {
            process.kill(pid, 0);
            alive = true;
            break;
          } catch (e) {
            // Process doesn't exist
          }
        }
      }

      // Count topics
      const topicsDir = join(sessionDir, "topics");
      const topicCount = existsSync(topicsDir) ? readdirSync(topicsDir).length : 0;

      return {
        session_id: sessionId,
        pids,
        alive,
        topic_count: topicCount,
      };
    });

    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            sessions_dir: sessionsDir,
            session_count: sessions.length,
            active_count: sessions.filter((s) => s.alive).length,
            sessions,
          }, null, 2),
        },
      ],
    };
  } catch (e) {
    return { content: [{ type: "text", text: `Error listing sessions: ${e}` }] };
  }
}

function getNodeHealth(nodeName?: string) {
  const heartbeatsDir = join(SHM_BASE, "heartbeats");

  if (!existsSync(heartbeatsDir)) {
    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            status: "no_heartbeats",
            message: "No heartbeats directory found - no nodes are running",
          }, null, 2),
        },
      ],
    };
  }

  try {
    let files = readdirSync(heartbeatsDir);
    if (nodeName) {
      files = files.filter((f) => f.includes(nodeName));
    }

    const nodes = files.map((file) => {
      const filePath = join(heartbeatsDir, file);
      const stats = statSync(filePath);

      let heartbeat: Record<string, unknown> = {};
      try {
        const content = readFileSync(filePath, "utf-8");
        heartbeat = JSON.parse(content);
      } catch (e) {
        heartbeat = { parse_error: true };
      }

      const age_ms = Date.now() - stats.mtime.getTime();
      const healthy = age_ms < 2000; // Healthy if heartbeat within 2s

      return {
        name: file.replace(".json", ""),
        healthy,
        age_ms,
        last_heartbeat: stats.mtime.toISOString(),
        ...heartbeat,
      };
    });

    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            heartbeats_dir: heartbeatsDir,
            node_count: nodes.length,
            healthy_count: nodes.filter((n) => n.healthy).length,
            nodes,
          }, null, 2),
        },
      ],
    };
  } catch (e) {
    return { content: [{ type: "text", text: `Error getting node health: ${e}` }] };
  }
}

function tailLogs(lines = 50, filter?: string) {
  const logsPath =
    process.platform === "linux"
      ? "/dev/shm/horus_logs"
      : process.platform === "darwin"
      ? "/tmp/horus_logs"
      : join(process.env.TEMP || "/tmp", "horus_logs");

  if (!existsSync(logsPath)) {
    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            status: "no_logs",
            message: "Log file not found - no HORUS application has written logs",
            path: logsPath,
          }, null, 2),
        },
      ],
    };
  }

  try {
    let content = readFileSync(logsPath, "utf-8");
    let logLines = content.split("\n").filter(Boolean);

    if (filter) {
      logLines = logLines.filter((line) => line.includes(filter));
    }

    logLines = logLines.slice(-lines);

    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            log_path: logsPath,
            line_count: logLines.length,
            filter: filter || null,
            logs: logLines,
          }, null, 2),
        },
      ],
    };
  } catch (e) {
    return { content: [{ type: "text", text: `Error reading logs: ${e}` }] };
  }
}

function checkShm() {
  if (!existsSync(SHM_BASE)) {
    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            status: "not_initialized",
            message: `HORUS shared memory not found at ${SHM_BASE}`,
            hint: "Run a HORUS application to initialize shared memory",
          }, null, 2),
        },
      ],
    };
  }

  try {
    // Get total size
    const result = execSync(`du -sh ${SHM_BASE} 2>/dev/null || echo "0 ${SHM_BASE}"`, {
      encoding: "utf-8",
    });
    const totalSize = result.split("\t")[0].trim();

    // Count files
    const fileCount = execSync(`find ${SHM_BASE} -type f 2>/dev/null | wc -l`, {
      encoding: "utf-8",
    }).trim();

    // List subdirectories
    const subdirs = readdirSync(SHM_BASE)
      .filter((f) => statSync(join(SHM_BASE, f)).isDirectory())
      .map((dir) => {
        const dirPath = join(SHM_BASE, dir);
        const count = readdirSync(dirPath).length;
        return { name: dir, item_count: count };
      });

    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            shm_base: SHM_BASE,
            total_size: totalSize,
            file_count: parseInt(fileCount),
            subdirectories: subdirs,
          }, null, 2),
        },
      ],
    };
  } catch (e) {
    return { content: [{ type: "text", text: `Error checking shm: ${e}` }] };
  }
}

function killSession(sessionId: string) {
  const pidsDir = join(SHM_BASE, "sessions", sessionId, "pids");

  if (!existsSync(pidsDir)) {
    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            status: "not_found",
            session_id: sessionId,
          }, null, 2),
        },
      ],
    };
  }

  try {
    const pids = readdirSync(pidsDir)
      .map((f) => parseInt(f))
      .filter((n) => !isNaN(n));

    const results = pids.map((pid) => {
      try {
        process.kill(pid, "SIGTERM");
        return { pid, status: "killed" };
      } catch (e) {
        return { pid, status: "already_dead" };
      }
    });

    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            session_id: sessionId,
            results,
          }, null, 2),
        },
      ],
    };
  } catch (e) {
    return { content: [{ type: "text", text: `Error killing session: ${e}` }] };
  }
}

function cleanShm(dryRun: boolean) {
  if (!existsSync(SHM_BASE)) {
    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({ status: "nothing_to_clean", shm_base: SHM_BASE }, null, 2),
        },
      ],
    };
  }

  const sessionsDir = join(SHM_BASE, "sessions");
  const toClean: string[] = [];

  if (existsSync(sessionsDir)) {
    const sessionIds = readdirSync(sessionsDir);

    for (const sessionId of sessionIds) {
      const pidsDir = join(sessionsDir, sessionId, "pids");
      let alive = false;

      if (existsSync(pidsDir)) {
        const pids = readdirSync(pidsDir)
          .map((f) => parseInt(f))
          .filter((n) => !isNaN(n));

        for (const pid of pids) {
          try {
            process.kill(pid, 0);
            alive = true;
            break;
          } catch (e) {
            // Process doesn't exist
          }
        }
      }

      if (!alive) {
        toClean.push(sessionId);
      }
    }
  }

  if (!dryRun && toClean.length > 0) {
    for (const sessionId of toClean) {
      try {
        execSync(`rm -rf "${join(sessionsDir, sessionId)}"`, { encoding: "utf-8" });
      } catch (e) {
        // Ignore errors
      }
    }
  }

  return {
    content: [
      {
        type: "text",
        text: JSON.stringify({
          dry_run: dryRun,
          stale_sessions: toClean,
          cleaned: dryRun ? 0 : toClean.length,
        }, null, 2),
      },
    ],
  };
}
