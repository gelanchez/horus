#!/usr/bin/env node
/**
 * HORUS Developer MCP Server
 *
 * Provides tools to boost Claude Code productivity when developing HORUS:
 * - Codebase navigation and understanding
 * - Runtime debugging (topics, nodes, logs)
 * - Benchmark tracking and regression detection
 * - Documentation validation
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  ListResourcesRequestSchema,
  ReadResourceRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";

import { codebaseTools, handleCodebaseTool } from "./tools/codebase.js";
import { runtimeTools, handleRuntimeTool } from "./tools/runtime.js";
import { benchmarkTools, handleBenchmarkTool } from "./tools/benchmark.js";
import { validationTools, handleValidationTool } from "./tools/validation.js";
import { initDatabase } from "./db.js";

// Initialize the MCP server
const server = new Server(
  {
    name: "horus-mcp",
    version: "0.1.0",
  },
  {
    capabilities: {
      tools: {},
      resources: {},
    },
  }
);

// Combine all tools
const allTools = [
  ...codebaseTools,
  ...runtimeTools,
  ...benchmarkTools,
  ...validationTools,
];

// Handle tool listing
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return { tools: allTools };
});

// Handle tool calls
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;
  const toolArgs = args || {};

  try {
    // Route to appropriate handler
    if (codebaseTools.some(t => t.name === name)) {
      return await handleCodebaseTool(name, toolArgs);
    }
    if (runtimeTools.some(t => t.name === name)) {
      return await handleRuntimeTool(name, toolArgs);
    }
    if (benchmarkTools.some(t => t.name === name)) {
      return await handleBenchmarkTool(name, toolArgs);
    }
    if (validationTools.some(t => t.name === name)) {
      return await handleValidationTool(name, toolArgs);
    }

    return {
      content: [{ type: "text", text: `Unknown tool: ${name}` }],
      isError: true,
    };
  } catch (error) {
    return {
      content: [{ type: "text", text: `Error: ${error}` }],
      isError: true,
    };
  }
});

// Handle resource listing (for context about HORUS)
server.setRequestHandler(ListResourcesRequestSchema, async () => {
  return {
    resources: [
      {
        uri: "horus://architecture",
        name: "HORUS Architecture Overview",
        description: "High-level architecture of the HORUS robotics framework",
        mimeType: "text/markdown",
      },
      {
        uri: "horus://crates",
        name: "HORUS Crates",
        description: "List of all crates in the HORUS workspace",
        mimeType: "application/json",
      },
    ],
  };
});

// Handle resource reads
server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
  const { uri } = request.params;

  if (uri === "horus://architecture") {
    return {
      contents: [
        {
          uri,
          mimeType: "text/markdown",
          text: `# HORUS Architecture

## Core Crates
- **horus_core**: Runtime system (nodes, Hub/Link IPC, Scheduler, shared memory)
- **horus_macros**: Procedural macros (node!, message!)
- **horus_manager**: CLI and dashboard
- **horus_library**: Built-in nodes, algorithms, messages

## IPC System
- **Hub<T>**: Multi-producer multi-consumer pub/sub (~481ns latency)
- **Link<T>**: Single-producer single-consumer point-to-point (~248ns latency)
- Shared memory via /dev/shm/horus/ (Linux) or platform equivalents

## Key Abstractions
- **Node**: Trait for computational units with tick() method
- **Scheduler**: Orchestrates node execution with priority levels
- **NodeInfo**: Runtime context passed to nodes (logging, params)

## Directory Structure
- /dev/shm/horus/topics/ - IPC topics
- /dev/shm/horus/heartbeats/ - Node health
- /dev/shm/horus/sessions/ - Active sessions
`,
        },
      ],
    };
  }

  if (uri === "horus://crates") {
    return {
      contents: [
        {
          uri,
          mimeType: "application/json",
          text: JSON.stringify({
            crates: [
              { name: "horus", path: "horus/", description: "Main unified crate" },
              { name: "horus_core", path: "horus_core/", description: "Core runtime" },
              { name: "horus_macros", path: "horus_macros/", description: "Procedural macros" },
              { name: "horus_manager", path: "horus_manager/", description: "CLI and dashboard" },
              { name: "horus_library", path: "horus_library/", description: "Built-in nodes and algorithms" },
              { name: "horus_router", path: "horus_router/", description: "Network router" },
              { name: "horus_py", path: "horus_py/", description: "Python bindings" },
              { name: "horus_benchmarks", path: "benchmarks/", description: "Performance benchmarks" },
            ],
          }, null, 2),
        },
      ],
    };
  }

  return {
    contents: [{ uri, mimeType: "text/plain", text: "Resource not found" }],
  };
});

// Start the server
async function main() {
  // Initialize SQLite database for metrics
  await initDatabase();

  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("HORUS MCP Server running");
}

main().catch(console.error);
