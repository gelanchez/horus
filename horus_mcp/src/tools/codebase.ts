/**
 * Codebase navigation and understanding tools
 *
 * These tools help Claude quickly understand the HORUS codebase structure,
 * find relevant files, and understand module relationships.
 */

import { Tool } from "@modelcontextprotocol/sdk/types.js";
import { execSync } from "child_process";
import { readFileSync, existsSync, readdirSync, statSync } from "fs";
import { join, basename, dirname } from "path";
import { glob } from "glob";

// Detect HORUS root directory
function getHorusRoot(): string {
  // Check common locations
  const candidates = [
    process.env.HORUS_ROOT,
    "/home/lord-patpak/horus/HORUS",
    join(process.cwd(), ".."),
    process.cwd(),
  ].filter(Boolean) as string[];

  for (const candidate of candidates) {
    if (existsSync(join(candidate, "Cargo.toml"))) {
      const content = readFileSync(join(candidate, "Cargo.toml"), "utf-8");
      if (content.includes("horus_core") || content.includes("[workspace]")) {
        return candidate;
      }
    }
  }

  return "/home/lord-patpak/horus/HORUS";
}

const HORUS_ROOT = getHorusRoot();

export const codebaseTools: Tool[] = [
  {
    name: "horus_get_architecture",
    description:
      "Get a high-level overview of HORUS architecture including all crates, their purposes, and dependencies. Use this first when starting work on HORUS.",
    inputSchema: {
      type: "object",
      properties: {},
    },
  },
  {
    name: "horus_find_module",
    description:
      "Find where a specific module, struct, trait, or function is defined in HORUS. Much faster than grep for known HORUS concepts.",
    inputSchema: {
      type: "object",
      properties: {
        name: {
          type: "string",
          description: "Name to search for (e.g., 'Hub', 'Node', 'Scheduler', 'Link')",
        },
        type: {
          type: "string",
          enum: ["struct", "trait", "fn", "mod", "any"],
          description: "Type of item to find (default: any)",
        },
      },
      required: ["name"],
    },
  },
  {
    name: "horus_get_crate_info",
    description:
      "Get detailed information about a specific HORUS crate including its modules, dependencies, and public API.",
    inputSchema: {
      type: "object",
      properties: {
        crate_name: {
          type: "string",
          description: "Crate name (e.g., 'horus_core', 'horus_manager', 'horus_library')",
        },
      },
      required: ["crate_name"],
    },
  },
  {
    name: "horus_list_nodes",
    description:
      "List all built-in node implementations in horus_library with their file locations.",
    inputSchema: {
      type: "object",
      properties: {
        category: {
          type: "string",
          description: "Optional category filter (e.g., 'sensors', 'motors', 'control')",
        },
      },
    },
  },
  {
    name: "horus_get_message_types",
    description:
      "List all message types defined in HORUS for IPC communication.",
    inputSchema: {
      type: "object",
      properties: {},
    },
  },
  {
    name: "horus_find_tests",
    description:
      "Find test files and test functions related to a specific module or feature.",
    inputSchema: {
      type: "object",
      properties: {
        query: {
          type: "string",
          description: "Module or feature to find tests for",
        },
      },
      required: ["query"],
    },
  },
  {
    name: "horus_get_dependencies",
    description:
      "Get the dependency graph for a crate or show what depends on a specific crate.",
    inputSchema: {
      type: "object",
      properties: {
        crate_name: {
          type: "string",
          description: "Crate to analyze",
        },
        direction: {
          type: "string",
          enum: ["deps", "rdeps"],
          description: "deps = what this crate depends on, rdeps = what depends on this crate",
        },
      },
      required: ["crate_name"],
    },
  },
];

export async function handleCodebaseTool(
  name: string,
  args: Record<string, unknown>
): Promise<{ content: Array<{ type: string; text: string }> }> {
  switch (name) {
    case "horus_get_architecture":
      return getArchitecture();
    case "horus_find_module":
      return findModule(args.name as string, args.type as string);
    case "horus_get_crate_info":
      return getCrateInfo(args.crate_name as string);
    case "horus_list_nodes":
      return listNodes(args.category as string | undefined);
    case "horus_get_message_types":
      return getMessageTypes();
    case "horus_find_tests":
      return findTests(args.query as string);
    case "horus_get_dependencies":
      return getDependencies(args.crate_name as string, args.direction as string);
    default:
      return { content: [{ type: "text", text: `Unknown tool: ${name}` }] };
  }
}

function getArchitecture() {
  const architecture = {
    root: HORUS_ROOT,
    crates: [
      {
        name: "horus",
        path: "horus/",
        description: "Main unified crate - re-exports from all other crates",
        key_exports: ["prelude::*"],
      },
      {
        name: "horus_core",
        path: "horus_core/",
        description: "Core runtime system",
        key_exports: ["Node", "Hub", "Link", "Scheduler", "NodeInfo", "HorusResult"],
        modules: ["communication", "core", "memory", "scheduling", "error", "params"],
      },
      {
        name: "horus_macros",
        path: "horus_macros/",
        description: "Procedural macros for reducing boilerplate",
        key_exports: ["node!", "message!"],
      },
      {
        name: "horus_manager",
        path: "horus_manager/",
        description: "CLI tool and dashboard",
        commands: ["run", "new", "init", "pkg", "monitor", "dashboard", "sim2d", "sim3d"],
      },
      {
        name: "horus_library",
        path: "horus_library/",
        description: "Built-in nodes, algorithms, and message types",
        subdirs: ["nodes/", "algorithms/", "messages/", "tools/"],
      },
      {
        name: "horus_benchmarks",
        path: "benchmarks/",
        description: "Performance benchmarks",
        benchmarks: ["link_performance", "production_messages", "network_transport"],
      },
    ],
    ipc_system: {
      hub: {
        description: "Multi-producer multi-consumer pub/sub",
        latency: "~481ns round-trip",
        file: "horus_core/src/communication/hub.rs",
      },
      link: {
        description: "Single-producer single-consumer point-to-point",
        latency: "~248ns round-trip",
        file: "horus_core/src/communication/link.rs",
      },
      shared_memory: {
        linux: "/dev/shm/horus/",
        macos: "/tmp/horus/",
        windows: "%TEMP%/horus/",
      },
    },
    cli_entry: "horus_manager/src/main.rs",
  };

  return {
    content: [
      {
        type: "text",
        text: JSON.stringify(architecture, null, 2),
      },
    ],
  };
}

function findModule(name: string, itemType?: string) {
  const patterns: Record<string, string> = {
    struct: `pub struct ${name}`,
    trait: `pub trait ${name}`,
    fn: `pub fn ${name}`,
    mod: `pub mod ${name}`,
    any: `${name}`,
  };

  const pattern = patterns[itemType || "any"];

  try {
    const result = execSync(
      `grep -rn "${pattern}" --include="*.rs" ${HORUS_ROOT}/horus* ${HORUS_ROOT}/benchmarks 2>/dev/null | grep -v target | head -30`,
      { encoding: "utf-8", maxBuffer: 10 * 1024 * 1024 }
    );

    const matches = result
      .split("\n")
      .filter(Boolean)
      .map((line) => {
        const [filePath, ...rest] = line.split(":");
        const lineNum = rest[0];
        const content = rest.slice(1).join(":").trim();
        return {
          file: filePath.replace(HORUS_ROOT + "/", ""),
          line: parseInt(lineNum),
          content: content.substring(0, 100),
        };
      });

    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({ query: name, type: itemType || "any", matches }, null, 2),
        },
      ],
    };
  } catch (e) {
    return {
      content: [{ type: "text", text: `No matches found for '${name}'` }],
    };
  }
}

function getCrateInfo(crateName: string) {
  const cratePath = join(HORUS_ROOT, crateName);
  const cargoToml = join(cratePath, "Cargo.toml");

  if (!existsSync(cargoToml)) {
    return { content: [{ type: "text", text: `Crate '${crateName}' not found` }] };
  }

  const cargoContent = readFileSync(cargoToml, "utf-8");

  // Get source files
  const srcPath = join(cratePath, "src");
  let modules: string[] = [];
  if (existsSync(srcPath)) {
    modules = readdirSync(srcPath)
      .filter((f) => f.endsWith(".rs") || statSync(join(srcPath, f)).isDirectory())
      .map((f) => f.replace(".rs", ""));
  }

  // Get lib.rs exports
  const libRs = join(srcPath, "lib.rs");
  let exports: string[] = [];
  if (existsSync(libRs)) {
    const content = readFileSync(libRs, "utf-8");
    const pubUseMatches = content.match(/pub use [^;]+;/g) || [];
    exports = pubUseMatches.map((m) => m.replace("pub use ", "").replace(";", ""));
  }

  // Parse dependencies from Cargo.toml
  const depMatches = cargoContent.match(/\[dependencies\]([\s\S]*?)(\[|$)/);
  let deps: string[] = [];
  if (depMatches) {
    deps = depMatches[1]
      .split("\n")
      .filter((line) => line.includes("="))
      .map((line) => line.split("=")[0].trim());
  }

  return {
    content: [
      {
        type: "text",
        text: JSON.stringify(
          {
            name: crateName,
            path: cratePath.replace(HORUS_ROOT + "/", ""),
            modules,
            exports: exports.slice(0, 20),
            dependencies: deps,
          },
          null,
          2
        ),
      },
    ],
  };
}

function listNodes(category?: string) {
  const nodesPath = join(HORUS_ROOT, "horus_library", "nodes");

  if (!existsSync(nodesPath)) {
    // Try alternative path
    const altPath = join(HORUS_ROOT, "horus_library", "src", "nodes");
    if (!existsSync(altPath)) {
      return { content: [{ type: "text", text: "Nodes directory not found" }] };
    }
  }

  try {
    const result = execSync(
      `find ${HORUS_ROOT}/horus_library -name "*.rs" -path "*/nodes/*" | head -50`,
      { encoding: "utf-8" }
    );

    const nodes = result
      .split("\n")
      .filter(Boolean)
      .map((path) => ({
        name: basename(path, ".rs"),
        path: path.replace(HORUS_ROOT + "/", ""),
        category: dirname(path).split("/").pop(),
      }))
      .filter((n) => !category || n.category?.includes(category));

    return {
      content: [{ type: "text", text: JSON.stringify({ nodes }, null, 2) }],
    };
  } catch (e) {
    return { content: [{ type: "text", text: `Error listing nodes: ${e}` }] };
  }
}

function getMessageTypes() {
  const messagesPath = join(HORUS_ROOT, "horus_library", "messages");

  try {
    const result = execSync(
      `grep -rn "pub struct" --include="*.rs" ${HORUS_ROOT}/horus_library/messages ${HORUS_ROOT}/horus_library/src/messages 2>/dev/null || true`,
      { encoding: "utf-8" }
    );

    const messages = result
      .split("\n")
      .filter(Boolean)
      .map((line) => {
        const match = line.match(/pub struct (\w+)/);
        return match ? match[1] : null;
      })
      .filter(Boolean);

    return {
      content: [
        { type: "text", text: JSON.stringify({ message_types: [...new Set(messages)] }, null, 2) },
      ],
    };
  } catch (e) {
    return { content: [{ type: "text", text: `Error getting messages: ${e}` }] };
  }
}

function findTests(query: string) {
  try {
    const result = execSync(
      `grep -rn "#\\[test\\]" --include="*.rs" -A 2 ${HORUS_ROOT} | grep -i "${query}" | head -20`,
      { encoding: "utf-8", maxBuffer: 10 * 1024 * 1024 }
    );

    const tests = result
      .split("\n")
      .filter(Boolean)
      .map((line) => {
        const [filePath, content] = line.split(":", 2);
        return { file: filePath?.replace(HORUS_ROOT + "/", ""), content: content?.trim() };
      });

    return {
      content: [{ type: "text", text: JSON.stringify({ query, tests }, null, 2) }],
    };
  } catch (e) {
    return { content: [{ type: "text", text: `No tests found for '${query}'` }] };
  }
}

function getDependencies(crateName: string, direction: string) {
  try {
    if (direction === "rdeps") {
      // What depends on this crate
      const result = execSync(
        `grep -rn "${crateName}" --include="Cargo.toml" ${HORUS_ROOT} | grep -v "^${HORUS_ROOT}/${crateName}"`,
        { encoding: "utf-8" }
      );

      const dependents = result
        .split("\n")
        .filter((line) => line.includes("[dependencies]") === false && line.includes(crateName))
        .map((line) => line.split(":")[0].replace(HORUS_ROOT + "/", "").replace("/Cargo.toml", ""));

      return {
        content: [
          {
            type: "text",
            text: JSON.stringify({ crate: crateName, dependents: [...new Set(dependents)] }, null, 2),
          },
        ],
      };
    } else {
      // What this crate depends on
      const cargoToml = join(HORUS_ROOT, crateName, "Cargo.toml");
      if (!existsSync(cargoToml)) {
        return { content: [{ type: "text", text: `Crate '${crateName}' not found` }] };
      }

      const content = readFileSync(cargoToml, "utf-8");
      const deps = content
        .split("\n")
        .filter((line) => line.match(/^[a-z_]+ = /))
        .map((line) => line.split("=")[0].trim());

      return {
        content: [{ type: "text", text: JSON.stringify({ crate: crateName, dependencies: deps }, null, 2) }],
      };
    }
  } catch (e) {
    return { content: [{ type: "text", text: `Error getting dependencies: ${e}` }] };
  }
}
