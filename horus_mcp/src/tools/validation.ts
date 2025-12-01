/**
 * Documentation and code validation tools
 *
 * These tools validate the HORUS codebase:
 * - Check documentation coverage
 * - Validate code samples in docs
 * - Check for broken links
 * - Verify API consistency
 */

import { Tool } from "@modelcontextprotocol/sdk/types.js";
import { execSync } from "child_process";
import { readFileSync, existsSync, readdirSync } from "fs";
import { join, basename } from "path";
import { glob } from "glob";

function getHorusRoot(): string {
  return process.env.HORUS_ROOT || "/home/lord-patpak/horus/HORUS";
}

export const validationTools: Tool[] = [
  {
    name: "horus_check_docs",
    description:
      "Check documentation for issues: broken internal links, outdated code samples, missing pages.",
    inputSchema: {
      type: "object",
      properties: {
        check_type: {
          type: "string",
          enum: ["links", "code_samples", "coverage", "all"],
          description: "Type of check to perform (default: all)",
        },
      },
    },
  },
  {
    name: "horus_check_api_coverage",
    description:
      "Check which public APIs are documented and which are missing documentation.",
    inputSchema: {
      type: "object",
      properties: {
        crate_name: {
          type: "string",
          description: "Crate to check (default: all crates)",
        },
      },
    },
  },
  {
    name: "horus_validate_cargo",
    description:
      "Run cargo check and clippy to find compilation errors and warnings.",
    inputSchema: {
      type: "object",
      properties: {
        fix: {
          type: "boolean",
          description: "Whether to apply automatic fixes (default: false)",
        },
      },
    },
  },
  {
    name: "horus_check_tests",
    description:
      "Run the test suite and report results.",
    inputSchema: {
      type: "object",
      properties: {
        package: {
          type: "string",
          description: "Specific package to test (default: all)",
        },
        filter: {
          type: "string",
          description: "Test name filter",
        },
      },
    },
  },
  {
    name: "horus_check_formatting",
    description:
      "Check if code is properly formatted with rustfmt.",
    inputSchema: {
      type: "object",
      properties: {
        fix: {
          type: "boolean",
          description: "Whether to apply formatting fixes (default: false)",
        },
      },
    },
  },
  {
    name: "horus_find_todos",
    description:
      "Find TODO, FIXME, HACK, and XXX comments in the codebase.",
    inputSchema: {
      type: "object",
      properties: {
        type: {
          type: "string",
          enum: ["TODO", "FIXME", "HACK", "XXX", "all"],
          description: "Type of comment to find (default: all)",
        },
      },
    },
  },
  {
    name: "horus_check_dead_code",
    description:
      "Find potentially dead code using cargo's dead_code warnings.",
    inputSchema: {
      type: "object",
      properties: {},
    },
  },
];

export async function handleValidationTool(
  name: string,
  args: Record<string, unknown>
): Promise<{ content: Array<{ type: string; text: string }> }> {
  switch (name) {
    case "horus_check_docs":
      return checkDocs(args.check_type as string);
    case "horus_check_api_coverage":
      return checkApiCoverage(args.crate_name as string);
    case "horus_validate_cargo":
      return validateCargo(args.fix as boolean ?? false);
    case "horus_check_tests":
      return checkTests(args.package as string, args.filter as string);
    case "horus_check_formatting":
      return checkFormatting(args.fix as boolean ?? false);
    case "horus_find_todos":
      return findTodos(args.type as string);
    case "horus_check_dead_code":
      return checkDeadCode();
    default:
      return { content: [{ type: "text", text: `Unknown tool: ${name}` }] };
  }
}

async function checkDocs(checkType?: string) {
  const root = getHorusRoot();
  const docsDir = join(root, "docs-site", "content", "docs");
  const checks = checkType === "all" || !checkType ? ["links", "code_samples", "coverage"] : [checkType];

  const results: Record<string, unknown> = {};

  if (checks.includes("links")) {
    // Find all internal links and check if target exists
    try {
      const mdxFiles = await glob(`${docsDir}/**/*.mdx`);
      const brokenLinks: Array<{ file: string; link: string }> = [];

      for (const file of mdxFiles) {
        const content = readFileSync(file, "utf-8");
        const linkMatches = content.match(/\]\(\/[a-z0-9/-]+\)/g) || [];

        for (const match of linkMatches) {
          const link = match.slice(2, -1); // Remove ]( and )
          const targetPath = join(docsDir, link);

          // Check various possible paths
          const exists =
            existsSync(`${targetPath}.mdx`) ||
            existsSync(join(targetPath, "index.mdx")) ||
            existsSync(targetPath);

          if (!exists && !link.includes("#") && !link.startsWith("/dev/")) {
            brokenLinks.push({
              file: file.replace(root + "/", ""),
              link,
            });
          }
        }
      }

      results.links = {
        total_files: mdxFiles.length,
        broken_links: brokenLinks.slice(0, 20),
        broken_count: brokenLinks.length,
      };
    } catch (e) {
      results.links = { error: String(e) };
    }
  }

  if (checks.includes("code_samples")) {
    // Find rust code blocks and check basic syntax
    try {
      const mdxFiles = await glob(`${docsDir}/**/*.mdx`);
      const issues: Array<{ file: string; issue: string }> = [];

      for (const file of mdxFiles) {
        const content = readFileSync(file, "utf-8");
        const codeBlocks = content.match(/```rust[\s\S]*?```/g) || [];

        for (const block of codeBlocks) {
          // Check for common issues
          if (block.includes("unimplemented!()") && !block.includes("todo!")) {
            issues.push({
              file: file.replace(root + "/", ""),
              issue: "Contains unimplemented!() - may be incomplete",
            });
          }
          if (block.includes("unwrap()") && !block.includes("// ")) {
            // Only flag if no comment explaining it
            // This is a heuristic, not perfect
          }
        }
      }

      results.code_samples = {
        total_files: mdxFiles.length,
        potential_issues: issues.slice(0, 10),
        issue_count: issues.length,
      };
    } catch (e) {
      results.code_samples = { error: String(e) };
    }
  }

  if (checks.includes("coverage")) {
    // Check for key pages that should exist
    const requiredPages = [
      "getting-started/installation.mdx",
      "getting-started/quick-start.mdx",
      "core-concepts/core-concepts-nodes.mdx",
      "core-concepts/core-concepts-hub.mdx",
      "core-concepts/core-concepts-scheduler.mdx",
      "api/core.mdx",
      "troubleshooting.mdx",
    ];

    const missing = requiredPages.filter((page) => !existsSync(join(docsDir, page)));

    results.coverage = {
      required_pages: requiredPages.length,
      missing_pages: missing,
      coverage_percent: ((requiredPages.length - missing.length) / requiredPages.length * 100).toFixed(1),
    };
  }

  return {
    content: [
      {
        type: "text",
        text: JSON.stringify({ docs_dir: docsDir, checks: results }, null, 2),
      },
    ],
  };
}

function checkApiCoverage(crateName?: string) {
  const root = getHorusRoot();
  const crates = crateName ? [crateName] : ["horus_core", "horus_macros", "horus_library"];

  const results: Record<string, unknown> = {};

  for (const crate of crates) {
    const libRs = join(root, crate, "src", "lib.rs");
    if (!existsSync(libRs)) continue;

    try {
      // Find public items
      const content = readFileSync(libRs, "utf-8");
      const pubItems = content.match(/pub (struct|trait|fn|enum|type|mod) \w+/g) || [];

      // Find documented items (with /// or //!)
      const docItems = content.match(/\/\/[\/!].*\n\s*pub (struct|trait|fn|enum|type|mod) \w+/g) || [];

      results[crate] = {
        public_items: pubItems.length,
        documented_items: docItems.length,
        coverage_percent: pubItems.length > 0
          ? ((docItems.length / pubItems.length) * 100).toFixed(1)
          : "N/A",
        undocumented: pubItems
          .filter((item) => !docItems.some((doc) => doc.includes(item)))
          .slice(0, 10),
      };
    } catch (e) {
      results[crate] = { error: String(e) };
    }
  }

  return {
    content: [{ type: "text", text: JSON.stringify({ api_coverage: results }, null, 2) }],
  };
}

function validateCargo(fix: boolean) {
  const root = getHorusRoot();

  try {
    // Run cargo check
    const checkResult = execSync(
      `cargo check --workspace 2>&1 || true`,
      { cwd: root, encoding: "utf-8", maxBuffer: 10 * 1024 * 1024 }
    );

    // Run clippy
    const clippyCmd = fix
      ? "cargo clippy --workspace --fix --allow-dirty 2>&1 || true"
      : "cargo clippy --workspace 2>&1 || true";
    const clippyResult = execSync(clippyCmd, {
      cwd: root,
      encoding: "utf-8",
      maxBuffer: 10 * 1024 * 1024,
    });

    // Parse errors and warnings
    const errors = (checkResult.match(/error\[E\d+\]/g) || []).length;
    const warnings = (clippyResult.match(/warning:/g) || []).length;

    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            status: errors === 0 ? "ok" : "errors",
            errors,
            warnings,
            fix_applied: fix,
            check_output: checkResult.slice(-2000),
            clippy_output: clippyResult.slice(-2000),
          }, null, 2),
        },
      ],
    };
  } catch (e) {
    return { content: [{ type: "text", text: `Error: ${e}` }] };
  }
}

function checkTests(packageName?: string, filter?: string) {
  const root = getHorusRoot();

  let cmd = "cargo test --workspace";
  if (packageName) cmd = `cargo test -p ${packageName}`;
  if (filter) cmd += ` ${filter}`;
  cmd += " -- --nocapture 2>&1";

  try {
    const result = execSync(cmd, {
      cwd: root,
      encoding: "utf-8",
      maxBuffer: 20 * 1024 * 1024,
      timeout: 300000, // 5 minutes
    });

    // Parse test results
    const passedMatch = result.match(/(\d+) passed/);
    const failedMatch = result.match(/(\d+) failed/);
    const ignoredMatch = result.match(/(\d+) ignored/);

    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            status: failedMatch ? "failed" : "passed",
            passed: passedMatch ? parseInt(passedMatch[1]) : 0,
            failed: failedMatch ? parseInt(failedMatch[1]) : 0,
            ignored: ignoredMatch ? parseInt(ignoredMatch[1]) : 0,
            output: result.slice(-3000),
          }, null, 2),
        },
      ],
    };
  } catch (e: unknown) {
    const err = e as { stdout?: string; stderr?: string };
    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            status: "error",
            error: String(e),
            output: (err.stdout || "") + (err.stderr || ""),
          }, null, 2),
        },
      ],
    };
  }
}

function checkFormatting(fix: boolean) {
  const root = getHorusRoot();

  const cmd = fix ? "cargo fmt" : "cargo fmt --check";

  try {
    const result = execSync(`${cmd} 2>&1`, {
      cwd: root,
      encoding: "utf-8",
    });

    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            status: "formatted",
            fix_applied: fix,
            output: result || "All files properly formatted",
          }, null, 2),
        },
      ],
    };
  } catch (e: unknown) {
    const err = e as { stdout?: string };
    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            status: "needs_formatting",
            fix_applied: fix,
            files_needing_format: (err.stdout || "").split("\n").filter(Boolean).slice(0, 20),
          }, null, 2),
        },
      ],
    };
  }
}

function findTodos(type?: string) {
  const root = getHorusRoot();
  const patterns = type === "all" || !type
    ? ["TODO", "FIXME", "HACK", "XXX"]
    : [type];

  const results: Record<string, Array<{ file: string; line: number; text: string }>> = {};

  for (const pattern of patterns) {
    try {
      const output = execSync(
        `grep -rn "${pattern}" --include="*.rs" ${root}/horus* ${root}/benchmarks 2>/dev/null | head -50`,
        { encoding: "utf-8", maxBuffer: 5 * 1024 * 1024 }
      );

      results[pattern] = output
        .split("\n")
        .filter(Boolean)
        .map((line) => {
          const parts = line.split(":");
          return {
            file: parts[0].replace(root + "/", ""),
            line: parseInt(parts[1]),
            text: parts.slice(2).join(":").trim().slice(0, 100),
          };
        });
    } catch (e) {
      results[pattern] = [];
    }
  }

  const totalCount = Object.values(results).reduce((a, b) => a + b.length, 0);

  return {
    content: [
      {
        type: "text",
        text: JSON.stringify({
          total_count: totalCount,
          by_type: results,
        }, null, 2),
      },
    ],
  };
}

function checkDeadCode() {
  const root = getHorusRoot();

  try {
    const result = execSync(
      `RUSTFLAGS="-Wdead_code" cargo check --workspace 2>&1 | grep "warning: " | head -30`,
      { cwd: root, encoding: "utf-8", maxBuffer: 10 * 1024 * 1024 }
    );

    const warnings = result.split("\n").filter(Boolean);

    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            dead_code_warnings: warnings.length,
            warnings: warnings.slice(0, 20),
          }, null, 2),
        },
      ],
    };
  } catch (e) {
    return { content: [{ type: "text", text: `Error checking dead code: ${e}` }] };
  }
}
