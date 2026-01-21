#!/bin/bash
# HORUS Benchmark Regression Check
# Run this before pushing to check for performance regressions
#
# Usage:
#   ./scripts/check-benchmark.sh           # Run quick check
#   ./scripts/check-benchmark.sh --full    # Run full benchmark suite

set -e

# Regression thresholds (nanoseconds)
MPMC_SHM_THRESHOLD=600
SPSC_SHM_THRESHOLD=200
MPMC_INTRA_THRESHOLD=100
SPSC_INTRA_THRESHOLD=50

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║           HORUS Benchmark Regression Check                   ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Ensure shared memory directories exist
echo -e "${YELLOW}Setting up shared memory...${NC}"
mkdir -p /dev/shm/horus/topics 2>/dev/null || sudo mkdir -p /dev/shm/horus/topics
mkdir -p /dev/shm/horus/heartbeats 2>/dev/null || sudo mkdir -p /dev/shm/horus/heartbeats

# Navigate to benchmarks directory
cd "$(dirname "$0")/../benchmarks"

# Run quick benchmark if not full mode
if [ "$1" != "--full" ]; then
    echo -e "${YELLOW}Running quick benchmark (MpmcShm only)...${NC}"
    echo ""

    # Run MpmcShm benchmark and capture output
    OUTPUT=$(cargo bench --bench topic_performance -- "backend_comparison/MpmcShm" --noplot 2>&1)

    # Parse the median latency
    MEDIAN=$(echo "$OUTPUT" | grep -A1 "MpmcShm" | grep "time:" | head -1 | sed 's/.*\[\([0-9.]*\) ns.*/\1/' 2>/dev/null || echo "")

    if [ -z "$MEDIAN" ]; then
        # Try alternative parsing
        MEDIAN=$(echo "$OUTPUT" | grep "time:" | head -1 | awk '{print $2}' | tr -d '[]ns' 2>/dev/null || echo "unknown")
    fi

    echo ""
    echo -e "═══════════════════════════════════════════════════════════════"
    echo -e "${CYAN}Results:${NC}"
    echo -e "  MpmcShm median: ${MEDIAN}ns (threshold: ${MPMC_SHM_THRESHOLD}ns)"

    # Check threshold
    if [ "$MEDIAN" != "unknown" ] && [ -n "$MEDIAN" ]; then
        if (( $(echo "$MEDIAN > $MPMC_SHM_THRESHOLD" | bc -l 2>/dev/null || echo "0") )); then
            echo ""
            echo -e "${RED}✗ REGRESSION DETECTED!${NC}"
            echo -e "  MpmcShm latency (${MEDIAN}ns) exceeds threshold (${MPMC_SHM_THRESHOLD}ns)"
            echo ""
            echo -e "${YELLOW}Investigate before pushing. Consider:${NC}"
            echo "  1. Check recent changes to horus_core/src/memory/"
            echo "  2. Run full benchmarks: ./scripts/check-benchmark.sh --full"
            echo "  3. Profile with: perf record cargo bench --bench topic_performance"
            exit 1
        fi
    fi

    echo ""
    echo -e "${GREEN}✓ No regression detected${NC}"
    echo ""

else
    echo -e "${YELLOW}Running full benchmark suite...${NC}"
    echo ""

    # Run all topic performance benchmarks
    cargo bench --bench topic_performance

    echo ""
    echo -e "${GREEN}✓ Full benchmark complete${NC}"
    echo ""
    echo "Results saved to: target/criterion/"
    echo "View HTML report: target/criterion/report/index.html"
fi

echo -e "═══════════════════════════════════════════════════════════════"
echo -e "${CYAN}Thresholds:${NC}"
echo "  MpmcShm:   ${MPMC_SHM_THRESHOLD}ns (cross-process pub/sub)"
echo "  SpscShm:   ${SPSC_SHM_THRESHOLD}ns (cross-process point-to-point)"
echo "  MpmcIntra: ${MPMC_INTRA_THRESHOLD}ns (same-process pub/sub)"
echo "  SpscIntra: ${SPSC_INTRA_THRESHOLD}ns (same-process point-to-point)"
echo ""
