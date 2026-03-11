#!/usr/bin/env bash
# After SFT v2 finishes: re-run boundary delineation + analysis on the new checkpoint
# Usage: bash scripts/post_sft_analysis.sh

set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate RLforESC
mkdir -p logs data/knowledge_boundaries_v2 results/boundary_analysis_v2

NEW_CKPT="checkpoints/qwen25-esc-v2"
BOUNDARIES="data/knowledge_boundaries_v2/train_turns_with_boundaries.json"

echo "========================================"
echo " Stage 1: Knowledge Boundary Delineation (v2)"
echo "========================================"
PIDS=()
for GPU_ID in 0 1 2 3; do
    python scripts/delineate_knowledge_boundary.py \
        --model_path "$NEW_CKPT" \
        --gpu_id $GPU_ID \
        --num_gpus 4 \
        --output_dir data/knowledge_boundaries_v2 \
        > logs/delineate_v2_gpu${GPU_ID}.log 2>&1 &
    PIDS+=($!)
    echo "  GPU $GPU_ID started (PID ${PIDS[-1]})"
done

for PID in "${PIDS[@]}"; do
    wait $PID && echo "  PID $PID OK" || echo "  PID $PID FAILED"
done

# Re-merge (in case of race)
python - <<'PYEOF'
import json
from pathlib import Path
merged = []
for i in range(4):
    p = Path(f"data/knowledge_boundaries_v2/train_turns_shard{i}.json")
    if p.exists():
        merged.extend(json.load(open(p)))
merged.sort(key=lambda r: (r["dialog_id"], r["turn_idx"]))
out = Path("data/knowledge_boundaries_v2/train_turns_with_boundaries.json")
json.dump(merged, open(out, "w"), ensure_ascii=False, indent=2)
print(f"Merged {len(merged)} turns → {out}")
PYEOF

echo ""
echo "========================================"
echo " Stage 2: Boundary Analysis (v2)"
echo "========================================"
python scripts/analyze_knowledge_boundary.py \
    --boundaries_path "$BOUNDARIES" \
    --output_dir results/boundary_analysis_v2

echo ""
echo "========================================"
echo " Stage 3: Side-by-side Comparison (v1 vs v2)"
echo "========================================"
python - <<'PYEOF'
import json
from pathlib import Path

def load_stats(path):
    with open(path) as f:
        return json.load(f)

v1 = load_stats("results/boundary_analysis/boundary_analysis.json")
v2 = load_stats("results/boundary_analysis_v2/boundary_analysis.json")

print("\n" + "="*60)
print("  SFT v1 (lr=2e-5, epoch=3, maxlen=2048)")
print("  vs")
print("  SFT v2 (lr=1e-5, epoch=2, maxlen=1024)")
print("="*60)

o1, o2 = v1["overall"], v2["overall"]
print(f"\n{'Metric':<30} {'v1':>10} {'v2':>10} {'Δ':>8}")
print("-"*58)
for key in ["HK_pct", "WK_pct", "UK_pct", "mean_ci", "mean_ei"]:
    v = o2[key] - o1[key]
    sign = "▲" if v > 0 else "▼" if v < 0 else "="
    print(f"  {key:<28} {o1[key]:>10.4f} {o2[key]:>10.4f} {sign}{abs(v):>6.4f}")

print("\n  Per-strategy mean_ci:")
print(f"  {'Strategy':<36} {'v1 ci':>8} {'v2 ci':>8} {'Δ':>8}")
print("  " + "-"*60)
strats = list(v1["by_strategy"].keys())
for s in sorted(strats, key=lambda x: v1["by_strategy"][x]["mean_ci"]):
    c1 = v1["by_strategy"].get(s, {}).get("mean_ci", 0)
    c2 = v2["by_strategy"].get(s, {}).get("mean_ci", 0)
    delta = c2 - c1
    sign = "▲" if delta > 0.001 else "▼" if delta < -0.001 else "="
    print(f"  {s:<36} {c1:>8.4f} {c2:>8.4f} {sign}{abs(delta):>6.4f}")

# Save comparison
result = {
    "v1": {"lr": "2e-5", "epochs": 3, "maxlen": 2048, "overall": o1},
    "v2": {"lr": "1e-5", "epochs": 2, "maxlen": 1024, "overall": o2},
}
Path("results").mkdir(exist_ok=True)
with open("results/sft_comparison.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"\nComparison saved → results/sft_comparison.json")
PYEOF

echo ""
echo "All done."
