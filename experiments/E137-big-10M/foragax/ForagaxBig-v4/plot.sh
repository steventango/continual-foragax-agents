#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."
source .venv/bin/activate

EXP=experiments/E137-big-10M/foragax/ForagaxBig-v4

python src/process_data.py "$EXP"

# fig_e137_conv_comparison: PPO and PPO-RTU LN 128 Conv/PConvConv variants
for fmt in pdf png; do
  MPLBACKEND=Agg python src/learning_curve.py "$EXP" \
    --filter-alg-apertures PPO_LN_128_Conv:9 PPO_LN_128_PConvConv:9 PPO-RTU_LN_128_Conv:9 PPO-RTU_LN_128_PConvConv:9 \
    --metrics rolling_reward_10000 \
    --sample-type every \
    --plot-name fig_e137_conv_comparison \
    --legend \
    --save-type "$fmt"
done

echo "Done. Plots saved to $EXP/plots/"
