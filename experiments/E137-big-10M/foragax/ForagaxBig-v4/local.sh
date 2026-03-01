#!/usr/bin/env bash
source .venv/bin/activate

EXP=experiments/E137-big-10M/foragax/ForagaxBig-v4/9
export LOCAL_MEM_FRACTION=0.22

python ./scripts/local.py -e $EXP/PPO_LN_128_Conv.json          --runs 1 --entry ./src/rtu_ppo.py --gpu --vmap 1 &
python ./scripts/local.py -e $EXP/PPO-RTU_LN_128_Conv.json      --runs 1 --entry ./src/rtu_ppo.py --gpu --vmap 1 &
python ./scripts/local.py -e $EXP/PPO_LN_128_PConvConv.json     --runs 1 --entry ./src/rtu_ppo.py --gpu --vmap 1 &
python ./scripts/local.py -e $EXP/PPO-RTU_LN_128_PConvConv.json --runs 1 --entry ./src/rtu_ppo.py --gpu --vmap 1 &
wait

echo "ALL DONE"
