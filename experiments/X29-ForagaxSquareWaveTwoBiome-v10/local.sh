# python scripts/local.py --runs 5 --entry src/continuing_main.py -e experiments/X29-ForagaxSquareWaveTwoBiome-v10/foragax-sweep/ForagaxSquareWaveTwoBiome-v10/9/DQN_LN_Conv_Color.json --gpu --vmap 30
# python scripts/local.py --runs 5 --entry src/continuing_main.py -e experiments/X29-ForagaxSquareWaveTwoBiome-v10/foragax-sweep/ForagaxSquareWaveTwoBiome-v10/9/DQN_LN_Conv_RGB.json --gpu --vmap 30
# python scripts/local.py --runs 5 --entry src/continuing_main.py -e experiments/X29-ForagaxSquareWaveTwoBiome-v10/foragax-sweep/ForagaxSquareWaveTwoBiome-v10/9/DQN_LN_CoordConv_RGB.json --gpu --vmap 30


# python scripts/local.py --runs 5 --entry src/continuing_main.py -e experiments/X29-ForagaxSquareWaveTwoBiome-v10/foragax-sweep/ForagaxSquareWaveTwoBiome-v10/9/DQN_LN_MLP_Color.json --gpu --vmap 30
python scripts/local.py --runs 5 --entry src/continuing_main.py -e experiments/X29-ForagaxSquareWaveTwoBiome-v10/foragax-sweep/ForagaxSquareWaveTwoBiome-v10/9/DQN_LN_MLP_RGB.json --gpu --vmap 30 &
python scripts/local.py --runs 5 --entry src/continuing_main.py -e experiments/X29-ForagaxSquareWaveTwoBiome-v10/foragax-sweep/ForagaxSquareWaveTwoBiome-v10/9/DQN_LN_PConv_RGB.json --gpu --vmap 30 &
python scripts/local.py --runs 5 --entry src/continuing_main.py -e experiments/X29-ForagaxSquareWaveTwoBiome-v10/foragax-sweep/ForagaxSquareWaveTwoBiome-v10/9/DQN_LN_PConvConv_RGB.json --gpu --vmap 30 &
wait

# python ./scripts/local.py -e experiments/X29-ForagaxSquareWaveTwoBiome-v10/foragax-sweep/ForagaxSquareWaveTwoBiome-v10/9/ActorCriticConv-reward-trace.json --runs 5 --entry ./src/rtu_ppo.py  --gpu --vmap 180
# python ./scripts/local.py -e experiments/X29-ForagaxSquareWaveTwoBiome-v10/foragax-sweep/ForagaxSquareWaveTwoBiome-v10/9/ActorCriticMLP-reward-trace.json --runs 5 --entry ./src/rtu_ppo.py  --gpu --vmap 180
