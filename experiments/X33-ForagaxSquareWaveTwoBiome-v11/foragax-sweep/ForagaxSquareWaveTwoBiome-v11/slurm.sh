for fov in 9;
do
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --time 03:00:00 --runs 10 --entry src/continuing_main.py --force -e experiments/X33-ForagaxSquareWaveTwoBiome-v11/foragax-sweep/ForagaxSquareWaveTwoBiome-v11/${fov}/DQN.json
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --time 03:00:00 --runs 10 --entry src/continuing_main.py --force -e experiments/X33-ForagaxSquareWaveTwoBiome-v11/foragax-sweep/ForagaxSquareWaveTwoBiome-v11/${fov}/DQN_ReDo.json
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --time 03:00:00 --runs 10 --entry src/continuing_main.py --force -e experiments/X33-ForagaxSquareWaveTwoBiome-v11/foragax-sweep/ForagaxSquareWaveTwoBiome-v11/${fov}/DQN_ReDo_PreActLN.json
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --time 03:00:00 --runs 10 --entry src/continuing_main.py --force -e experiments/X33-ForagaxSquareWaveTwoBiome-v11/foragax-sweep/ForagaxSquareWaveTwoBiome-v11/${fov}/DQN_ReDo_PostLNScore.json
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-32G.json --time 03:00:00 --runs 10 --entry src/continuing_main.py --force -e experiments/X33-ForagaxSquareWaveTwoBiome-v11/foragax-sweep/ForagaxSquareWaveTwoBiome-v11/${fov}/DRQN_B32.json
done
