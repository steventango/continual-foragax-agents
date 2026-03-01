for fov in 9;
do
    # python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-24.json --time 03:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v5/${fov}/PPO_LN_128.json
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-24.json --time 03:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v5/${fov}/PPO_LN_HINT_128.json
    # python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-24.json --time 03:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v5/${fov}/PPO_LN_RT_128.json

    # python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-24.json --time 03:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v5/${fov}/PPO-RTU_LN_128.json

    python scripts/slurm.py --cluster clusters/vulcan-cpu.json --time 03:00:00 --runs 30 --entry src/continuing_main.py --force -e experiments/E136-big/foragax/ForagaxBig-v5/${fov}/DQN_LN.json
    python scripts/slurm.py --cluster clusters/vulcan-cpu.json --time 03:00:00 --runs 30 --entry src/continuing_main.py --force -e experiments/E136-big/foragax/ForagaxBig-v5/${fov}/DQN_LN_HINT.json
    python scripts/slurm.py --cluster clusters/vulcan-cpu.json --time 03:00:00 --runs 30 --entry src/continuing_main.py --force -e experiments/E136-big/foragax/ForagaxBig-v5/${fov}/DQN_LN_RT.json
done
