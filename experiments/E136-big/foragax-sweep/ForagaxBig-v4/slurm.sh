for fov in {5,9,15}
do
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-256.json --time 01:00:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax-sweep/ForagaxBig-v4/${fov}/PPO_128.json
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-256.json --time 01:00:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax-sweep/ForagaxBig-v4/${fov}/PPO_LN_128.json
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-256.json --time 01:00:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax-sweep/ForagaxBig-v4/${fov}/PPO_LN_RT_128.json
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-24.json --time 01:00:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax-sweep/ForagaxBig-v4/${fov}/PPO-RTU_128.json
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-24.json --time 01:00:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax-sweep/ForagaxBig-v4/${fov}/PPO-RTU_LN_128.json
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-24.json --time 01:00:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax-sweep/ForagaxBig-v4/${fov}/PPO-RTU_128_512.json
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-24.json --time 01:00:00 --runs 5 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax-sweep/ForagaxBig-v4/${fov}/PPO-RTU_LN_128_512.json
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-256.json --time 03:00:00 --runs 5 --entry src/continuing_main.py --force -e experiments/E136-big/foragax-sweep/ForagaxBig-v4/${fov}/DQN.json
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-256.json --time 03:00:00 --runs 5 --entry src/continuing_main.py --force -e experiments/E136-big/foragax-sweep/ForagaxBig-v4/${fov}/DQN_LN.json
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-256.json --time 03:00:00 --runs 5 --entry src/continuing_main.py --force -e experiments/E136-big/foragax-sweep/ForagaxBig-v4/${fov}/DQN_LN_RT.json
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-256.json --time 03:00:00 --runs 5 --entry src/continuing_main.py --force -e experiments/E136-big/foragax-sweep/ForagaxBig-v4/${fov}/DRQN_1_1.json
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-256.json --time 03:00:00 --runs 5 --entry src/continuing_main.py --force -e experiments/E136-big/foragax-sweep/ForagaxBig-v4/${fov}/DRQN_0_2.json
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-256.json --time 03:00:00 --runs 5 --entry src/continuing_main.py --force -e experiments/E136-big/foragax-sweep/ForagaxBig-v4/${fov}/DRQN_LN_1_1.json
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-256.json --time 03:00:00 --runs 5 --entry src/continuing_main.py --force -e experiments/E136-big/foragax-sweep/ForagaxBig-v4/${fov}/DRQN_LN_0_2.json
done
