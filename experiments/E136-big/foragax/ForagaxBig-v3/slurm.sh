for fov in {5,7,9,15}
do
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-24.json --time 03:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v3/${fov}/PPO_128.json
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-24.json --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v3/${fov}/PPO_128_frozen_1M.json
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-24.json --time 02:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v3/${fov}/PPO_128_frozen_5M.json

    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-24.json --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v3/${fov}/PPO_128.json
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-24.json --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v3/${fov}/PPO_128_frozen_1M.json
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-24.json --time 02:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v3/${fov}/PPO_128_frozen_5M.json

    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-24.json --time 03:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v3/${fov}/PPO-RTU_128.json
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-24.json --time 03:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v3/${fov}/PPO-RTU_128_frozen_1M.json
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-24.json --time 03:00:00 --runs 30 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v3/${fov}/PPO-RTU_128_frozen_5M.json

    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-24.json --time 03:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v3/${fov}/PPO-RTU_128.json
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-24.json --time 03:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v3/${fov}/PPO-RTU_128_frozen_1M.json
    python scripts/slurm.py --cluster clusters/vulcan-gpu-vmap-24.json --time 03:00:00 --runs 1 --entry src/rtu_ppo.py --force -e experiments/E136-big/foragax/ForagaxBig-v3/${fov}/PPO-RTU_128_frozen_5M.json
done