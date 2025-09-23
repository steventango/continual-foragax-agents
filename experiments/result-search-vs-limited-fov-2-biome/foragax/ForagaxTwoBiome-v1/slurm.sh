python scripts/slurm.py --cluster clusters/vulcan-cpu-2h.json --runs 30 --entry src/continuing_main.py --force -e experiments/result-search-vs-limited-fov-2-biome/foragax/ForagaxTwoBiome-v1/7/DQN_B1000.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-2h.json --runs 30 --entry src/continuing_main.py --force -e experiments/result-search-vs-limited-fov-2-biome/foragax/ForagaxTwoBiome-v1/7/DQN_B10000.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-2h.json --runs 30 --entry src/continuing_main.py --force -e experiments/result-search-vs-limited-fov-2-biome/foragax/ForagaxTwoBiome-v1/7/DQN_B100000.json

python scripts/slurm.py --cluster clusters/vulcan-cpu-15m.json --runs 30 --entry src/continuing_main.py --force -e experiments/result-search-vs-limited-fov-2-biome/foragax/ForagaxTwoBiome-v1/Baselines/Random.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-30m.json --runs 30 --entry src/continuing_main.py --force -e experiments/result-search-vs-limited-fov-2-biome/foragax/ForagaxTwoBiome-v1/Baselines/Search-Brown-Avoid-Green.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-30m.json --runs 30 --entry src/continuing_main.py --force -e experiments/result-search-vs-limited-fov-2-biome/foragax/ForagaxTwoBiome-v1/Baselines/Search-Brown.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-4h.json --runs 30 --entry src/continuing_main.py --force -e experiments/result-search-vs-limited-fov-2-biome/foragax/ForagaxTwoBiome-v1/Baselines/Search-Morel-Avoid-Green.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-4h.json --runs 30 --entry src/continuing_main.py --force -e experiments/result-search-vs-limited-fov-2-biome/foragax/ForagaxTwoBiome-v1/Baselines/Search-Morel.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-1h.json --runs 30 --entry src/continuing_main.py --force -e experiments/result-search-vs-limited-fov-2-biome/foragax/ForagaxTwoBiome-v1/Baselines/Search-Nearest.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-4h.json --runs 30 --entry src/continuing_main.py --force -e experiments/result-search-vs-limited-fov-2-biome/foragax/ForagaxTwoBiome-v1/Baselines/Search-Oracle.json
python scripts/slurm.py --cluster clusters/vulcan-cpu-1h.json --runs 30 --entry src/continuing_main.py --force -e experiments/result-search-vs-limited-fov-2-biome/foragax/ForagaxTwoBiome-v1/Baselines/Search-Oyster.json
