rsync -azP vulcan:/home/stang5/scratch/continual-foragax-agents/results .
# TODO: change the lop paths to be identical to ours so it is easier to sync
# move to top level data/results/...
rsync -azP ../continual-foragax-loss-of-plasticity/lop/rl/data/foragax/ForagaxTwoBiomeSmall-v2-15/PPO/data ./results/foragax/ForagaxTwoBiomeSmall-v2-15/PPO
rsync -azP ../continual-foragax-loss-of-plasticity/lop/rl/data/foragax/ForagaxTwoBiomeSmall-v2-15/PPO_CB/data ./results/foragax/ForagaxTwoBiomeSmall-v2-15/PPO_CB
