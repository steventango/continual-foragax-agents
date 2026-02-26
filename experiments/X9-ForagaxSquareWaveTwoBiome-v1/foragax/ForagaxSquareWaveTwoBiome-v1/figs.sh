python src/learning_curve.py experiments/X8-ForagaxSquareWaveTwoBiome-v1/foragax/ForagaxSquareWaveTwoBiome-v1 --plot-name rtu --filter-alg-apertures Search-Oracle RealTimeActorCriticMLP:5 RealTimeActorCriticMLP:9 RealTimeActorCriticMLP:15 --legend
python src/learning_curve.py experiments/X8-ForagaxSquareWaveTwoBiome-v1/foragax/ForagaxSquareWaveTwoBiome-v1 --plot-name rtu-normalized --filter-alg-apertures Search-Oracle RealTimeActorCriticMLP:5 RealTimeActorCriticMLP:9 RealTimeActorCriticMLP:15 --normalize Search-Oracle --legend

python src/learning_curve.py experiments/X8-ForagaxSquareWaveTwoBiome-v1/foragax/ForagaxSquareWaveTwoBiome-v1 --plot-name rtu-frozen --filter-alg-apertures Search-Oracle RealTimeActorCriticMLP_frozen_5M:5 RealTimeActorCriticMLP_frozen_5M:9 RealTimeActorCriticMLP_frozen_5M:15 --legend
python src/learning_curve.py experiments/X8-ForagaxSquareWaveTwoBiome-v1/foragax/ForagaxSquareWaveTwoBiome-v1 --plot-name rtu-frozen-normalized --filter-alg-apertures Search-Oracle RealTimeActorCriticMLP_frozen_5M:5 RealTimeActorCriticMLP_frozen_5M:9 RealTimeActorCriticMLP_frozen_5M:15 --normalize Search-Oracle --legend

python src/learning_curve.py experiments/X8-ForagaxSquareWaveTwoBiome-v1/foragax/ForagaxSquareWaveTwoBiome-v1 --plot-name ppo --filter-alg-apertures Search-Oracle ActorCriticMLP:5 ActorCriticMLP:9 ActorCriticMLP:15 --legend
python src/learning_curve.py experiments/X8-ForagaxSquareWaveTwoBiome-v1/foragax/ForagaxSquareWaveTwoBiome-v1 --plot-name ppo-normalized --filter-alg-apertures Search-Oracle ActorCriticMLP:5 ActorCriticMLP:9 ActorCriticMLP:15 --normalize Search-Oracle --legend

python src/learning_curve.py experiments/X8-ForagaxSquareWaveTwoBiome-v1/foragax/ForagaxSquareWaveTwoBiome-v1 --plot-name ppo-l2 --filter-alg-apertures Search-Oracle ActorCriticMLP-l2:5 ActorCriticMLP-l2:9 ActorCriticMLP-l2:15 --legend
python src/learning_curve.py experiments/X8-ForagaxSquareWaveTwoBiome-v1/foragax/ForagaxSquareWaveTwoBiome-v1 --plot-name ppo-l2-normalized --filter-alg-apertures Search-Oracle ActorCriticMLP-l2:5 ActorCriticMLP-l2:9 ActorCriticMLP-l2:15 --normalize Search-Oracle --legend

python src/learning_curve.py experiments/X8-ForagaxSquareWaveTwoBiome-v1/foragax/ForagaxSquareWaveTwoBiome-v1 --plot-name ppo-reward-trace --filter-alg-apertures Search-Oracle ActorCriticMLP-reward-trace:5 ActorCriticMLP-reward-trace:9 ActorCriticMLP-reward-trace:15 --legend
python src/learning_curve.py experiments/X8-ForagaxSquareWaveTwoBiome-v1/foragax/ForagaxSquareWaveTwoBiome-v1 --plot-name ppo-reward-trace-normalized --filter-alg-apertures Search-Oracle ActorCriticMLP-reward-trace:5 ActorCriticMLP-reward-trace:9 ActorCriticMLP-reward-trace:15 --normalize Search-Oracle --legend

python src/learning_curve.py experiments/X8-ForagaxSquareWaveTwoBiome-v1/foragax/ForagaxSquareWaveTwoBiome-v1 --plot-name ppo-reward-trace-frozen --filter-alg-apertures Search-Oracle ActorCriticMLP-reward-trace_frozen_5M:5 ActorCriticMLP-reward-trace_frozen_5M:9 ActorCriticMLP-reward-trace_frozen_5M:15 --legend
python src/learning_curve.py experiments/X8-ForagaxSquareWaveTwoBiome-v1/foragax/ForagaxSquareWaveTwoBiome-v1 --plot-name ppo-reward-trace-frozen-normalized --filter-alg-apertures Search-Oracle ActorCriticMLP-reward-trace_frozen_5M:5 ActorCriticMLP-reward-trace_frozen_5M:9 ActorCriticMLP-reward-trace_frozen_5M:15 --normalize Search-Oracle --legend
