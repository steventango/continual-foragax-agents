# uv run python src/learning_curve.py experiments/E92-search-limited-fov/foragax/ForagaxTwoBiome-v10 --filter-algs DQN Search-Oracle Search-Brown-Avoid-Green --plot-name fig1a --ylim 0 0.8 --auto-label &
# uv run python src/learning_curve.py experiments/E92-search-limited-fov/foragax/ForagaxTwoBiome-v10 --filter-algs DQN_frozen_1M DQN_frozen_5M --plot-name fig1b --ylim 0 0.8 --auto-label &
# uv run python src/biome_bar.py experiments/E92-search-limited-fov/foragax/ForagaxTwoBiome-v10 --filter-algs DQN Search-Oracle Search-Brown-Avoid-Green --plot-name fig1c &
uv run python src/learning_curve.py experiments/E96-search-limited-fov/foragax/ForagaxTwoBiome-v13 --filter-algs DQN Search-Oracle Search-Brown-Avoid-Green --plot-name fig1a --ylim 0 1 --auto-label &
uv run python src/learning_curve.py experiments/E96-search-limited-fov/foragax/ForagaxTwoBiome-v13 --filter-algs DQN_frozen_1M DQN_frozen_5M --plot-name fig1b --ylim 0 1 --auto-label &
uv run python src/biome_bar.py experiments/E96-search-limited-fov/foragax/ForagaxTwoBiome-v13 --filter-algs DQN Search-Oracle Search-Brown-Avoid-Green --plot-name fig1c &
wait
