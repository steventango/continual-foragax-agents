python src/learning_curve.py experiments/E92-search-limited-fov/foragax-sweep/ForagaxTwoBiome-v10 &
python src/learning_curve.py experiments/E93-mitigations/foragax-sweep/ForagaxTwoBiome-v10 &
python src/learning_curve.py experiments/E94-baselines-vs-recurrent/foragax-sweep/ForagaxTwoBiome-v10 &
python src/learning_curve.py experiments/E96-search-limited-fov/foragax-sweep/ForagaxTwoBiome-v13 &
python src/learning_curve.py experiments/E97-mitigations/foragax-sweep/ForagaxTwoBiome-v13 &
python src/learning_curve.py experiments/E98-baselines-vs-recurrent/foragax-sweep/ForagaxTwoBiome-v13 &
wait
python src/mushroom_curve.py experiments/E92-search-limited-fov/foragax-sweep/ForagaxTwoBiome-v10 &
python src/mushroom_curve.py experiments/E93-mitigations/foragax-sweep/ForagaxTwoBiome-v10 &
python src/mushroom_curve.py experiments/E94-baselines-vs-recurrent/foragax-sweep/ForagaxTwoBiome-v10 &
python src/mushroom_curve.py experiments/E96-search-limited-fov/foragax-sweep/ForagaxTwoBiome-v13 &
python src/mushroom_curve.py experiments/E97-mitigations/foragax-sweep/ForagaxTwoBiome-v13 &
python src/mushroom_curve.py experiments/E98-baselines-vs-recurrent/foragax-sweep/ForagaxTwoBiome-v13 &
wait
python src/biome_curve.py experiments/E92-search-limited-fov/foragax-sweep/ForagaxTwoBiome-v10 &
python src/biome_curve.py experiments/E93-mitigations/foragax-sweep/ForagaxTwoBiome-v10 &
python src/biome_curve.py experiments/E94-baselines-vs-recurrent/foragax-sweep/ForagaxTwoBiome-v10 &
python src/biome_curve.py experiments/E96-search-limited-fov/foragax-sweep/ForagaxTwoBiome-v13 &
python src/biome_curve.py experiments/E97-mitigations/foragax-sweep/ForagaxTwoBiome-v13 &
python src/biome_curve.py experiments/E98-baselines-vs-recurrent/foragax-sweep/ForagaxTwoBiome-v13 &
wait
