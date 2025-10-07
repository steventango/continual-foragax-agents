python src/learning_curve.py experiments/E99-weather/foragax/ForagaxWeather-v5 --plot-name fig4a --sample-type every --auto-label &
python src/learning_curve.py experiments/E99-weather/foragax/ForagaxWeather-v5 --plot-name fig4b --sample-type 9950000:11000000 --auto-label &
python src/learning_bar.py experiments/E99-weather/foragax/ForagaxWeather-v5 --plot-name fig4c --bars "DQN|9|every|" "DQN_L2|9|every|" "DQN_frozen_5M|9|every|" &
wait
