config := default
valconfig := validation
file := result.txt

train: src/train.py
	python3 src/train.py $(config) > $(config)2.txt

validate: src/validate.py data/scores/
	python3 src/validate.py $(valconfig) val > data/scores/current_val.txt

test: src/validate.py data/scores/
	python3 src/validate.py $(valconfig) test > data/scores/current_test.txt

predict: src/predict.py
	python3 -m src.predict $(config)

extract_features: src/features/extract.py
	python3 src/features/extract.py $(config)

fetch_data: src/data/fetch_all_datas.sh
	chmod +x ./src/data/fetch_all_datas.sh
	./src/data/fetch_all_datas.sh
