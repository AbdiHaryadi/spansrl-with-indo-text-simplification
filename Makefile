config := default

test: src/test.py
	python3 src/test.py $(config)

predict: src/predict.py
	python3 -m src.predict $(config)

extract_features: src/features/extract.py
	python3 src/features/extract.py $(config)

fetch_data: src/data/fetch_all_datas.sh
	chmod +x ./src/data/fetch_all_datas.sh
	./src/data/fetch_all_datas.sh
