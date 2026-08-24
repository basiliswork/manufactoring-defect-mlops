install:
	pip install --upgrade pip && pip install -r requirements.txt

lint:
	pylint --disable=R,C src/*.py

test:
	PYTHONPATH=. pytest tests/

run:
	python src/app.py