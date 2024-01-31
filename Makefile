.PHONY: all

install:
	poetry install

test:
	poetry run pytest .

lint:
	poetry run ruff .

lock:
	poetry lock
