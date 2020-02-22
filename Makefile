.PHONY: all

compile-reqs:
	pip-compile requirements.in > requirements.txt
	pip-compile requirements-dev.in > requirements-dev.txt

test:
	pytest .

lint:
	tox -e lint

release: compile-reqs
	tox -r
