.PHONY: all

compile-reqs:
	cat requirements.in | sort | uniq > requirements.in.tmp && mv requirements.in.tmp requirements.in
	pip-compile --output-file=requirements.txt requirements.in

	cat requirements.in requirements-dev.in | sort | uniq > requirements-dev.in.tmp && mv requirements-dev.in.tmp requirements-dev.in
	pip-compile --output-file=requirements-dev.txt requirements-dev.in

test:
	pytest .

lint:
	tox -e lint

release: compile-reqs
	tox -r
