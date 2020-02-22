.PHONY: all

compile-reqs:
	cat requirements.in | sort | uniq > requirements.in.tmp && mv requirements.in.tmp requirements.in
	pip-compile requirements.in > requirements.txt

	cat requirements-dev.in | sort | uniq > requirements-dev.in.tmp && mv requirements-dev.in.tmp requirements-dev.in
	pip-compile requirements-dev.in > requirements-dev.txt

test:
	pytest .

lint:
	tox -e lint

release: compile-reqs
	tox -r
