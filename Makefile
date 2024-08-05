.PHONY: tests coverage

setup:
	poetry install -v --all-extras

check:
	poetry run pylint source_code
	poetry run black . --check
	poetry run isort . --check --gitignore
	poetry run flake8 source_code
	poetry run flake8 tests

lint:
	echo running black...
	black .
	echo running isort...
	isort . --gitignore
	echo running flake8...
	flake8 .
	echo running autoflake...
	autoflake -i --remove-all-unused-imports -r --ignore-init-module-imports . --exclude venv
