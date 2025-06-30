# AGENTS.md

test: pytest --maxfail=1 --disable-warnings -q
lint: flake8 src tests
typecheck: mypy src
