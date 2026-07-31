SHELL := /bin/bash

.PHONY: help validate materialize train-smoke test

help:
	@echo "Available targets:"
	@echo "  make validate SLUG=<experiment_slug> PROFILE=devsample"
	@echo "  make materialize SLUG=<experiment_slug> PROFILE=devsample"
	@echo "  make train-smoke SLUG=<experiment_slug> PROFILE=devsample STEPS=2"
	@echo "  make test"

validate:
	python3 scripts/validate_experiment.py --slug $(SLUG) --dataset-profile $(PROFILE)

materialize:
	python3 scripts/materialize_runs.py --slug $(SLUG) --dataset-profile $(PROFILE)

train-smoke:
	python3 scripts/train_experiment.py --slug $(SLUG) --dataset-profile $(PROFILE) --trainer smoke --smoke-steps $(STEPS)

test:
	python3 -m unittest discover -s tests
