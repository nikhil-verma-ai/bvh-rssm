.PHONY: test test-unit test-env smoke-train smoke-serve full-run lint

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

test-env:
	pytest tests/integration/test_environments.py -v

smoke-train:
	python scripts/train.py +experiment=smoke_train

smoke-serve:
	python scripts/serve.py &
	sleep 2
	curl -s -X POST http://localhost:8000/predict \
		-H "Content-Type: application/json" \
		-d '{"obs": [0.0, 0.0, 0.0], "action": [0.0], "state": null}' | python -m json.tool
	pkill -f "scripts/serve.py" || true

full-run:
	python scripts/train.py +experiment=full_run

lint:
	python -m pyright bvh_rssm/
