# Shortcuts. Each target is just the command shown; run `make <target>` from the repo root
# with the virtual environment activated.
.PHONY: test lint format cov synthetic yahoo network dashboard

test:            ## run offline unit tests
	python -m pytest -q

lint:            ## static checks (style, unused imports, likely bugs)
	ruff check . && ruff format --check .

format:          ## auto-format code in place
	ruff format . && ruff check . --fix

cov:             ## tests + line coverage report
	python -m pytest -q --cov=alphalab --cov-report=term-missing

synthetic:       ## full pipeline on offline synthetic data
	alphalab ingest --config configs/synthetic.yaml && alphalab build --config configs/synthetic.yaml

yahoo:           ## full pipeline on live Yahoo data (needs internet)
	alphalab ingest --config configs/yahoo_us_core.yaml && alphalab build --config configs/yahoo_us_core.yaml

network:         ## live-data tests only
	python -m pytest -m network -q

dashboard:       ## open the results dashboard (needs saved results; read-only)
	streamlit run dashboard/app.py
