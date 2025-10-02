ENV?=configs/default.yaml

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

download:
	python scripts/download_power.py --config $(ENV)

preprocess:
	PYTHONPATH=. python scripts/preprocess.py --config $(ENV)

train:
	PYTHONPATH=. python scripts/train.py --config $(ENV)

evaluate:
	PYTHONPATH=. python scripts/evaluate.py --config $(ENV)

predict:
	PYTHONPATH=. python scripts/predict.py --config $(ENV)

tune_thresholds:
	PYTHONPATH=. python scripts/tune_thresholds.py --config $(ENV)

plots-train:
	python scripts/plot_training_curves.py --history models/train_history.json --outdir reports/figures

plots-eval:
	PYTHONPATH=. python scripts/plot_eval_curves.py --config $(ENV) --figdir reports/figures --metdir reports/metrics

metrics-md:
	python scripts/make_metrics_md.py --metrics_csv reports/metrics/test_metrics.csv --outmd reports/metrics/METRICS.md

app:
	streamlit run app/app.py -- --config $(ENV)
