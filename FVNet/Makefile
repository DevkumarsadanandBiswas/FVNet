.PHONY: install test smoke
install:
	pip install -e ".[dev]"
test:
	pytest -q
smoke:  # tiny CPU end-to-end run on synthetic data (2D and 3D)
	python scripts/make_synthetic_data.py --dim 2 --out data/synth2d --n 8
	python scripts/run_all.py -c configs/smoke_2d.yaml
	python scripts/make_synthetic_data.py --dim 3 --out data/synth3d --n 8
	python scripts/run_all.py -c configs/smoke_3d.yaml
