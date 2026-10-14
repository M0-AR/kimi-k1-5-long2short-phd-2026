.PHONY: run docker-repro verify clean

run:
	python3 experiments/run_all.py

docker-repro:
	docker compose up --build

verify:
	python3 -c "import json,glob;fs=sorted(glob.glob('benchmarks/*.json'));[print(f+': '+str(json.load(open(f)).get('claim_verified',json.load(open(f)).get('all_verified')))) for f in fs]"

clean:
	rm -f benchmarks/*.png
