# make test        unit tests on synthetic books (seconds, no data needed)
# make check       the published simple-bot and guard-policy numbers on recording 2 (needs data/)
# make results     every table in the README and WRITEUP from the recordings and the saved policies (needs data/, ~hours)
# make fx          the rough forex test on 1-minute bars (needs data/fx/<PAIR>_M1.json, minutes)
# make retrain     retrain the published policies and the seeds (slow; results then differ slightly from the saved ones)
PY := python3
D4 := data/btcusd_l3_part4.csv

.PHONY: test check results retrain fx
test:
	$(PY) -m pytest -q tests

check:
	$(PY) regress_check.py

results: test
	$(PY) compare.py data/btcusd_l3_part3.csv && mv comparison.csv comparison_part3.csv
	$(PY) compare.py data/btcusd_l3_part2.csv
	for s in base lat250 lat1000 through lat1000i; do $(PY) stress.py $$s || exit 1; done
	for s in base lat1000 through lat1000i; do $(PY) stress.py $$s $(D4) || exit 1; done
	$(PY) volguard.py && $(PY) volguard.py 1000 && $(PY) volguard.py 0 $(D4)
	$(PY) select_gen.py && $(PY) select_gen.py $(D4)

retrain:
	$(PY) baselines.py
	$(PY) learn.py unguarded none && $(PY) learn.py capped cap && $(PY) learn.py cbf cbf
	cd seeds && for s in 1 2 3 4 5; do $(PY) ../learn.py cbf_s$$s cbf $$s && $(PY) ../learn.py none_s$$s none $$s || exit 1; done

fx:
	for p in EURUSD GBPUSD USDJPY; do for g in cbf none; do for s in 1 2 3 4 5; do $(PY) fx_bars.py train $$p $$g $$s > /dev/null || exit 1; done; done; done
	$(PY) fx_bars.py test
