# One-command reproduction of the spherical-harmonic truncation study.
#
#   make            -> create venv, install pinned deps, run the full pipeline
#   make figures    -> run the pipeline (assumes deps already installed)
#   make clean      -> remove generated figures and results (keeps cached data)
#   make distclean  -> also remove the virtual environment

PYTHON ?= python
VENV    = venv
BIN     = $(VENV)/bin

.PHONY: all figures figures-cache aux clean distclean test audit pdf

all: $(VENV)/.installed figures

$(VENV)/.installed: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt
	touch $@

# Core pipeline + auxiliary figures A/B/D (rendered from results/*.json|csv;
# skipped gracefully if a producing script has not been run -- see `make aux`).
figures:
	$(BIN)/python make_figures.py

# Regenerate the auxiliary result tables that feed figures A/B/D
# (planetary_curves.csv, water_fill.json, codec_gap.json,
# radius_compressibility.csv). Uses the cached spherical-harmonic models; no
# new download once the first run has populated the pyshtools cache.
aux:
	$(BIN)/python planetary.py
	$(BIN)/python ratedist.py
	$(BIN)/python codec.py
	$(BIN)/python radius.py

# Rebuild EVERY figure (1-5 + A/B/D) from committed artifacts only:
# figures 1-5 from results/cache.pkl, A/B/D from results/*.json|csv.
# No model download and no recomputation.
figures-cache:
	$(BIN)/python make_figures.py --from-cache results/cache.pkl

clean:
	rm -f figures/fig*.png figures/fig*.pdf figures/captions.md
	rm -f results/*.csv results/*.json results/*.txt

distclean: clean
	rm -rf $(VENV)

test:
	$(PYTHON) -m pytest tests/ -v

audit:
	$(PYTHON) audit.py

pdf:
	(cd geoid/geoid_paper_arxiv_overleaf && \
	 PATH=/Library/TeX/texbin:$$PATH pdflatex main.tex && \
	 PATH=/Library/TeX/texbin:$$PATH bibtex main && \
	 PATH=/Library/TeX/texbin:$$PATH pdflatex main.tex && \
	 PATH=/Library/TeX/texbin:$$PATH pdflatex main.tex) && \
	cp geoid/geoid_paper_arxiv_overleaf/main.pdf geoid/paper_draft.pdf
