#!/bin/sh
# Build main.pdf.  EXTRA_TEXINPUTS can point to local shims for minimal TeX installs
# (e.g. a grfext.sty stub on TinyTeX); a full TeX Live needs nothing extra.
set -e
cd "$(dirname "$0")"
export TEXINPUTS="${EXTRA_TEXINPUTS:+$EXTRA_TEXINPUTS//:}"
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null
bibtex main > /dev/null || true
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null
grep -E "^!|Warning: (Citation|Reference)" main.log | sort | uniq || true
echo "built main.pdf"
