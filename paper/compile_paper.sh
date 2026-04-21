#!/usr/bin/env sh
# Clean auxiliary files then run pdflatex twice so \eqref/\ref resolve (no ??).
set -e
cd "$(dirname "$0")"
rm -f main.aux main.out main.toc main.lof main.lot main.log main.synctex.gz
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
echo "OK: see main.pdf and check main.log for warnings."
