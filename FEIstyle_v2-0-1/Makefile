FILE = thesis
LATEXMK_OPTIONS = -interaction=nonstopmode -halt-on-error -silent
LINE_WIDTH = error_line=240 half_error_line=160 max_print_line=240

all: pdf

pdf:
	$(LINE_WIDTH) latexmk -pdf -synctex=1 $(LATEXMK_OPTIONS) $(FILE).tex

clean:
	latexmk -c -silent $(FILE).tex

refresh: clean
	$(MAKE) pdf

.PHONY: all pdf clean refresh
