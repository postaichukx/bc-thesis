FILE = thesis
BUILD_DIR = .build

LATEXMK_OPTIONS = -pdf -synctex=1 -output-directory=$(BUILD_DIR) -interaction=nonstopmode -silent
LINE_WIDTH = error_line=240 half_error_line=160 max_print_line=240

all: pdf

pdf:
	$(LINE_WIDTH) latexmk $(LATEXMK_OPTIONS) $(FILE).tex

clean:
	rm -rf $(BUILD_DIR)

refresh: clean pdf

.PHONY: all pdf clean refresh
