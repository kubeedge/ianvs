# Makefile for Markdown compilation
# Usage: make html, make pdf, make all, etc.

PYTHON = python
MD_COMPILER = md_compiler.py
MD_FILES = README.md
OUTPUT_DIR = docs/compiled

# Default target
.PHONY: help
help:
	@echo "Markdown Compilation Makefile"
	@echo "Available targets:"
	@echo "  html       - Convert MD files to HTML"
	@echo "  pdf        - Convert MD files to PDF (requires weasyprint)"
	@echo "  text       - Preview MD files as text"
	@echo "  all        - Convert to all formats"
	@echo "  clean      - Remove generated files"
	@echo "  install    - Install required dependencies"
	@echo ""
	@echo "Examples:"
	@echo "  make html"
	@echo "  make pdf"
	@echo "  make all"

# Create output directory
$(OUTPUT_DIR):
	mkdir -p $(OUTPUT_DIR)

# Convert to HTML
.PHONY: html
html: $(OUTPUT_DIR)
	@echo "📄 Converting Markdown files to HTML..."
	@for file in $(MD_FILES); do \
		$(PYTHON) $(MD_COMPILER) $$file -f html -o $(OUTPUT_DIR)/$$(basename $$file .md).html; \
	done
	@echo "✅ HTML conversion completed!"

# Convert to PDF
.PHONY: pdf
pdf: $(OUTPUT_DIR)
	@echo "📄 Converting Markdown files to PDF..."
	@for file in $(MD_FILES); do \
		$(PYTHON) $(MD_COMPILER) $$file -f pdf -o $(OUTPUT_DIR)/$$(basename $$file .md).pdf; \
	done
	@echo "✅ PDF conversion completed!"

# Preview as text
.PHONY: text
text:
	@echo "📄 Previewing Markdown files..."
	@for file in $(MD_FILES); do \
		$(PYTHON) $(MD_COMPILER) $$file -f text; \
	done

# Convert to all formats
.PHONY: all
all: $(OUTPUT_DIR)
	@echo "📄 Converting Markdown files to all formats..."
	@for file in $(MD_FILES); do \
		$(PYTHON) $(MD_COMPILER) $$file -f html -o $(OUTPUT_DIR)/$$(basename $$file .md).html; \
		$(PYTHON) $(MD_COMPILER) $$file -f pdf -o $(OUTPUT_DIR)/$$(basename $$file .md).pdf; \
	done
	@echo "🎉 All conversions completed!"

# Install dependencies
.PHONY: install
install:
	@echo "📦 Installing Markdown compilation dependencies..."
	$(PYTHON) -m pip install markdown
	@echo "Optional: Install weasyprint for PDF support:"
	@echo "  $(PYTHON) -m pip install weasyprint"

# Clean generated files
.PHONY: clean
clean:
	@echo "🧹 Cleaning generated files..."
	rm -rf $(OUTPUT_DIR)
	rm -f *.html *.pdf
	@echo "✅ Cleanup completed!"

# Quick HTML conversion for README
.PHONY: readme
readme:
	$(PYTHON) $(MD_COMPILER) README.md -f html

# Watch for changes (requires inotify-tools on Linux)
.PHONY: watch
watch:
	@echo "👀 Watching for changes to Markdown files..."
	@echo "Press Ctrl+C to stop"
	@while true; do \
		$(PYTHON) $(MD_COMPILER) README.md -f html; \
		sleep 2; \
	done 