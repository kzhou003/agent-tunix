# Makefile for Sphinx documentation

# You can set these variables from the command line.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
PAPER         ?=
BUILDDIR      ?= _build

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) .

.PHONY: help Makefile

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  clean       remove build directory"
	@echo "  html        make standalone HTML files"
	@echo "  dirhtml     make HTML files named index.html in directories"
	@echo "  singlehtml  make single-file HTML web pages"
	@echo "  pickle      make pickle files"
	@echo "  json        make JSON files"
	@echo "  htmlhelp    make HTML files and a HTML help project"
	@echo "  qthelp      make HTML files and a Qt help project"
	@echo "  devhelp     make HTML files as a devhelp project"
	@echo "  epub        make an epub file"
	@echo "  latex       make LaTeX files, you can set PAPER=a4 or PAPER=letter"
	@echo "  latexpdf    make LaTeX files and run pdflatex on them"
	@echo "  text        make text files"
	@echo "  man         make manual pages"
	@echo "  texinfo     make Texinfo files"
	@echo "  gettext     make PO message catalogs"
	@echo "  changes     make an overview of all changed/added/deprecated items"
	@echo "  xml         make Docutils-compatible XML files"
	@echo "  pseudoxml   make pseudoxml-2.1 files"
	@echo "  linkcheck   check all external links for integrity"
	@echo "  doctest     run all doctests embedded in the documentation (if enabled)"
	@echo "  coverage    run coverage check of the documentation (if enabled)"

.PHONY: clean
clean:
	rm -rf $(BUILDDIR)

.PHONY: html
html:
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML files are in $(BUILDDIR)/html."

.PHONY: dirhtml
dirhtml:
	$(SPHINXBUILD) -b dirhtml $(ALLSPHINXOPTS) $(BUILDDIR)/dirhtml
	@echo
	@echo "Build finished. The HTML files are in $(BUILDDIR)/dirhtml."

.PHONY: singlehtml
singlehtml:
	$(SPHINXBUILD) -b singlehtml $(ALLSPHINXOPTS) $(BUILDDIR)/singlehtml
	@echo
	@echo "Build finished. The HTML files are in $(BUILDDIR)/singlehtml."

.PHONY: pickle
pickle:
	$(SPHINXBUILD) -b pickle $(ALLSPHINXOPTS) $(BUILDDIR)/pickle
	@echo
	@echo "Build finished. The pickle files are in $(BUILDDIR)/pickle."

.PHONY: json
json:
	$(SPHINXBUILD) -b json $(ALLSPHINXOPTS) $(BUILDDIR)/json
	@echo
	@echo "Build finished. The JSON files are in $(BUILDDIR)/json."

.PHONY: htmlhelp
htmlhelp:
	$(SPHINXBUILD) -b htmlhelp $(ALLSPHINXOPTS) $(BUILDDIR)/htmlhelp
	@echo
	@echo "Build finished. The HTML help files are in $(BUILDDIR)/htmlhelp."

.PHONY: qthelp
qthelp:
	$(SPHINXBUILD) -b qthelp $(ALLSPHINXOPTS) $(BUILDDIR)/qthelp
	@echo
	@echo "Build finished. The HTML help files are in $(BUILDDIR)/qthelp."

.PHONY: devhelp
devhelp:
	$(SPHINXBUILD) -b devhelp $(ALLSPHINXOPTS) $(BUILDDIR)/devhelp
	@echo
	@echo "Build finished. The HTML files are in $(BUILDDIR)/devhelp."

.PHONY: epub
epub:
	$(SPHINXBUILD) -b epub $(ALLSPHINXOPTS) $(BUILDDIR)/epub
	@echo
	@echo "Build finished. The epub file is in $(BUILDDIR)/epub."

.PHONY: latex
latex:
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(BUILDDIR)/latex
	@echo
	@echo "Build finished. The LaTeX files are in $(BUILDDIR)/latex."

.PHONY: latexpdf
latexpdf:
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(BUILDDIR)/latex
	cd $(BUILDDIR)/latex && make all-pdf
	@echo
	@echo "Build finished. The PDF files are in $(BUILDDIR)/latex."

.PHONY: latexpdf-all
latexpdf-all:
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(BUILDDIR)/latex
	cd $(BUILDDIR)/latex && make all-pdf-explicit
	@echo
	@echo "Build finished. The PDF files are in $(BUILDDIR)/latex."

.PHONY: text
text:
	$(SPHINXBUILD) -b text $(ALLSPHINXOPTS) $(BUILDDIR)/text
	@echo
	@echo "Build finished. The text files are in $(BUILDDIR)/text."

.PHONY: man
man:
	$(SPHINXBUILD) -b man $(ALLSPHINXOPTS) $(BUILDDIR)/man
	@echo
	@echo "Build finished. The man files are in $(BUILDDIR)/man."

.PHONY: texinfo
texinfo:
	$(SPHINXBUILD) -b texinfo $(ALLSPHINXOPTS) $(BUILDDIR)/texinfo
	@echo
	@echo "Build finished. The Sphinx texinfo files are in $(BUILDDIR)/texinfo."

.PHONY: gettext
gettext:
	$(SPHINXBUILD) -b gettext $(BUILDDIR)/gettext
	@echo
	@echo "Build finished. The gettext files are in $(BUILDDIR)/gettext."

.PHONY: changes
changes:
	$(SPHINXBUILD) -b changes $(ALLSPHINXOPTS) $(BUILDDIR)/changes
	@echo
	@echo "Build finished. The changes overview file is in $(BUILDDIR)/changes."

.PHONY: linkcheck
linkcheck:
	$(SPHINXBUILD) -b linkcheck $(ALLSPHINXOPTS) $(BUILDDIR)/linkcheck
	@echo
	@echo "Build finished. The link check complete; look for any errors in the $(BUILDDIR)/linkcheck/output.txt file."

.PHONY: doctest
doctest:
	$(SPHINXBUILD) -b doctest $(ALLSPHINXOPTS) $(BUILDDIR)/doctest
	@echo
	@echo "Test passed. The doctests are in $(BUILDDIR)/doctest/doctest.txt file."

.PHONY: coverage
coverage:
	$(SPHINXBUILD) -b coverage $(ALLSPHINXOPTS) $(BUILDDIR)/coverage
	@echo
	@echo "Test passed. The coverage report is in $(BUILDDIR)/coverage/python.txt."

.PHONY: xml
xml:
	$(SPHINXBUILD) -b xml $(ALLSPHINXOPTS) $(BUILDDIR)/xml
	@echo
	@echo "Build finished. The XML files are in $(BUILDDIR)/xml."

.PHONY: pseudoxml
pseudoxml:
	$(SPHINXBUILD) -b pseudoxml $(ALLSPHINXOPTS) $(BUILDDIR)/pseudoxml
	@echo
	@echo "Build finished. The pseudoxml-2.1 files are in $(BUILDDIR)/pseudoxml."

.PHONY: serve
serve: html
	@echo "Serving documentation at http://localhost:8000"
	cd $(BUILDDIR)/html && python -m http.server 8000
