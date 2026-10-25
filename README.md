# IPM manuscript

Final source for the Information Processing and Management submission.

## Compile

```bash
cd paper/ipm
tectonic --keep-logs main.tex
tectonic --keep-logs title_page.tex
```

or use a standard TeX Live installation:

```bash
latexmk -pdf main.tex
```

The CAS class files, bibliography style, thumbnail assets, and all referenced
figures are included in this directory.

## Files

- `main.tex`: manuscript source
- `main.pdf`: anonymized manuscript for review
- `title_page.tex`: author-identifying title page source
- `title_page.pdf`: title page for editorial use
- `cover_letter.tex`: cover-letter source
- `cover_letter.pdf`: two-page cover letter with the editor response
- `refs.bib`: bibliography
- `figs/`: figures used by the manuscript
- `cas-sc.cls`, `cas-common.sty`, `cas-model2-names.bst`: CAS template files
- `thumbnails/`: template assets used by the author block

## Build notes

- Footnote hyperlinks are disabled with `hyperfootnotes=false` because the CAS
  class creates empty footnote anchors. Footnote text remains visible; normal
  citation, URL, and cross-reference links remain enabled.
- `main.pdf` is the anonymized reviewer manuscript. `title_page.pdf` is the
  separate author-identifying title page for editorial use.
- The title uses a plain `\title[mode=title]{...}` command and automatic line
  wrapping.
- References use the CAS template's author-year style.
- Byte quantities use KiB and MiB.
