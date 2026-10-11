# XeLaTeX + Biber pipeline (matches the global VS Code recipe)
$pdf_mode = 5;            # 5 = XeLaTeX
$xelatex = 'xelatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';
$bibtex_use = 2;         # run Biber automatically as needed
$pdflatex = 'xelatex %O %S';
$biber = 'biber %O %S';
$clean_ext = 'synctex.gz synctex.gz(busy) run.xml bbl bcf blg fls fdb_latexmk gls glo ist acn glg alg acr';

# Run makeglossaries automatically so the List of Abbreviations populates
add_cus_dep('glo', 'gls', 0, 'run_makeglossaries');
add_cus_dep('acn', 'acr', 0, 'run_makeglossaries');
sub run_makeglossaries { system('makeglossaries', $_[0]); }
