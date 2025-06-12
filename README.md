# urban_trees
Beispielcode und Datensatz für die Berechnung des Wasserhaushalts von Stadtbäumen,
sowie für die Optimierung von Zisterngröße und -auffangfläche um Wasserstress zu vermeiden.
Dieses Repository ist angelehnt an den Report:

"Bewässerung von Stadtbäumen", Paul Voit, Universität Potsdam, 2025


doi of this repo:
XXXXX

Kontact: [voit@uni-potsdam.de](voit@uni-potsdam.de)

ORCIDs des Autors:
P. Voit:  [0000-0003-1005-0979](https://orcid.org/0000-0003-1005-0979)

Installation:
Der Code wurde in Python 3.10 implementiert. Verwendete Pakete sind Numpy (Harris et al., 2020), Pandas (McKinney, 2010) und scipy (Virtanen et al., 2020).

Um die Notebooks anzusehen, muss jupyter installiert sein.
Der Benutzer kann die beigefügte Datei urban_tree.yaml verwenden, um eine Conda-Umgebung zu erstellen, die alle notwendigen Pakete enthält. Dies kann wie folgt erfolgen:
`conda config --add channels conda-forge`
`conda config --set channel_priority strict`
`conda env create -f urban_tree.yml`

Sobald die Umgebung installiert ist, muss sie aktiviert werden. Öffnen Sie ein
 Terminal Ihrer Wahl und geben Sie folgenden Befehl ein:
conda activate urban_tree
Anschließend starten Sie Jupyter Notebook mit:
jupyter notebook
Nun können die bereitgestellten Notebooks im Browser ausgeführt werden.

Alternativ können Sie alle notwendigen Pakete auch manuell installieren, ohne eine
Conda-Umgebung zu verwenden – in diesem Fall kann jedoch nicht garantiert werden, dass alle Pakete korrekt funktionieren.

# Beinhaltete Dateien
## urban_trees.ipynb
Beispielcode und Datensatz für die Berechnung des Wasserhaushalts von Stadtbäumen,
sowie für die Optimierung von Zisterngröße und -auffangfläche um Wasserstress zu vermeiden.

## urban_tree_model.py
Der Programmcode des Wasserhaushaltsmodells für Stadtbäume.

## example_data_potsdam.csv
Observationsdaten des Niederschlags, sowie berechnete potentielle Evapotranspiration.
Station Potsdam, ID: 3987
Die Niederschlagsdaten stammen vom deutschen Wetterdienst (DWD, 2024)

## urban_tree.yml
Diese Datei kann benutzt werden um alle erforderlichen Programmbibliotheken in einer
conda-Umgebung zu installieren. Siehe "Installation".


# Referenzen
Burn, D. H.: Evaluation of regional flood frequency analysis with a region of influence approach, Water Resources Research, 26, 2257–2265,
publisher: Wiley Online Library, 1990.

DWD. (2024). Klimadaten: gemessene Parame-
ter an DWD-Stationen und gleichgestellten Partnernetzsta-
tionen [data set] [zuletzt aufgerufen: 19.05.2025]. https :
/ / opendata . dwd . de / climate _ environment / CDC /
observations_germany/climate/daily/kl/

Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). DOI: 10.1038/s41586-020-2649-2.

Hoyer, S. & Hamman, J., (2017). xarray: N-D labeled Arrays and Datasets in Python. Journal of Open Research Software. 5(1), p.10. DOI: https://doi.org/10.5334/jors.148

Data structures for statistical computing in python, McKinney, Proceedings of the 9th Python in Science Conference, Volume 445, 2010

Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., Burovski, E., Peterson, P., Weckesser, W.,
Bright, J., van der Walt, S. J., Brett, M., Wilson, J., Millman, K. J., Mayorov, N., Nelson, A. R. J., Jones, E., Kern, R., Lar-
son, E., . . . SciPy 1.0 Contributors. (2020). SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods, 17,
261–272. https://doi.org/10.1038/s41592-
019-0686-2