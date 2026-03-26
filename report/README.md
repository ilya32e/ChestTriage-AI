# Rapport LaTeX

Ce dossier contient une version academique du rapport en LaTeX:

- `main.tex` : document principal
- `generated_results.tex` : tableaux de resultats generes depuis MLflow

## Mettre a jour les tableaux de resultats

Depuis la racine du projet:

```powershell
python .\scripts\export_report_tables.py
```

Le script lit `mlruns/` et remplit automatiquement `report/generated_results.tex`.

## Compiler le rapport

Une distribution LaTeX n'est pas installee sur cette machine au moment de la generation du rapport. Il faut donc installer `MiKTeX`, `TeX Live` ou `TinyTeX`, puis compiler depuis le dossier `report`:

```powershell
cd .\report
pdflatex .\main.tex
pdflatex .\main.tex
```

Deux passes sont recommandees pour la table des matieres et les references internes.

## Notes

- Le rapport n'inclut pas les quickstarts dans les tableaux finaux.
- Les cellules `n.d.` signifient qu'un run final n'est pas encore termine ou non exporte dans MLflow.
- Les figures d'EDA et de classification supervisee sont chargees directement depuis `../artifacts/`.
- La version actuelle de `main.tex` depasse `8 500` mots environ, ce qui vise volontairement un format academique d'environ `20 a 30 pages` de contenu hors page de garde et sommaire. La verification exacte en nombre de pages doit etre faite apres compilation.
