### main_bvs_v2 — по (task, agent)

| task | agent | n_ok/n | mean_final ± SE | mean_tokens | mean_coverage | mean_hints |
|---|---|---|---|---|---|---|
| adult_income | baseline | 5/5 | 0.9025 ± 0.0033 | 12,384 | 0.000 | 0.0 |
| adult_income | scaffold | 3/5 | 0.9037 ± 0.0031 | 15,397 | 0.667 | 6.2 |
| breast_cancer_roc_auc | baseline | 5/5 | 0.9982 ± 0.0002 | 5,032 | 0.000 | 0.0 |
| breast_cancer_roc_auc | scaffold | 5/5 | 0.9983 ± 0.0002 | 8,902 | 0.552 | 4.2 |
| california_housing | baseline | 4/5 | 0.5125 ± 0.0130 | 29,137 | 0.000 | 0.0 |
| california_housing | scaffold | 5/5 | 0.5119 ± 0.0004 | 26,457 | 0.705 | 12.6 |
| diabetes_rmse | baseline | 5/5 | 55.6722 ± 1.3623 | 24,212 | 0.000 | 0.0 |
| diabetes_rmse | scaffold | 4/5 | 55.0544 ± 0.6709 | 25,607 | 0.810 | 14.6 |
| titanic_survival | baseline | 5/5 | 0.8388 ± 0.0083 | 9,997 | 0.000 | 0.0 |
| titanic_survival | scaffold | 4/5 | 0.8333 ± 0.0048 | 12,354 | 0.610 | 4.0 |
| wine_quality | baseline | 4/5 | 0.6434 ± 0.0097 | 26,282 | 0.000 | 0.0 |
| wine_quality | scaffold | 5/5 | 0.6472 ± 0.0116 | 24,364 | 0.695 | 14.8 |

### ablation_hints — по (task, agent)

| task | agent | n_ok/n | mean_final ± SE | mean_tokens | mean_coverage | mean_hints |
|---|---|---|---|---|---|---|
| adult_income | baseline | 3/3 | 0.9035 ± 0.0044 | 6,952 | 0.000 | 0.0 |
| adult_income | scaffold_L1 | 3/3 | 0.9049 ± 0.0030 | 6,863 | 0.524 | 2.0 |
| adult_income | scaffold_L1L2 | 3/3 | 0.8996 ± 0.0002 | 7,739 | 0.492 | 2.0 |
| adult_income | scaffold_full | 3/3 | 0.9019 ± 0.0026 | 17,686 | 0.619 | 6.0 |
| california_housing | baseline | 3/3 | 0.5049 ± 0.0049 | 26,712 | 0.000 | 0.0 |
| california_housing | scaffold_L1 | 2/3 | 0.5166 ± 0.0030 | 29,785 | 0.714 | 5.0 |
| california_housing | scaffold_L1L2 | 2/3 | 0.5045 ± 0.0073 | 27,126 | 0.667 | 10.0 |
| california_housing | scaffold_full | 2/3 | 0.5071 ± 0.0027 | 27,790 | 0.730 | 13.0 |
| titanic_survival | baseline | 3/3 | 0.8429 ± 0.0022 | 23,571 | 0.000 | 0.0 |
| titanic_survival | scaffold_L1 | 3/3 | 0.8346 ± 0.0013 | 23,806 | 0.698 | 6.7 |
| titanic_survival | scaffold_L1L2 | 3/3 | 0.8379 ± 0.0049 | 15,915 | 0.698 | 5.0 |
| titanic_survival | scaffold_full | 3/3 | 0.8389 ± 0.0088 | 20,425 | 0.714 | 8.0 |