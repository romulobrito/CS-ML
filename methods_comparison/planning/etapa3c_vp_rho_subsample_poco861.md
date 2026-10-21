# Etapa 3c -- Rho subsample: CLP sparse b vs RF sparse (Poco 861)

## Objetivo

Quantificar quanta calibracao esparsa (fracao rho) cada metodo precisa para
se aproximar do RF oracle (treino denso 87/87).

## Protocolo

Por fold depth-block CV:
- `cal_train`, `cal_val`, `cal_test` = subamostra rho das linhas do fold
- CLP: b em cal_*; z0 = encode(0); lambda em val
- RF sparse: treina so em cal_train; prediz test_idx
- Referencias: RF oracle, Gassmann

rho = 10/87: indices fixos dos 10 plugs (nao aleatorio).

## Resultados (2026-06-24)

| rho | n_cal test | CLP MAPE | RF sparse MAPE |
|-----|------------|----------|----------------|
| 0 | 0 | 12,2% | 15,4% |
| 0,10 | 3 | 11,4% | 10,1% |
| 0,115 plugs | 3 | 11,0% | 11,0% |
| 0,30 | 9 | 7,9% | 10,9% |
| 0,50 | 14 | 5,7% | 8,2% |
| 1,00 | 29 | 1,6% | 9,0% |

Referencias: RF oracle 8,2%; Gassmann 15,4%.

**Crossover CLP vs oracle:** rho ~ 0,3.
**RF sparse:** nunca alcanca oracle (min 8,2% em rho=0,5).

## Leitura

- Com sônico denso no treino: RF oracle (Etapa 3).
- Com ~30% calibracao esparsa em teste: CLP supera oracle.
- Com 10 plugs apenas: CLP = RF sparse (~11%), ambos abaixo do oracle.
- rho=1 CLP usa resíduo sônico em todo bloco de teste em b (cenario ancora completa em teste).

Script: `run_861_clp_vp_rho_subsample.py`
