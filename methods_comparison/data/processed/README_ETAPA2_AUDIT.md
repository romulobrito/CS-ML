# Etapa 2 -- pacote de auditoria (poco 861)

Rodada que alimenta os slides/relatorio da Etapa 2: **2026-07-26** (UTC),
metricas alinhadas a `poco861_etapa2_dem_sc.tex` (junho/julho de 2026 no texto).
O README em `dlis_861_gassmann/README.md` tem numeros antigos (junho-15);
use os JSON/MANIFEST desta pasta.

## O que este commit versiona

| Item | Caminho |
|------|---------|
| Planilha ROCKPHYS (abas Velocity, Porosity, Mineralogy, Rock info) | `methods_comparison/data/ROCKPHYS_Database_04_12_2024 (7).xlsx` |
| Export GeoDict / PoreInfo (familias meso-macro e micro) | `methods_comparison/data/General_Output_Results_PoreInfo_samples_w861_MOGNOTomo.xlsx` |
| Tabela integrada dos 10 plugs CT | `methods_comparison/data/processed/861_integrated_ct_samples.xlsx` |
| Mesmo conteudo CT em CSV | `methods_comparison/data/processed/861_ct_samples.csv` |
| Ingest ROCKPHYS (Vp seco eixo Z, 22.1 MPa) | `methods_comparison/data/processed/rockphys_861/` |
| Parametros HFU (CT e calibrados; HFU4 = media 2+3) | `methods_comparison/data/processed/dem_sc_runs/hfu_calibration/` |
| Validacao lab, calibracao, LOO, A/B CT, NMR | `dem_sc_runs/lab_validation/`, `lab_calibration/`, `multiscale_ab/`, `multiscale_nmr/` |
| Sonico DSI processado + escala TDEP | `methods_comparison/data/processed/dlis_861/` |
| Gassmann vs DSI | `methods_comparison/data/processed/dlis_861_gassmann/` |
| Perfil enriquecido 87 linhas (ja estava no git) | `methods_comparison/data/processed/861_integrated_logs_enriched.xlsx` |

Nao entram: DLIS brutos, figuras PNG, volumes 3D, POCs de calibracao hierarquica.

## Condicao do Vp de laboratorio

A aba Velocity tem bloco seco (`vp_z_km_s`) e bloco saturado (`vp_z_km_s_sat`).
O ingest usa so o seco, transmissao, eixo Z, 22.1 MPa.
Nas 196 linhas do 861 as colunas saturadas estao vazias.
Nao ha neste repositorio um relatorio experimental de preparacao/saturacao dos plugs.

Familias de poro no PoreInfo: rotulos "Macro-mesoporosity" e "Microcroporosity"
(Anselmetti et al., 1998) nas fichas por amostra. Criterios de limiar estao
na planilha GeoDict; os volumes 3D nao foram versionados.

NMR: CMFF = fluido livre, BFV = fluido ligado, CMRP_3MS = porosidade NMR
(frame 75B do DLIS `3-brsa-861-sps_8_cmr_ecs.dlis`). Fracoes no DEM:
`f_meso = CMFF/(CMFF+BFV)`, `f_micro = BFV/(CMFF+BFV)`.

## Dependencias (Etapa 2)

Python 3.10+. Pacotes usados pelos scripts DEM/DLIS:

```
numpy pandas scipy matplotlib openpyxl dlisio
```

(`requirements.txt` na raiz lista o nucleo; `scipy` e `openpyxl`/`dlisio`
sao necessarios para calibracao e extracao.)

## Comandos da rodada de producao

A partir da raiz do repositorio, com o PYTHONPATH incluindo
`methods_comparison/scripts`:

```bash
export PYTHONPATH="methods_comparison/scripts:${PYTHONPATH}"

python methods_comparison/scripts/run_861_rockphys_ingest.py
python methods_comparison/scripts/run_861_dlis_sonic_extract.py
python methods_comparison/scripts/extract_861_cmr.py
python methods_comparison/scripts/run_861_dem_sc_lab_validation.py
python methods_comparison/scripts/run_861_dem_sc_lab_calibration.py
python methods_comparison/scripts/run_861_dem_sc_profile_87.py
python methods_comparison/scripts/run_861_dem_sc_gassmann.py
python methods_comparison/scripts/run_861_dlis_dem_validation.py
python methods_comparison/scripts/run_861_dem_sc_multiscale_ab.py
python methods_comparison/scripts/run_861_dem_sc_multiscale_nmr.py
```

PoreInfo -> tabela CT (ja gerado; exige a planilha GeoDict acima):

```bash
python methods_comparison/scripts/integrate_861_mogno_ct.py
```

Identificacao da execucao dos slides: MANIFEST `Generated: 2026-07-26T16:56:51Z`
em `dem_sc_runs/lab_calibration/` (MAPE in-sample 9.1%, LOO 14.8%);
Gassmann `2026-07-26T16:57:15Z` (MAPE 7.3%, vies +0.27 km/s).
