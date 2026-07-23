# LaTeX -- Relatorio Poco 861

Guia de escrita (prosa, interpretacao de tabelas/figuras, JSON/README):
`../docs/GUIA_ESCRITA_CIENTIFICA.md`

## Arquivos principais

| Arquivo | Conteudo |
|---------|----------|
| `poco861_etapa1_ml.tex` | Etapa 1: ML baseline, perfil e plugs CT |
| `poco861_etapa1_beamer.tex` | Apresentacao Beamer da Etapa 1 (16:9) |
| `poco861_etapa2_dem_sc.tex` | Etapa 2: DEM/SC, calibracao HFU, DLIS, Gassmann |
| `poco861_etapa3_residual.tex` | Etapa 3: ML residual Vp (hibrido fisica + RF) |
| `poco861_beamer_DRP_layout_package/poco861_etapa1_beamer_DRP.tex` | Beamer Etapa 1 (layout DRP) |
| `poco861_beamer_DRP_layout_package/poco861_etapa2_beamer_DRP.tex` | Beamer Etapa 2 (layout DRP) |

## Beamer Etapa 1 (`poco861_etapa1_beamer.tex`)

Compilar: `make beamer1`. Estilo alinhado ao beamer do poço 1045 e ao guia de escrita.

Boas praticas adotadas nos slides:

- Um slide, uma mensagem; titulo do frame = conclusao (nao rotulo generico).
- Slide de roteiro apos a capa; secoes Beamer (`Contexto`, `Protocolo`, etc.).
- Figura sempre com legenda `\slidecaption{...}` abaixo.
- Fechamento em prosa ou `block` com titulo especifico (ex.: ``O que isso mostra?'').
- Tabelas enxutas; numeros iguais ao relatorio `poco861_etapa1_ml.tex`.
- Sem caminhos de arquivo no corpo dos slides.

Figuras lidas via `\graphicspath` a partir de `data/processed/ml_runs/` (ver tabela abaixo).

O Beamer (`poco861_etapa1_beamer.tex`) le as PNGs direto do pipeline via
`\graphicspath` (como o beamer 1045 aponta para `paper_clp_csgm/figures/`).

| Nome no LaTeX (relatorio `figures/`) | Origem no pipeline |
|--------------------------------------|--------------------|
| `fig_fzi_log_correlations` | `ml_runs/diagnostics_861/figures/log_features_vs_FZI_lab_corr.png` |
| `fig_fzi_oracle_vs_logs` | `ml_runs/diagnostics_861/figures/FZI_oracle_vs_logs_r2_by_fold.png` |
| `fig_target_comparison_r2` | `ml_runs/diagnostics_861/figures/target_comparison_mean_r2.png` |
| `fig_phi_rf_pred_vs_obs` | `ml_runs/well_profile/phi_lab/rf/phi_pred_vs_obs.png` |
| `fig_phi_shap_bar` | `ml_runs/well_profile/phi_lab/rf/shap_bar.png` |
| `phi_lab_oof_depth_panels.png` | `ml_runs/well_profile/phi_lab/compare/figures/` |
| `phi_lab_oof_rmse_by_depth_fold.png` | `ml_runs/well_profile/phi_lab/compare/figures/` |
| `fig_phi_model_comparison` | `ml_runs/well_profile/phi_lab/alternatives/phi_model_comparison_mean_r2.png` |
| `fig_hfu_classifier_oof` | `ml_runs/well_profile/hfu/classifier/hfu_model_comparison_oof.png` |
| `fig_ct_plugs_fzi_wireline` | `ml_runs/ct_plugs/by_target/FZI_lab/wireline_only/plug_out_pred_vs_obs_rf.png` |

Opcional: copiar com os nomes `fig_*` para `latex/figures/` (relatorio PDF).

## Figuras Etapa 2 (`figures/fig2_*`)

Copiadas de `data/processed/dem_sc_runs/` e `data/processed/dlis_861/`.

## Figuras Etapa 3 (`figures/fig3_*`)

Copiadas de `data/processed/ml_runs/residual_vp/` (RF) e
`data/processed/ml_runs/residual_vp/clp_csgm/` (CLP Etapa 3b).

| Nome LaTeX | Origem |
|------------|--------|
| `fig3_clp_comparison_mape` | `clp_csgm/figures/comparison_mape_bar.png` |
| `fig3_clp_vp_hybrid_depth` | `clp_csgm/figures/vp_depth_rf_vs_clp.png` |
| `fig3_clp_three_methods_depth` | `clp_csgm/figures/vp_depth_three_methods.png` |
| `fig3_clp_residual_scatter` | `clp_csgm/figures/clp_ridge_residual_scatter.png` |
| `fig3_rho_sweep_mape` | `clp_csgm/rho_subsample/figures/rho_sweep_mape.png` |

## Beamer Etapa 2 DRP (`poco861_etapa2_beamer_DRP.tex`)

Compilar: `make beamer2-drp` (a partir de `methods_comparison/latex`).

Mesmo layout DRP da Etapa 1 (capa, rodape, Carlito, blocos). Roteiro
metodologico alinhado a `poco861_etapa2_dem_sc.tex`:

1. Heranca Etapa 1
2. Cadeia DEM/SC, premissas e ordem das inclusoes (literatura)
3. Escalas de validacao
4. Calibracao lab (linha de base, in-sample, LOO, multiescala 2f)
5. Perfil + DSI por HFU
6. Gassmann (2b) e NMR multiescala (2g)
7. Conclusoes e caminho Etapa 3

Figuras: `latex/figures/fig2_*` via `\graphicspath`.

## Compilar

```bash
cd methods_comparison/latex
make beamer1
make beamer2-drp
make etapa2
# ou: make (etapa1 + etapa2 + etapa3)
make etapa3
```

Requisitos: `pdflatex`, pacotes `babel`, `booktabs`, `beamer`, `tikz`, `siunitx`, `graphicx`, `carlito`.
