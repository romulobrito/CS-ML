# Build PDF reports for Poco 861
MAIN1 = poco861_etapa1_ml
BEAMER1 = poco861_etapa1_beamer
MAIN2 = poco861_etapa2_dem_sc
MAIN3 = poco861_etapa3_residual
DRP_DIR = poco861_beamer_DRP_layout_package
BEAMER1_DRP = poco861_etapa1_beamer_DRP
BEAMER2_DRP = poco861_etapa2_beamer_DRP

.PHONY: all etapa1 beamer1 beamer1-drp beamer2-drp etapa2 etapa3 clean

all: etapa1 etapa2 etapa3

etapa1:
	pdflatex -interaction=nonstopmode $(MAIN1).tex
	pdflatex -interaction=nonstopmode $(MAIN1).tex

beamer1:
	pdflatex -interaction=nonstopmode $(BEAMER1).tex
	pdflatex -interaction=nonstopmode $(BEAMER1).tex

beamer1-drp:
	cd $(DRP_DIR) && pdflatex -interaction=nonstopmode $(BEAMER1_DRP).tex
	cd $(DRP_DIR) && pdflatex -interaction=nonstopmode $(BEAMER1_DRP).tex

beamer2-drp:
	cd $(DRP_DIR) && pdflatex -interaction=nonstopmode $(BEAMER2_DRP).tex
	cd $(DRP_DIR) && pdflatex -interaction=nonstopmode $(BEAMER2_DRP).tex

etapa2:
	pdflatex -interaction=nonstopmode $(MAIN2).tex
	pdflatex -interaction=nonstopmode $(MAIN2).tex

etapa3:
	pdflatex -interaction=nonstopmode $(MAIN3).tex
	pdflatex -interaction=nonstopmode $(MAIN3).tex

clean:
	rm -f $(MAIN1).aux $(MAIN1).log $(MAIN1).out $(MAIN1).toc
	rm -f $(BEAMER1).aux $(BEAMER1).log $(BEAMER1).nav $(BEAMER1).out $(BEAMER1).snm $(BEAMER1).toc
	rm -f $(MAIN2).aux $(MAIN2).log $(MAIN2).out $(MAIN2).toc
	rm -f $(MAIN3).aux $(MAIN3).log $(MAIN3).out $(MAIN3).toc
	rm -f $(DRP_DIR)/$(BEAMER1_DRP).aux $(DRP_DIR)/$(BEAMER1_DRP).log \
	      $(DRP_DIR)/$(BEAMER1_DRP).nav $(DRP_DIR)/$(BEAMER1_DRP).out \
	      $(DRP_DIR)/$(BEAMER1_DRP).snm $(DRP_DIR)/$(BEAMER1_DRP).toc
	rm -f $(DRP_DIR)/$(BEAMER2_DRP).aux $(DRP_DIR)/$(BEAMER2_DRP).log \
	      $(DRP_DIR)/$(BEAMER2_DRP).nav $(DRP_DIR)/$(BEAMER2_DRP).out \
	      $(DRP_DIR)/$(BEAMER2_DRP).snm $(DRP_DIR)/$(BEAMER2_DRP).toc
