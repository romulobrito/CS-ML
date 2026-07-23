# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 15:35:23 2025

@author: Irineu
"""

import numpy as np
from EffectiveMedium import dem1 

def aspect_ratio_Image_dem_multiscale(vp_m, k1, mu1, ro1, k2, mu2, ro2, phi, frac_inc_known: list, asp_inc_known: list, fmic):
    """
    Função de inversão da Razão de Aspecto (Aspect Ratio) de Microporos (P3)
    usando o Modelo DEM de Múltiplas Inclusões (Multiscale) com ordenação unificada.

    A ordem das inclusões é determinada pela maior frequência, incluindo fmic 
    (a fração do poro invertido), garantindo uma sequência DEM consistente.

    Parâmetros de entrada:
    vp_m: Velocidade P medida (observada)
    k1, mu1, ro1: Módulos e densidade do mineral (matriz)
    k2, mu2, ro2: Módulos e densidade do fluido/inclusão (poro)
    phi: Porosidade total
    frac_inc_known (list): Lista de frações de poros conhecidas (fmac, fmic_p1, fmic_p2, ...)
    asp_inc_known (list): Lista de Aspect Ratios correspondentes às frações de poros conhecidas (aspmac, aspmic_p1, aspmic_p2, ...)
    fmic: Fração dos Microporos (P3) - AR é invertido (asp)
    
    Retorna:
    vpa, vsa: Velocidades calculadas no melhor ajuste
    rho: Densidade calculada
    aspmic: Razão de aspecto invertida para microporos (P3) (melhor ajuste)
    fit: Taxa de ajuste (1 - Erro Relativo Mínimo)
    """
    
    # Validação de entrada
    if len(frac_inc_known) != len(asp_inc_known):
        raise ValueError("As listas 'frac_inc_known' e 'asp_inc_known' devem ter o mesmo número de elementos.")

    # erro atual - inicia alto
    ea = 10000.0

    passo = 0.001
    
    # Inicialização das variáveis de retorno
    vpa = None
    vsa = None
    rho = None
    aspmic = 0.01
    
    asp_testes = np.arange(0.01, 1.0 + passo, passo) 
    phiperc = 1.0 # Parâmetro padrão para o DEM Clássico (phic)

    for asp in asp_testes:
        
        # 1. COMBINAR TODAS AS INCLUSÕES: conhecidas + a que está sendo invertida (fmic)
        
        # Cria uma lista de todas as frações de poros (Fmac, P1, P2, P3)
        all_fractions = np.array(frac_inc_known + [fmic])
        
        # Cria uma lista de todos os Aspect Ratios (ARs conhecidos + AR sendo testado 'asp')
        all_aspect_ratios = np.array(asp_inc_known + [asp])
        
        # 2. ORDENAR AS INCLUSÕES
        
        # Ordena os índices das frações combinadas do maior para o menor.
        idx_ord_unified = np.argsort(all_fractions)[::-1]
        
        kdem = k1
        mudem = mu1
        
        # 3. DEM SEQUENCIAL ÚNICO (aplica na ordem de maior frequência)
        
        for i in range(len(idx_ord_unified)):
            idx = idx_ord_unified[i].item()
            frac = all_fractions[idx].item()
            aspect_r = all_aspect_ratios[idx].item()
            
            # Aplica o DEM: o resultado se torna a matriz para a próxima inclusão
            kdem, mudem, _, _, _ = dem1(kdem, mudem, k2, mu2, aspect_r, frac * phi, phic=phiperc)
            
        # 4. Cálculo das velocidades e erro
        rot = (1.0 - phi) * ro1 + phi * ro2
        
        vpt = np.sqrt((kdem + (4.0 * mudem / 3.0)) / rot)
        vst = np.sqrt(mudem / rot)
        
        ec = np.abs(vpt - vp_m) / vp_m # Erro relativo corrente
        
        if ec > ea: 
            # Se o erro parar de diminuir e começar a incrementar, Pare!
            break
        else: 
            # Guarda o erro corrente e os resultados
            ea = ec 
            vpa = vpt
            vsa = vst
            rho = rot
            aspmic = asp # Armazena o melhor AR encontrado
    
    # Taxa de acerto/ajuste
    fit = 1.0 - ea

    return vpa, vsa, rho, aspmic, fit

# ----------------------------------------------------
# EXEMPLO DE USO (Dados de Teste)
# ----------------------------------------------------

if __name__ == "__main__":
    
    # Dados de entrada (Exemplo)
    #F7289H
    vp_m = 5.087
    k1 = 68.0
    mu1 = 32.58
    ro1 = 2.719
    
    phi = 0.1242
    
    k2 = 0.0; mu2 = 0.0; ro2 = 0.001; 
    
    # Inclusões conhecidas (fmac, fmic_p1, fmic_p2) - suposição de 3 conhecidas na resolução microCT
    frac_inc_known = [0.7811, 0.0148, 0.0149] 
    asp_inc_known = [0.5562, 0.5760, 0.6196] 

    # Fração da inclusão a ser invertida (P3) - microporos abaixo da resolução microCT
    fmic = 1-sum(frac_inc_known)
    #fmic = 0.1892 
    
    # Chamada da função unificada
    vpa_out, vsa_out, rho_out, aspmic_out, fit_out = aspect_ratio_Image_dem_multiscale(
        vp_m, k1, mu1, ro1, k2, mu2, ro2, phi, 
        frac_inc_known, asp_inc_known, fmic
    )

    # Impressão dos resultados
    print("-" * 50)
    print("RESULTADOS DA INVERSÃO DEM MULTISCALE (Versão Unificada)")
    print("-" * 50)
    print(f"Vp Medida: {vp_m:.4f} km/s")
    print(f"Razão de Aspecto Invertida (P3 - aspmic): {aspmic_out:.4f}")
    print(f"Vp Calculada (Vp_a): {vpa_out:.4f} km/s")
    print(f"Taxa de Ajuste (fit): {fit_out * 100:.2f}%")
    print("-" * 50)