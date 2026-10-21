# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 09:30:27 2025

@author: Irineu
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from numba import njit

@njit
def geometric_inclusion_PQ(k, mu, ka, mua, asp):
    # Tratar asp == 1.0 (esfera)
    if asp == 1.0: #Note: A implementação da esfera é mais simples no caso DEM, possibilita uma solução analítica e exata
        pa = (k - ka) / (k + (4/3) * mu)
        zeta = mu*(9*k+8*mu)/(6*(k+2*mu))
        qa = (mu - mua) / (mu + zeta) 
        return pa, qa
    
    # ******* Cálculo de theta e fn *****************
    if asp < 1.0:  # Esferoide Oblato (asp < 1.0)
      
        theta = (asp / ((1.0 - asp**2)**1.5)) * (np.arccos(asp) - asp * np.sqrt(1.0 - asp**2))
        fn = (asp**2 / (1.0 - asp**2)) * (3.0 * theta - 2.0)
    
    else:  # Esferoide Prolato (asp > 1.0)
       
        theta = (asp / ((asp**2 - 1.0)**1.5)) * (asp * np.sqrt(asp**2 - 1.0) - np.arccosh(asp))
        fn = (asp**2 / (asp**2 - 1.0)) * (2.0 - 3.0 * theta)

        
    # Poisson ratio do meio efetivo (k, mu)
    nu = (3.0 * k - 2.0 * mu) / (2.0 * (3.0 * k + mu))
    r = (1.0 - 2.0 * nu) / (2.0 * (1.0 - nu))
    
    # Parâmetros de contraste (inclusão 'a' em relação à matriz efetiva)
    a = mua / mu - 1.0
    b = (1.0 / 3.0) * (ka / k - mua / mu)
    
    # ******* Fatores F1a a F9a *****************
    # As funções Fi do paper de Mukerji (e outros) são mapeadas para fia no MATLAB
    
    f1a = 1.0 + a * ((3.0 / 2.0) * (fn + theta) - r * ((3.0 / 2.0) * fn + (5.0 / 2.0) * theta - (4.0 / 3.0)))
    
    f2a = 1.0 + a * (1.0 + (3.0 / 2.0) * (fn + theta) - (r / 2.0) * (3.0 * fn + 5.0 * theta)) + b * (3.0 - 4.0 * r)
    f2a += (a / 2.0) * (a + 3.0 * b) * (3.0 - 4.0 * r) * (fn + theta - r * (fn - theta + 2.0 * theta**2))
    
    f3a = 1.0 + a * (1.0 - (fn + (3.0 / 2.0) * theta) + r * (fn + theta))
    
    f4a = 1.0 + (a / 4.0) * (fn + 3.0 * theta - r * (fn - theta))
    
    f5a = a * (-fn + r * (fn + theta - (4.0 / 3.0))) + b * theta * (3.0 - 4.0 * r)
    
    f6a = 1.0 + a * (1.0 + fn - r * (fn + theta)) + b * (1.0 - theta) * (3.0 - 4.0 * r)
    
    f7a = 2.0 + (a / 4.0) * (3.0 * fn + 9.0 * theta - r * (3.0 * fn + 5.0 * theta)) + b * theta * (3.0 - 4.0 * r)
    
    f8a = a * (1.0 - 2.0 * r + (fn / 2.0) * (r - 1.0) + (theta / 2.0) * (5.0 * r - 3.0)) + b * (1.0 - theta) * (3.0 - 4.0 * r)
    
    f9a = a * ((r - 1.0) * fn - r * theta) + b * theta * (3.0 - 4.0 * r)
    
    # ******* Fatores P e Q (Concentration Factors) *****************
    
    # O P do artigo (T_ii*T_jj / 3) é o pa do MATLAB
    pa = 3.0 * f1a / f2a
    qa = (2.0 / f3a) + (1.0 / f4a) + ((f4a * f5a + f6a * f7a - f8a * f9a) / (f2a * f4a))
    
    # Ajuste para a forma do DEM, conforme o script MATLAB
    pa = pa / 3.0
    qa = qa / 5.0
    
    return pa, qa
    
@njit
def dem_deriv(t, y, k1, mu1, k2, mu2, asp, phic):
    """
    Função de derivada para o método Differential Effective Medium (DEM).
    Equivalente a demyprime.m.

    Parâmetros:
    t (float): Variável de integração (t_final = por / phic).
    y (array): [k, mu] - Módulos de compressão e cisalhamento efetivos no passo atual.
    k1, mu1: Módulos da matriz (fase 1).
    k2, mu2: Módulos das inclusões (fase 2).
    asp: Razão de aspecto (aspect ratio) das inclusões.
    phic: Porosidade de percolação.

    Retorna:
    yprime (array): [dK/dt, dMu/dt]
    """
    
    k, mu = y[0], y[1]
    yprime = np.zeros(2)

    # Nota: krc e murc são calculados em demyprime.m, mas parecem não ser usados
    # em demyprime.m, pois ka=k2 e mua=mu2. Vou manter o código original.
    # krc = k1 * k2 / ((1 - phic) * k2 + phic * k1)
    # murc = mu1 * mu2 / ((1 - phic) * mu2 + phic * mu1)

    ka = k2
    mua = mu2

    pa, qa = geometric_inclusion_PQ(k, mu, ka, mua, asp)
    
    # ******* Lado Direito do EDO (dK/dt e dMu/dt) *****************
    
    # t aqui representa phi / phic, onde phi é a porosidade.
    # O termo (1-t) vem da formulação DEM: dK/dphi * dphi/dt = (K_i - K) * P
    # e dphi/dt = phic (no DEM original é 1, no modificado é phic).
    # dK/dt = (Ka - K) * P / (1 - t)
    
    krhs = (ka - k) * pa
    
    # APLICAR np.real() aqui para remover a parte imaginária infinitesimal
    yprime[0] = np.real(krhs / (1.0 - t)) 
    
    murhs = (mua - mu) * qa
    # APLICAR np.real() aqui para remover a parte imaginária infinitesimal
    yprime[1] = np.real(murhs / (1.0 - t))
    
    return yprime


def _dem1_scalar(k1, mu1, k2, mu2, asp, por, phic=1):
    """
    DEM1 - Módulos elásticos efetivos usando a formulação Differential Effective Medium (DEM).
    Equivalente a dem1.m.

    [K, MU, KV, MUV, PORV] = DEM1(K1, MU1, K2, MU2, ASP, PHIC, POR)

    Parâmetros:
    k1, mu1: Módulos de compressão e cisalhamento da matriz de fundo.
    k2, mu2: Módulos de compressão e cisalhamento das inclusões.
    asp: Razão de aspecto das inclusões. <1 para oblatos; >1 para prolatos.
    phic: Porosidade de percolação para DEM modificado. =1 para DEM usual.
    por: Porosidade, fração da fase 2.

    Retorna:
    k, mu: Módulos de compressão e cisalhamento efetivos (valores finais).
    kv, muv: Arrays com os módulos durante a integração.
    porv: Array com as porosidades correspondentes.
    """
    
    # Vetor de estado inicial: y0 = [k1, mu1] (módulos da matriz)
    y0 = np.array([k1, mu1])
    
    # Variável de integração do DEM (t = phi / phic)
    # T0 (tempo inicial) = 0.0
    # Tfinal (tempo final) = por / phic
    tfinal = por / phic
    t_span = (0.0, tfinal)
    
    # Se tfinal for muito pequeno (por/phic ~ 0), o resolvedor pode falhar.
    # Pode ser necessário um tratamento de erro ou um valor mínimo.
    if tfinal <= 0.0:
        return k1, mu1, np.array([k1]), np.array([mu1]), np.array([0.0])

    # Argumentos adicionais para a função de derivada (DEMINPT global do MATLAB)
    args = (k1, mu1, k2, mu2, asp, phic)
    
    # Resolver o EDO usando solve_ivp, equivalente ao ode45m
    # rtol (tolerância relativa) é ajustado para 1e-5 (equivalente ao ode45m)
    # Para obter uma saída 'tout' densa, precisamos definir t_eval.
    # No MATLAB, o ode45 gera pontos automaticamente. Vamos usar o padrão,
    # que é o comportamento mais próximo de um resolvedor adaptativo.
    
    # Ajuste de tolerância e método
    tol_abs = 1e-10  # Embora ode45m tenha 1e-5, uma boa prática é usar atol/rtol
    tol_rel = 1e-5
    
    # A maneira mais simples de obter um resultado com o passo adaptativo do ode45
    # é não fornecer t_eval.
    sol = solve_ivp(
        dem_deriv, 
        t_span, 
        y0, 
        method='RK45', 
        args=args, 
        rtol=tol_rel, 
        atol=tol_abs
    )
    
    # Resultados
    # Se a solução foi bem-sucedida:
    if sol.success:
        # tout (t)
        tout = sol.t
        # yout (y)
        yout = sol.y.T # Transpor para ter [num_passos, num_variaveis]
        
        # Variáveis de saída do dem1.m:
        # k, mu: Último valor
        k = np.real(yout[-1, 0])
        mu = np.real(yout[-1, 1])
        
        # kv, muv: Todos os valores
        kv = np.real(yout[:, 0])
        muv = np.real(yout[:, 1])
        
        # porv: Porosidade no passo
        porv = phic * tout
    else:
        # Lidar com falha na integração, se necessário
        print(f"Failure solving ODE RK45 method (solve_ivp): {sol.message}")
        k = mu = kv = muv = porv = None
    
    return k, mu, kv, muv, porv
# Arquivo: EffectiveMedium.py (A nova função dem1)

def dem1(k1, mu1, k2, mu2, asp, por, phic=1.0):
    """
    WRAPPER DEM: Roteia o cálculo para modo Escalar (única amostra) 
    ou modo Iterador (múltiplas amostras).
    """
    # Verifica se o principal parâmetro de entrada (k1) é escalar (não-iterável)
    # Se for, assume que todos os outros parâmetros também são escalares.
    is_iterable_input = isinstance(k1, (list, np.ndarray))
    
    if not is_iterable_input:
        # MODO ESCALAR: Chama o resolvedor diretamente
        return _dem1_scalar(k1, mu1, k2, mu2, asp, por, phic)
    
    else:
        # MODO ITERADOR: Processa múltiplas amostras (sequencialmente)
        
        # 1. Ajusta o tamanho das listas de entrada
        n_samples = len(k1)
        
        # Função auxiliar para garantir que todos os parâmetros são listas do tamanho correto
        def _ensure_list_of_size_n(param, size):
            if not isinstance(param, (list, np.ndarray)):
                return [param] * size
            # Se já for uma lista, retorna a própria lista
            return param

        # 2. Garante que todos os parâmetros são listas (o phic é o mais importante aqui)
        k2_list = _ensure_list_of_size_n(k2, n_samples)
        mu2_list = _ensure_list_of_size_n(mu2, n_samples)
        asp_list = _ensure_list_of_size_n(asp, n_samples)
        por_list = _ensure_list_of_size_n(por, n_samples)
        phic_list = _ensure_list_of_size_n(phic, n_samples)
            
        # 3. Combina todas as entradas em um único iterador (conjunto de parâmetros por amostra)
        input_data = zip(k1, mu1, k2_list, mu2_list, asp_list, por_list, phic_list)
        
        # 4. Listas para armazenar todos os resultados
        k_eff_list = []
        mu_eff_list = []
        k_history_list = []
        mu_history_list = []
        phi_history_list = []
        
        # 5. Itera sobre cada amostra e chama o resolvedor escalar
        for k1_s, mu1_s, k2_s, mu2_s, asp_s, por_s, phic_s in input_data:
            k_s, mu_s, kv_s, muv_s, porv_s = _dem1_scalar(
                k1_s, mu1_s, k2_s, mu2_s, asp_s, por_s, phic_s
            )
            
            # Armazena os resultados, tratando possíveis falhas (None)
            k_eff_list.append(k_s)
            mu_eff_list.append(mu_s)
            k_history_list.append(kv_s)
            mu_history_list.append(muv_s)
            phi_history_list.append(porv_s)

        # 6. Retorna listas de resultados (uma lista por tipo de saída)
        return k_eff_list, mu_eff_list, k_history_list, mu_history_list, phi_history_list
    
if __name__ == "__main__":
    # ----------------------------------------------------
    # INÍCIO DO CRONÔMETRO
    # ----------------------------------------------------
    import time
    total_start_time = time.time()
    
    # Exemplo de uso 1 (escalar):
    # Bulk/Shear moduli for Quartz
    k_min = 37.9 # GPa
    mu_min = 44.5 # GPa
    
    # Inclusions: Dry Pores (K=0, Mu=0)
    k_pore = 0.0
    mu_pore = 0.0
    
    # Inclusions parameters
    aspect_ratio = 0.1  # Aspect ratio (e.g., flat crack)
    #percolation_phi = 1.0 # Classical DEM
    target_porosity = 0.3 # Target porosity
    
    k_eff, mu_eff, k_history, mu_history, phi_history = dem1(
        k1=k_min, mu1=mu_min, 
        k2=k_pore, mu2=mu_pore, 
        asp=aspect_ratio, por=target_porosity
    )
    
    # ----------------------------------------------------
    # FIM DO CRONÔMETRO E EXIBIÇÃO DO TEMPO TOTAL
    # ----------------------------------------------------
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print("-" * 50)
    print(f"TEMPO TOTAL DE EXECUÇÃO (Incluindo inicialização e finalização do Pool): {total_duration:.2f} segundos")
    print("-" * 50)
    
    print(f"Initial K (Matrix): {k_min:.2f} GPa, Mu: {mu_min:.2f} GPa")
    print(f"Target Porosity: {target_porosity:.2f}")
    print(f"Effective K (final): {k_eff:.2f} GPa, Mu: {mu_eff:.2f} GPa")
    
    # plot
    plt.rcParams['font.size']=14
    plt.rcParams['font.family']='arial'
    
    plt.figure(figsize=(6,6))
    plt.xlabel('Porosity')
    plt.ylabel('Elastic moduli [GPa]')
    plt.title('DEM dry-pore modelling')
    plt.plot(phi_history, k_history, 'k', label='Bulk modulus')
    plt.plot(phi_history, mu_history, 'b', label='Shear modulus')
    plt.plot(target_porosity, k_eff, 'sr', label='Target Bulk modulus')
    plt.plot(target_porosity, mu_eff, 'or', label='Target Shear modulus')
    plt.grid(ls='--')
    plt.legend()
    plt.show()
    
    # ----------------------------------------------------
    
    # Exemplo de uso2 (list): 
    # ----------------------------------------------------
    # INÍCIO DO CRONÔMETRO
    # ----------------------------------------------------
    total_start_time = time.time()
    
    # Testing 3 samples in order: Quartz, Calcite, Dolomite
    # Elastic moduli
    k_min = [37.9, 76.8, 94.9] # mineral Bulk modulus, GPa
    mu_min = [44.5, 32, 45] # mineral Shear modulus, GPa
    
    # Inclusions: Dry Pores (K=0, Mu=0)
    k_pore = mu_pore = [0] * len(k_min) # dry pores (or fluid values if saturated) - list of same condition for dry
    
    # Inclusions parameters
    aspect_ratio = [0.5, 0.1, 0.01] # pore aspect ratio
       
    target_porosity = [0.2, 0.15, 0.10] # porosity (faction)
    
    k_eff_list, mu_eff_list, k_history_list, mu_history_list, phi_history_list = dem1(
        k1=k_min, mu1=mu_min, 
        k2=k_pore, mu2=mu_pore, 
        asp=aspect_ratio, por=target_porosity
    )
    
    # ----------------------------------------------------
    # FIM DO CRONÔMETRO E EXIBIÇÃO DO TEMPO TOTAL
    # ----------------------------------------------------
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print("-" * 50)
    print(f"TEMPO TOTAL DE EXECUÇÃO (Incluindo inicialização e finalização do Pool): {total_duration:.2f} segundos")
    print("-" * 50)
    # ----------------------------------------------------
    
    # Itera sobre os resultados para plotar
    for i in range(len(k_eff_list)):
        k_eff = k_eff_list[i]
        mu_eff = mu_eff_list[i]
        k_history = k_history_list[i]
        mu_history = mu_history_list[i]
        phi_history = phi_history_list[i]
        
        asp = aspect_ratio[i]
        por = target_porosity[i]
        
        print(f"Resultado Amostra {i+1} (Asp={asp:.2f}, Por={por:.2f}): K={k_eff:.2f}, Mu={mu_eff:.2f}")

        # Plotar o resultado
        plt.figure(figsize=(6,6))
        plt.xlabel('Porosity')
        plt.ylabel('Elastic moduli [GPa]')
        plt.title(f'DEM dry-pore modelling - Amostra {i+1} (Asp={asp:.2f})')
        plt.plot(phi_history, k_history, 'k', label='Bulk modulus (K)')
        plt.plot(phi_history, mu_history, 'b', label='Shear modulus (Mu)')
        plt.plot(por, k_eff, 'sr', label='Target K (Final)')
        plt.plot(por, mu_eff, 'or', label='Target Mu (Final)')
        plt.grid(ls='--')
        plt.legend()
        plt.show()