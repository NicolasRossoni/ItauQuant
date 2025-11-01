"""
GenerateFakeData.py

Gera datasets sintéticos usando o modelo de dois fatores de Schwartz-Smith.

Simula:
- Estados latentes X_t (Ornstein-Uhlenbeck) e Y_t (Browniano com deriva)
- Calcula F_model(t, T_i) via fórmula fechada
- Adiciona ruído para criar F_mkt = F_model + noise
- Gera grade de TTM coerente

Salva CSVs: F_mkt.csv, ttm.csv, S.csv (opcional), costs.csv (opcional)

Função pública: GenerateFakeDataset(...)
Script CLI: python src/GenerateFakeData.py --dataset-name wti_synth_01 --T 1500 --M 8

Exemplos:
----------
Fake Data:
>>> python src/GenerateFakeData.py --dataset-name wti_synth_01 --T 1500 --M 8
Real Data: (YF)
>>> python src/GenerateFakeData.py --dataset-name brent_real_01 --M 8 --use-real-data --start-date 2023-01-01 --end-date 2023-12-31

"""

import numpy as np
import pandas as pd
import os
import logging
import argparse
from typing import Tuple, Dict
import yfinance as yf
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _fetch_real_data(start_date: str, end_date: str, ticker: str = "BZ=F") -> pd.DataFrame:
    """
    Busca dados reais do Yahoo Finance.
    
    Parâmetros
    ----------
    start_date : str
        Data inicial no formato 'YYYY-MM-DD'
    end_date : str
        Data final no formato 'YYYY-MM-DD'
    ticker : str
        Símbolo do ativo (default: "BZ=F" para Brent Crude Oil)
    
    Retorna
    -------
    pd.DataFrame
        Dados históricos do ativo
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Close']  # Usando apenas o preço ajustado

def GenerateFakeDataset(
    dataset_name: str,
    T: int,
    M: int,
    seed: int = 42,
    out_root: str = None, # Complementada a seguir (L93)
    use_real_data: bool = False,
    start_date: str = None,
    end_date: str = None
) -> dict:
    '''
    Gera dataset sintético no formato Schwartz-Smith.
    
    Parâmetros
    ----------
    dataset_name : str
        Nome do dataset (ex: "wti_synth_01").
    T : int
        Número de datas (timesteps).
    M : int
        Número de tenores (maturidades).
    seed : int
        Seed para reprodutibilidade.
    out_root : str
        Diretório raiz para salvar dados.
    
    Retorna
    -------
    dict
        {
            'F_mkt_path': str,
            'ttm_path': str,
            'S_path': str,
            'costs_path': str
        }
    
    Exemplos
    --------
    >>> paths = GenerateFakeDataset("wti_synth_01", T=1500, M=8)
    >>> print(paths['F_mkt_path'])
    '''
    # Definindo Diretório da Saída
    if out_root is None:
        out_root = "data/realData" if use_real_data else "data/fakedata"

    logger.info(f"=== Gerando dataset sintético: {dataset_name} ===")
    logger.info(f"T={T}, M={M}, seed={seed}, use_real_data={use_real_data}")
    logger.info(f"Diretório de Saída: {out_root}")
    
    np.random.seed(seed)
    
    if use_real_data:
        # Buscar dados reais para determinar T
        real_prices = _fetch_real_data(start_date, end_date)
        T = len(real_prices) # T = Número real de dias

        # Retornos Log para estimar parametros
        log_returns = np.log(real_prices / real_prices.shift(1)).dropna()
        vol = np.std(log_returns) * np.sqrt(252)  # Anualizado 
        
        # @ Log - Debug 
        #Logger.info(f"Dados reais carregados: {T} dias, volatilidade anualizada estimada: {vol:.4f}")

        # Parâmetros do modelo (ground truth)
        Theta = {
            'kappa': 1.5,
            'sigma_X': vol * 0.6,
            'sigma_Y': vol * 0.4, #foco maior no curto prazo (X)
            'rho': 0.4,
            'mu': np.mean(log_returns) * 252 #drift anualizado
        }

        Y = np.log(real_prices.values)  # Log dos preços reais
        X = np.zeros(T)  # Inicializar X como desvios com o tamanho correto

        # Garantir que são arrays 1D
        Y = Y.reshape(-1)  # Reshape para garantir array 1D
        X = X.reshape(-1)  # Reshape para garantir array 1D
        
        logger.info(f"Estados baseados em dados reais: X shape={X.shape}, Y shape={Y.shape}")

    else: 
        # Parâmetros do modelo (ground truth)
        Theta = {
            'kappa': 1.5,
            'sigma_X': 0.35,
            'sigma_Y': 0.15,
            'rho': 0.4,
            'mu': 0.02
        }
        X, Y = _simulate_states(T, Theta) # Metodo antigo 

    logger.info(f"Parâmetros do modelo: {Theta}")
    
    # 1. Simular estados X_t e Y_t --------- antigos 
    # X, Y = _simulate_states(T, Theta)

    logger.info(f"Estados simulados: X shape={X.shape}, Y shape={Y.shape}")
    
    # 2. Construir grade de TTM
    ttm = _build_ttm_grid(T, M)
    logger.info(f"Grade TTM construída: shape={ttm.shape}")
    
    # 3. Calcular F_model usando fórmula fechada
    F_model_path = _forward_closed_form_path(X, Y, Theta, ttm)
    logger.info(f"F_model calculado: shape={F_model_path.shape}")
    
    # 4. Injetar ruído de mercado para criar F_mkt (apenas para dados sintéticos)
    F_mkt = F_model_path if use_real_data else _inject_market_noise(F_model_path, ttm, scheme='lognormal')
    logger.info(f"F_mkt {'sem' if use_real_data else 'com'} ruído gerado: shape={F_mkt.shape}")
    
    # 5. Calcular spot (opcional)
    S = np.exp(X + Y)
    logger.info(f"Spot S calculado: shape={S.shape}")
    
    # 6. Gerar custos (opcional)
    costs = _generate_costs(M)
    
    # 7. Salvar CSVs
    out_dir = os.path.join(out_root, dataset_name)
    paths = _save_csvs(out_dir, F_mkt, ttm, S, costs, T, M)
    
    logger.info(f"=== Dataset salvo em: {out_dir} ===")
    return paths


def _simulate_states(T: int, Theta: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simula trajetórias de X_t (OU) e Y_t (Browniano).
    
    Equações discretizadas:
    X_{t+1} = X_t * exp(-kappa*dt) + sigma_X * sqrt((1-exp(-2*kappa*dt))/(2*kappa)) * eps_X
    Y_{t+1} = Y_t + mu*dt + sigma_Y * sqrt(dt) * eps_Y
    
    com correlação rho entre eps_X e eps_Y.
    
    Parâmetros
    ----------
    T : int
    Theta : dict
    
    Retorna
    -------
    X, Y : np.ndarray [T]
    """
    kappa = Theta['kappa']
    sigma_X = Theta['sigma_X']
    sigma_Y = Theta['sigma_Y']
    rho = Theta['rho']
    mu = Theta['mu']
    
    dt = 1.0 / 252.0  # Dados diários
    
    X = np.zeros(T)
    Y = np.zeros(T)
    
    # Inicialização
    X[0] = 0.0
    Y[0] = 4.0  # Log do preço inicial ~exp(4) ≈ 54.6
    
    # Matriz de correlação para shocks
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]
    
    for t in range(1, T):
        # Gerar shocks correlacionados
        eps = np.random.multivariate_normal(mean, cov)
        eps_X, eps_Y = eps[0], eps[1]
        
        # Atualizar X (Ornstein-Uhlenbeck)
        drift_X = -kappa * X[t-1] * dt
        diffusion_X = sigma_X * np.sqrt((1 - np.exp(-2*kappa*dt)) / (2*kappa)) * eps_X
        X[t] = X[t-1] + drift_X + diffusion_X
        
        # Atualizar Y (Browniano com deriva)
        drift_Y = mu * dt
        diffusion_Y = sigma_Y * np.sqrt(dt) * eps_Y
        Y[t] = Y[t-1] + drift_Y + diffusion_Y
    
    return X, Y


def _build_ttm_grid(T: int, M: int) -> np.ndarray:
    """
    Constrói grade de time-to-maturity [T, M] em anos.
    
    Estratégia: tenores fixos (ex: 1m, 2m, 3m, 6m, 9m, 12m, 18m, 24m)
    rolando ao longo do tempo.
    
    Parâmetros
    ----------
    T : int
    M : int
    
    Retorna
    -------
    np.ndarray [T, M]
    """
    # Tenores em meses (ajustável)
    tenor_months = np.array([1, 2, 3, 6, 9, 12, 18, 24])
    
    if M > len(tenor_months):
        # Estender linearmente
        tenor_months = np.linspace(1, 24, M)
    else:
        tenor_months = tenor_months[:M]
    
    # Converter para anos
    tenors_years = tenor_months / 12.0
    
    # Grade: tenores fixos para todas as datas
    ttm = np.tile(tenors_years, (T, 1))
    
    return ttm


def _forward_closed_form_path(
    X: np.ndarray,
    Y: np.ndarray,
    Theta: dict,
    ttm: np.ndarray
) -> np.ndarray:
    """
    Calcula F_model(t, T_i) para todo histórico usando fórmula fechada.
    
    Fórmula de Schwartz-Smith:
    F(t, T) = exp(
        X_t * exp(-kappa*Delta) + Y_t + mu*Delta
        + 0.5 * [var_X + var_Y + 2*cov_XY]
    )
    
    Parâmetros
    ----------
    X : np.ndarray [T]
    Y : np.ndarray [T]
    Theta : dict
    ttm : np.ndarray [T, M]
    
    Retorna
    -------
    np.ndarray [T, M]
    """
    T, M = ttm.shape
    F_model = np.zeros((T, M))
    
    kappa = Theta['kappa']
    sigma_X = Theta['sigma_X']
    sigma_Y = Theta['sigma_Y']
    rho = Theta['rho']
    mu = Theta['mu']
    
    for t in range(T):
        for i in range(M):
            Delta = ttm[t, i]
            
            # Média
            mean_X = X[t] * np.exp(-kappa * Delta)
            mean_Y = Y[t] + mu * Delta
            
            # Variância
            var_X = (sigma_X**2 / (2*kappa)) * (1 - np.exp(-2*kappa*Delta))
            var_Y = sigma_Y**2 * Delta
            cov_XY = rho * sigma_X * sigma_Y * (1 - np.exp(-kappa*Delta)) / kappa
            
            var_total = var_X + var_Y + 2*cov_XY
            
            # Forward
            F_model[t, i] = np.exp(mean_X + mean_Y + 0.5 * var_total)
    
    return F_model


def _inject_market_noise(
    F_model: np.ndarray,
    ttm: np.ndarray,
    scheme: str = 'lognormal'
) -> np.ndarray:
    """
    Adiciona ruído para simular F_mkt.
    
    Esquemas:
    - 'lognormal': F_mkt = F_model * exp(noise)
    - 'additive': F_mkt = F_model + noise
    
    Parâmetros
    ----------
    F_model : np.ndarray [T, M]
    ttm : np.ndarray [T, M]
    scheme : str
    
    Retorna
    -------
    np.ndarray [T, M]
    """
    T, M = F_model.shape
    
    if scheme == 'lognormal':
        # Ruído proporcional ao TTM (maior incerteza para maturidades longas)
        noise_vol = 0.01 + 0.005 * ttm  # 1% base + 0.5% por ano
        noise = np.random.randn(T, M) * noise_vol
        F_mkt = F_model * np.exp(noise)
    elif scheme == 'additive':
        noise_std = 0.5 * (1 + ttm)
        noise = np.random.randn(T, M) * noise_std
        F_mkt = F_model + noise
    else:
        raise ValueError(f"Esquema '{scheme}' não suportado.")
    
    # Garantir valores positivos
    F_mkt = np.maximum(F_mkt, 0.1)
    
    return F_mkt


def _generate_costs(M: int) -> pd.DataFrame:
    """
    Gera custos sintéticos por tenor.
    
    Parâmetros
    ----------
    M : int
    
    Retorna
    -------
    pd.DataFrame
    """
    tenors = [f"T{i+1}" for i in range(M)]
    tick_values = np.random.uniform(5, 15, M)
    fees = np.random.uniform(1, 3, M)
    
    costs_df = pd.DataFrame({
        'tenor': tenors,
        'tick_value': tick_values,
        'fee_per_contract': fees
    })
    
    return costs_df


def _save_csvs(
    out_dir: str,
    F_mkt: np.ndarray,
    ttm: np.ndarray,
    S: np.ndarray,
    costs: pd.DataFrame,
    T: int,
    M: int
) -> dict:
    """
    Salva CSVs no diretório de saída.
    
    Parâmetros
    ----------
    out_dir : str
    F_mkt : np.ndarray [T, M]
    ttm : np.ndarray [T, M]
    S : np.ndarray [T]
    costs : pd.DataFrame
    T : int
    M : int
    
    Retorna
    -------
    dict com paths
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Criar índices de datas
    dates = pd.date_range(start='2020-01-01', periods=T, freq='B')
    tenor_cols = [f'tenor_{i+1}' for i in range(M)]
    
    # F_mkt.csv
    F_mkt_df = pd.DataFrame(F_mkt, index=dates, columns=tenor_cols)
    F_mkt_path = os.path.join(out_dir, 'F_mkt.csv')
    F_mkt_df.to_csv(F_mkt_path)
    logger.info(f"Salvo: {F_mkt_path}")
    
    # ttm.csv
    ttm_df = pd.DataFrame(ttm, index=dates, columns=tenor_cols)
    ttm_path = os.path.join(out_dir, 'ttm.csv')
    ttm_df.to_csv(ttm_path)
    logger.info(f"Salvo: {ttm_path}")
    
    # S.csv
    S_df = pd.DataFrame({'S': S}, index=dates)
    S_path = os.path.join(out_dir, 'S.csv')
    S_df.to_csv(S_path)
    logger.info(f"Salvo: {S_path}")
    
    # costs.csv
    costs_path = os.path.join(out_dir, 'costs.csv')
    costs.to_csv(costs_path, index=False)
    logger.info(f"Salvo: {costs_path}")
    
    return {
        'F_mkt_path': F_mkt_path,
        'ttm_path': ttm_path,
        'S_path': S_path,
        'costs_path': costs_path
    }


def main():
    """CLI para geração de datasets."""
    parser = argparse.ArgumentParser(description='Gera dataset sintético Schwartz-Smith')
    parser.add_argument('--dataset-name', type=str, default='wti_synth_01',
                        help='Nome do dataset')
    parser.add_argument('--T', type=int, default=1500,
                        help='Número de timesteps')
    parser.add_argument('--M', type=int, default=8,
                        help='Número de tenores')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--out-root', type=str, default=None,
                        help='Diretório raiz para output')
    parser.add_argument('--use-real-data', action='store_true',
                      help='Usar dados reais do Yahoo Finance')
    parser.add_argument('--start-date', type=str,
                      help='Data inicial para dados reais (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str,
                      help='Data final para dados reais (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    paths = GenerateFakeDataset(
        dataset_name=args.dataset_name,
        T=args.T,
        M=args.M,
        seed=args.seed,
        out_root=args.out_root,
        use_real_data=args.use_real_data,
        start_date=args.start_date,
        end_date=args.end_date
        )
    
    print("\n=== Dataset gerado com sucesso ===")
    for key, path in paths.items():
        print(f"{key}: {path}")


if __name__ == '__main__':
    main()
