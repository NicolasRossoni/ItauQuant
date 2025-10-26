"""
PrepareTradingInputs.py

Prepara os insumos necessários para o motor de decisão de trading:
- Mispricing (DeltaF = F_model - F_mkt)
- Matriz de risco Sigma (covariância de retornos)
- Limites por tenor
- Limiares de entrada/saída
- Fricções (custos)

Função pública: PrepareTradingInputs(...)
"""

import numpy as np
import pandas as pd
import logging
from typing import Union, Optional, Dict
from sklearn.covariance import LedoitWolf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def PrepareTradingInputs(
    F_mkt_t: np.ndarray,
    F_model_t: np.ndarray,
    ttm_t: np.ndarray,
    cost: Union[np.ndarray, pd.Series, None],
    cfg: dict,
    F_mkt_hist: Optional[np.ndarray] = None,
    F_model_hist: Optional[np.ndarray] = None
) -> dict:
    """
    Prepara insumos para o motor de trading a partir de forwards de mercado e modelo.
    
    Parâmetros
    ----------
    F_mkt_t : np.ndarray [M]
        Forwards de mercado no tempo t*.
    F_model_t : np.ndarray [M]
        Forwards do modelo no tempo t*.
    ttm_t : np.ndarray [M]
        Time-to-maturity em anos para cada tenor em t*.
    cost : np.ndarray, pd.Series ou None [M]
        Custos por tenor (opcional).
    cfg : dict
        Configurações contendo 'risk', 'limits', 'thresh', 'costs'.
    F_mkt_hist : np.ndarray [T, M], opcional
        Histórico de forwards de mercado para estimar covariância.
    F_model_hist : np.ndarray [T, M], opcional
        Histórico de forwards do modelo para estimar covariância.
    
    Retorna
    -------
    dict
        {
            'mispricing': np.ndarray [M],
            'Sigma': np.ndarray [M, M],
            'limits': np.ndarray [M],
            'thresh': np.ndarray [M],
            'frictions': dict
        }
    
    Exemplos
    --------
    >>> inputs = PrepareTradingInputs(F_mkt_t, F_model_t, ttm_t, None, cfg)
    >>> print(inputs['mispricing'])
    >>> print(inputs['Sigma'])
    """
    logger.info("=== Iniciando PrepareTradingInputs ===")
    
    M = len(F_mkt_t)
    
    # 1. Calcular mispricing
    mispricing = _compute_mispricing(F_model_t, F_mkt_t)
    logger.info(f"Mispricing calculado: média={np.mean(mispricing):.4f}, "
                f"std={np.std(mispricing):.4f}")
    
    # 2. Estimar matriz de covariância
    Sigma = _estimate_covariance(
        F_mkt_hist, F_model_hist, cfg.get('risk', {}), M
    )
    logger.info(f"Matriz Sigma estimada: shape={Sigma.shape}, "
                f"eigenvalues min={np.min(np.linalg.eigvals(Sigma)):.6f}")
    
    # 3. Derivar limites e limiares
    limits, thresh = _derive_limits_and_thresholds(cfg, Sigma, cost, M)
    logger.info(f"Limites: {limits}")
    logger.info(f"Limiares (thresh): {thresh}")
    
    # 4. Construir fricções
    frictions = _build_frictions(cost, cfg, M)
    logger.info(f"Fricções: {frictions}")
    
    result = {
        'mispricing': mispricing,
        'Sigma': Sigma,
        'limits': limits,
        'thresh': thresh,
        'frictions': frictions
    }
    
    logger.info("=== PrepareTradingInputs concluído ===")
    return result


def _compute_mispricing(F_model_t: np.ndarray, F_mkt_t: np.ndarray) -> np.ndarray:
    """
    Calcula DeltaF percentual = (F_model - F_mkt) / F_mkt.
    
    Importante: Retorna mispricing em termos percentuais para ser compatível
    com Sigma calculada em log-retornos.
    
    Parâmetros
    ----------
    F_model_t : np.ndarray [M]
    F_mkt_t : np.ndarray [M]
    
    Retorna
    -------
    np.ndarray [M]
        Mispricing percentual (ex: 0.02 = 2%)
    """
    # Mispricing percentual para comparar com retornos
    mispricing_pct = (F_model_t - F_mkt_t) / (F_mkt_t + 1e-10)
    return mispricing_pct


def _estimate_covariance(
    F_mkt_hist: Optional[np.ndarray],
    F_model_hist: Optional[np.ndarray],
    risk_cfg: dict,
    M: int
) -> np.ndarray:
    """
    Estima matriz de covariância de retornos/resíduos.
    
    Estratégias:
    1. Se histórico disponível: usar janela lookback com shrinkage Ledoit-Wolf
    2. Caso contrário: diagonal com volatilidades estimadas
    
    Parâmetros
    ----------
    F_mkt_hist : np.ndarray [T, M] ou None
    F_model_hist : np.ndarray [T, M] ou None
    risk_cfg : dict
        {'source': 'returns'|'residuals', 'lookback': int, 'shrinkage': bool}
    M : int
        Número de tenores
    
    Retorna
    -------
    np.ndarray [M, M]
        Matriz de covariância (SPD - semi-positive definite)
    """
    source = risk_cfg.get('source', 'returns')
    lookback = risk_cfg.get('lookback', 60)
    use_shrinkage = risk_cfg.get('shrinkage', True)
    
    if F_mkt_hist is not None and len(F_mkt_hist) >= lookback:
        logger.info(f"Estimando Sigma com {source} usando lookback={lookback}")
        
        # Usar janela de lookback
        window = F_mkt_hist[-lookback:, :]
        
        if source == 'returns':
            # Log-retornos
            returns = np.diff(np.log(window + 1e-10), axis=0)
        elif source == 'residuals' and F_model_hist is not None:
            # Resíduos F_model - F_mkt
            window_model = F_model_hist[-lookback:, :]
            residuals = window_model - window
            returns = np.diff(residuals, axis=0)
        else:
            # Fallback: retornos simples
            returns = np.diff(window, axis=0) / (window[:-1, :] + 1e-10)
        
        # Remover NaNs/Infs
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
        
        if use_shrinkage and returns.shape[0] > M:
            # Ledoit-Wolf shrinkage
            try:
                lw = LedoitWolf()
                Sigma = lw.fit(returns).covariance_
                logger.info("Shrinkage de Ledoit-Wolf aplicado")
            except Exception as e:
                logger.warning(f"Ledoit-Wolf falhou: {e}. Usando covariância empírica.")
                Sigma = np.cov(returns, rowvar=False)
        else:
            # Covariância empírica
            Sigma = np.cov(returns, rowvar=False)
        
        # Garantir que é SPD
        Sigma = _ensure_spd(Sigma)
        
    else:
        logger.warning("Histórico insuficiente. Usando Sigma diagonal com vol padrão.")
        # Fallback: diagonal com volatilidade padrão
        default_vol = 0.02  # 2% vol diária
        Sigma = np.eye(M) * (default_vol ** 2)
    
    return Sigma


def _ensure_spd(Sigma: np.ndarray, min_eig: float = 1e-3) -> np.ndarray:
    """
    Garante que a matriz seja semi-positive definite.
    
    Método: ajusta autovalores negativos para min_eig.
    
    Parâmetros
    ----------
    Sigma : np.ndarray [M, M]
    min_eig : float
        Autovalor mínimo permitido (aumentado para 1e-3 para evitar z-scores absurdos)
    
    Retorna
    -------
    np.ndarray [M, M]
    """
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    
    # Corrigir autovalores muito baixos (causa z-scores inflados)
    eigvals_corrected = np.maximum(eigvals, min_eig)
    
    # Reconstruir matriz
    Sigma_spd = eigvecs @ np.diag(eigvals_corrected) @ eigvecs.T
    
    # Tornar simétrica
    Sigma_spd = (Sigma_spd + Sigma_spd.T) / 2
    
    return Sigma_spd


def _derive_limits_and_thresholds(
    cfg: dict,
    Sigma: np.ndarray,
    cost: Union[np.ndarray, None],
    M: int
) -> tuple:
    """
    Deriva limites e limiares por tenor.
    
    Limites:
    - Baseados em leverage total e cap por tenor
    
    Limiares:
    - z_in, z_out para histerese (podem ser ajustados por tenor)
    
    Parâmetros
    ----------
    cfg : dict
    Sigma : np.ndarray [M, M]
    cost : np.ndarray [M] ou None
    M : int
    
    Retorna
    -------
    limits : np.ndarray [M]
    thresh : np.ndarray [M] ou dict
    """
    limits_cfg = cfg.get('limits', {})
    thresh_cfg = cfg.get('thresh', {})
    
    # Limites por tenor
    leverage = limits_cfg.get('leverage', 3.0)
    per_tenor_cap = limits_cfg.get('per_tenor_cap', 0.3)
    
    # Limite uniforme por tenor (pode ser refinado com VaR)
    limits = np.ones(M) * per_tenor_cap * leverage
    
    # Limiares (z-scores)
    z_in = thresh_cfg.get('z_in', 1.5)
    z_out = thresh_cfg.get('z_out', 0.5)
    
    # Retornar como dict (mais flexível para TradeEngine)
    thresh = {
        'z_in': np.ones(M) * z_in,
        'z_out': np.ones(M) * z_out
    }
    
    return limits, thresh


def _build_frictions(
    cost: Union[np.ndarray, None],
    cfg: dict,
    M: int
) -> dict:
    """
    Consolida custos efetivos (tick, fee, slippage).
    
    Parâmetros
    ----------
    cost : np.ndarray [M] ou None
    cfg : dict
    M : int
    
    Retorna
    -------
    dict
        {
            'tick_value': np.ndarray [M],
            'fee': np.ndarray [M],
            'slippage': float
        }
    """
    costs_cfg = cfg.get('costs', {})
    default_tick = costs_cfg.get('default_tick_value', 10.0)
    default_fee = costs_cfg.get('default_fee', 2.0)
    
    if cost is not None:
        tick_value = np.array(cost)
        fee = np.array(cost) * 0.2  # 20% do tick como fee (exemplo)
    else:
        tick_value = np.ones(M) * default_tick
        fee = np.ones(M) * default_fee
    
    frictions = {
        'tick_value': tick_value,
        'fee': fee,
        'slippage': 0.0  # Pode ser estimado de dados históricos
    }
    
    return frictions
