"""
TradingStrategy.py

Implementação completa da estratégia de trading baseada em mispricing do modelo Schwartz-Smith.
Combina preparação de inputs de trading e motor de decisão.

FUNÇÕES PÚBLICAS PRINCIPAIS:
- PrepareTradingInputs(...): prepara mispricing, matriz de risco, limites, etc.
- TradeEngine(...): gera sinais, posições-alvo e ordens executáveis

ATENÇÃO: Este módulo implementa a estratégia de trading baseada em mispricing do modelo Schwartz-Smith.
Segue os princípios de modularidade, documentação clara e I/O determinístico.
"""

import numpy as np
import pandas as pd
import logging
from typing import Union, Optional, Dict, List, Tuple
from scipy.optimize import minimize
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
    
    Esta é uma das FUNÇÕES PÚBLICAS PRINCIPAIS do módulo TradingStrategy.py.
    
    Parâmetros
    ----------
    F_mkt_t : np.ndarray [M]
        Forwards de mercado no tempo t* (preços dos futuros observados).
    F_model_t : np.ndarray [M]
        Forwards teóricos do modelo no tempo t* (resultado do ComputeModelForward).
    ttm_t : np.ndarray [M]
        Time-to-maturity em anos para cada tenor em t*.
    cost : np.ndarray, pd.Series ou None [M]
        Custos de transação por tenor (opcional).
    cfg : dict
        Configurações contendo seções 'risk', 'limits', 'thresh', 'costs'.
    F_mkt_hist : np.ndarray [T, M], opcional
        Histórico de forwards de mercado para estimar covariância.
    F_model_hist : np.ndarray [T, M], opcional
        Histórico de forwards do modelo para estimar covariância.
    
    Retorna
    -------
    dict
        {
            'mispricing': np.ndarray [M] -> DeltaF = F_model - F_mkt,
            'Sigma': np.ndarray [M, M] -> matriz de covariância,
            'limits': np.ndarray [M] -> limites de posição,
            'thresh': dict -> {'z_in': float/array, 'z_out': float/array},
            'frictions': dict -> custos consolidados
        }
    
    Exemplos
    --------
    >>> import yaml
    >>> with open('config/default.yaml', 'r') as f:
    ...     cfg = yaml.safe_load(f)
    >>> inputs = PrepareTradingInputs(F_mkt_t, F_model_t, ttm_t, None, cfg)
    >>> print(f"Mispricing médio: {np.mean(inputs['mispricing']):.4f}")
    >>> print(f"Volatilidade média: {np.mean(np.sqrt(np.diag(inputs['Sigma']))):.4f}")
    """
    logger.info("=== Iniciando PrepareTradingInputs ===")
    
    M = len(F_mkt_t)
    
    # Validar inputs
    if len(F_model_t) != M or len(ttm_t) != M:
        raise ValueError(f"Shapes incompatíveis: F_mkt_t={M}, F_model_t={len(F_model_t)}, ttm_t={len(ttm_t)}")
    
    # 1. Calcular mispricing (Delta F)
    mispricing = _compute_mispricing(F_model_t, F_mkt_t)
    logger.info(f"Mispricing calculado: média={np.mean(mispricing):.4f}, std={np.std(mispricing):.4f}, "
                f"min={np.min(mispricing):.4f}, max={np.max(mispricing):.4f}")
    
    # 2. Estimar matriz de covariância
    risk_cfg = cfg.get('risk', {})
    Sigma = _estimate_covariance(F_mkt_hist, F_model_hist, risk_cfg, M)
    logger.info(f"Matriz Sigma estimada: shape={Sigma.shape}, eigenvalues min={np.min(np.linalg.eigvals(Sigma)):.6f}, "
                f"max={np.max(np.linalg.eigvals(Sigma)):.6f}")
    
    # 🔧 DEBUG: Valores detalhados de Sigma
    sigma_diag = np.sqrt(np.diag(Sigma))
    logger.info(f"DEBUG: Sigma diagonal (std): {np.sqrt(np.diag(Sigma))}")
    logger.info(f"DEBUG: Mispricing values: {mispricing}")
    logger.info(f"DEBUG: Momentum signals will be: {np.sign(mispricing)}")
    
    # 3. Derivar limites e limiares
    limits, thresh = _derive_limits_and_thresholds(cfg, Sigma, cost, M)
    logger.info(f"Limites por tenor: {limits}")
    logger.info(f"Limiares: z_in={thresh.get('z_in', 'N/A')}, z_out={thresh.get('z_out', 'N/A')}")
    
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


def TradeEngine(
    mispricing: np.ndarray,
    Sigma: np.ndarray,
    limits: np.ndarray,
    thresh: Union[np.ndarray, dict],
    frictions: dict,
    method: str = "vol_target",
    topK: Optional[int] = None,
    w_prev: Optional[np.ndarray] = None,
    cfg: Optional[dict] = None
) -> dict:
    """
    Motor de decisão de trading que transforma mispricing + risco + limites em ordens executáveis.
    
    🚀 ESTRATÉGIA MOMENTUM PURA:
    - Sinal = sign(mispricing) (sem z-scores ou histerese)
    - Se F_modelo > F_mercado → COMPRAR (momentum de alta)
    - Se F_modelo < F_mercado → VENDER (momentum de baixa)
    - Dimensionamento via vol-target
    
    Parâmetros
    ----------
    mispricing : np.ndarray [M]
        Diferenças F_modelo - F_mercado para cada tenor.
    Sigma : np.ndarray [M, M]
        Matriz de covariância dos retornos.
    limits : np.ndarray [M]
        Limites máximos de posição por tenor.
    thresh : dict ou np.ndarray
        (IGNORADO na estratégia momentum)
    frictions : dict
        Custos de transação e fricções.
    method : str
        Método de dimensionamento ("vol_target" ou "qp").
    topK : int, opcional
        Número máximo de tenores ativos.
    w_prev : np.ndarray [M], opcional
        Posições anteriores.
    cfg : dict, opcional
        Configurações adicionais.
    
    Retorna
    -------
    dict
        Contém 'signals', 'target_w', 'orders', 'mispricing'.
    """
    logger.info("=== Iniciando TradeEngine (MOMENTUM STRATEGY) ===")
    
    M = len(mispricing)
    w_prev = w_prev if w_prev is not None else np.zeros(M)
    
    # 1. 🚀 ESTRATÉGIA MOMENTUM: Sinal = sign(mispricing)
    signals = np.sign(mispricing)
    
    # 🔧 REDUZIDO: Threshold mais sensível para mais sinais
    noise_threshold = 0.05  # $0.05 por barril (era $0.10)
    signals = np.where(np.abs(mispricing) >= noise_threshold, signals, 0)
    
    logger.info(f"Mispricing: min={np.min(mispricing):.3f}, max={np.max(mispricing):.3f}, mean={np.mean(mispricing):.3f}")
    
    # 2. Selecionar tenores ativos (topK se especificado)
    if topK is not None and topK < M:
        # Selecionar os K maiores mispricings em valor absoluto
        active_mask = _select_topK_by_abs_mispricing(mispricing, topK)
        signals = signals * active_mask.astype(int)
        logger.info(f"Seleção topK={topK}: {np.sum(active_mask)} tenores ativos")
    
    signal_counts = {
        'BUY': np.sum(signals > 0),
        'SELL': np.sum(signals < 0), 
        'HOLD': np.sum(signals == 0)
    }
    logger.info(f"Sinais: BUY={signal_counts['BUY']}, SELL={signal_counts['SELL']}, HOLD={signal_counts['HOLD']}")
    
    # 3. Dimensionar posições
    logger.info(f"Dimensionamento: {method}")
    if method == "vol_target":
        target_w = _size_positions_vol_target(signals, Sigma, cfg or {}, M)
    elif method == "qp":
        target_w = _optimize_qp_momentum(mispricing, Sigma, limits, frictions, w_prev, cfg or {})
    else:
        raise ValueError(f"Método de dimensionamento inválido: {method}")
    
    # 4. Aplicar limites de posição
    target_w = _apply_limits(target_w, limits)
    logger.info(f"Target weights (após limites): soma={np.sum(np.abs(target_w)):.4f}, max={np.max(np.abs(target_w)):.4f}")
    
    # 5. Gerar ordens baseadas na diferença com posições anteriores
    orders = _build_orders(target_w, w_prev, frictions)
    logger.info(f"Ordens geradas: {len(orders)} operações")
    
    # Debug se não há ordens
    if len(orders) == 0:
        delta_w = target_w - w_prev
        logger.info(f"DEBUG: target_w = {target_w}")
        logger.info(f"DEBUG: w_prev = {w_prev}")
        logger.info(f"DEBUG: delta_w = {delta_w}")
        logger.info(f"DEBUG: abs(delta_w) = {np.abs(delta_w)}")
        logger.info(f"DEBUG: max abs(delta_w) = {np.max(np.abs(delta_w)):.6f}")
    
    result = {
        'signals': signals,
        'target_w': target_w,
        'orders': orders,
        'mispricing': mispricing  # Para análise posterior (em vez de z_scores)
    }
    
    logger.info("=== TradeEngine concluído ===")
    return result


# ==========================================
# FUNÇÕES AUXILIARES (PRIVADAS)
# ==========================================

def _compute_mispricing(F_model_t: np.ndarray, F_mkt_t: np.ndarray) -> np.ndarray:
    """
    Calcula mispricing: DeltaF = F_modelo - F_mercado.
    
    Parâmetros
    ----------
    F_model_t : np.ndarray [M]
    F_mkt_t : np.ndarray [M]
    
    Retorna
    -------
    np.ndarray [M]
    """
    return F_model_t - F_mkt_t


def _estimate_covariance(
    F_mkt_hist: Optional[np.ndarray],
    F_model_hist: Optional[np.ndarray],
    risk_cfg: dict,
    M: int
) -> np.ndarray:
    """
    Estima matriz de covariância usando retornos históricos ou fallback simples.
    
    Parâmetros
    ----------
    F_mkt_hist : np.ndarray [T, M] ou None
    F_model_hist : np.ndarray [T, M] ou None
    risk_cfg : dict
        {'source': 'returns'|'residuals', 'lookback': int, 'shrinkage': bool}
    M : int
    
    Retorna
    -------
    np.ndarray [M, M]
    """
    source = risk_cfg.get('source', 'returns')
    lookback = risk_cfg.get('lookback', 60)
    use_shrinkage = risk_cfg.get('shrinkage', True)
    
    if F_mkt_hist is not None and len(F_mkt_hist) > lookback:
        logger.info(f"Usando {source} para estimar covariância com janela de {lookback} dias")
        
        # Selecionar janela de dados
        hist_window = F_mkt_hist[-lookback:] if len(F_mkt_hist) > lookback else F_mkt_hist
        logger.info(f"DEBUG: hist_window shape: {hist_window.shape}")
        
        # Calcular retornos
        if source == 'returns':
            returns = np.diff(np.log(hist_window), axis=0)  # Log-returns
        else:  # residuals
            if F_model_hist is not None:
                model_window = F_model_hist[-lookback:] if len(F_model_hist) > lookback else F_model_hist
                residuals = hist_window - model_window
                returns = np.diff(residuals, axis=0)
            else:
                logger.warning("F_model_hist não disponível para residuals. Usando returns.")
                returns = np.diff(np.log(hist_window), axis=0)
        
        logger.info(f"DEBUG: returns shape: {returns.shape}")
        logger.info(f"DEBUG: returns std por tenor: {np.std(returns, axis=0)}")
        logger.info(f"DEBUG: returns mean por tenor: {np.mean(returns, axis=0)}")
        
        # Estimar covariância com shrinkage (se habilitado)
        if use_shrinkage:
            try:
                lw = LedoitWolf()
                Sigma, _ = lw.fit(returns).covariance_, lw.shrinkage_
                logger.info(f"DEBUG: Shrinkage usado, shrinkage factor: {_}")
            except Exception as e:
                logger.warning(f"Shrinkage falhou: {e}. Usando covariância empírica.")
                Sigma = np.cov(returns, rowvar=False)
        else:
            Sigma = np.cov(returns, rowvar=False)
        
        logger.info(f"DEBUG: Sigma raw diagonal: {np.sqrt(np.diag(Sigma))}")
        
        # Garantir que é SPD
        Sigma = _ensure_spd_matrix(Sigma)
        logger.info(f"DEBUG: Sigma final diagonal: {np.sqrt(np.diag(Sigma))}")
        
    else:
        logger.warning("Histórico insuficiente. Usando matriz diagonal simples.")
        # Fallback: matriz diagonal simples
        vol_default = 0.2  # 20% de vol anual
        daily_vol = vol_default / np.sqrt(252)
        Sigma = np.eye(M) * (daily_vol ** 2)
    
    return Sigma


def _derive_limits_and_thresholds(
    cfg: dict,
    Sigma: np.ndarray,
    cost: Union[np.ndarray, None],
    M: int
) -> Tuple[np.ndarray, dict]:
    """
    Deriva limites de posição e limiares de entrada/saída.
    
    Parâmetros
    ----------
    cfg : dict
    Sigma : np.ndarray [M, M]
    cost : np.ndarray [M] ou None
    M : int
    
    Retorna
    -------
    limits : np.ndarray [M]
    thresh : dict
    """
    limits_cfg = cfg.get('limits', {})
    thresh_cfg = cfg.get('thresh', {})
    
    # Limites baseados em alavancagem e cap por tenor
    leverage = limits_cfg.get('leverage', 3.0)
    per_tenor_cap = limits_cfg.get('per_tenor_cap', 0.3)
    
    # Limite = min(leverage/M, per_tenor_cap) para cada tenor
    base_limit = leverage / M
    limits = np.ones(M) * min(base_limit, per_tenor_cap)
    
    # Limiares
    z_in = thresh_cfg.get('z_in', 1.5)
    z_out = thresh_cfg.get('z_out', 0.5)
    
    thresh = {
        'z_in': z_in,
        'z_out': z_out
    }
    
    return limits, thresh


def _build_frictions(
    cost: Union[np.ndarray, None],
    cfg: dict,
    M: int
) -> dict:
    """
    Constrói dicionário de fricções/custos.
    
    Parâmetros
    ----------
    cost : np.ndarray [M] ou None
    cfg : dict
    M : int
    
    Retorna
    -------
    dict
    """
    costs_cfg = cfg.get('costs', {})
    
    if cost is not None:
        tick_value = np.array(cost)
        fee = np.array(cost) * 0.1  # 10% do tick como fee
    else:
        # Usar defaults
        default_tick = costs_cfg.get('default_tick_value', 10.0)
        default_fee = costs_cfg.get('default_fee', 2.0)
        tick_value = np.ones(M) * default_tick
        fee = np.ones(M) * default_fee
    
    frictions = {
        'tick_value': tick_value,
        'fee': fee,
        'slippage': costs_cfg.get('slippage', 0.0001)  # 1bp
    }
    
    return frictions


# FUNÇÃO REMOVIDA - NÃO USADA NA ESTRATÉGIA MOMENTUM:
# def _zscore(): Z-scores não são necessários para momentum puro


def _select_topK_by_abs_mispricing(mispricing: np.ndarray, topK: int) -> np.ndarray:
    """
    Seleciona os K tenores com maiores |mispricing| (para estratégia momentum).
    
    Parâmetros
    ----------
    mispricing : np.ndarray [M]
    topK : int
    
    Retorna
    -------
    np.ndarray [M] bool
        Máscara de tenores selecionados.
    """
    abs_mispricing = np.abs(mispricing)
    if topK >= len(abs_mispricing):
        return np.ones(len(abs_mispricing), dtype=bool)
    
    threshold = np.partition(abs_mispricing, -topK)[-topK]
    mask = abs_mispricing >= threshold
    return mask


def _size_positions_vol_target(
    signals: np.ndarray,
    Sigma: np.ndarray,
    cfg: dict,
    M: int
) -> np.ndarray:
    """
    Dimensiona posições para atingir vol-target.
    
    Método:
    1. Aloca peso igual a cada sinal ativo
    2. Normaliza pela soma dos pesos absolutos
    3. Escala total para atingir volatilidade-alvo de portfolio
    
    Parâmetros
    ----------
    signals : np.ndarray [M]
    Sigma : np.ndarray [M, M]
    cfg : dict
        {'sizing': {'vol_target': float}}
    M : int
    
    Retorna
    -------
    np.ndarray [M]
    """
    sizing_cfg = cfg.get('sizing', {})
    vol_target = sizing_cfg.get('vol_target', 0.10)  # 10% vol anual
    
    # Pesos iniciais baseados em sinais
    w = signals.copy().astype(float)
    
    # Se não há sinais ativos, retornar zero
    sum_abs_w = np.sum(np.abs(w))
    if sum_abs_w == 0:
        return np.zeros(M)
    
    # Normalizar para soma unitária (valor absoluto)
    w = w / sum_abs_w
    
    # Calcular volatilidade do portfolio
    portfolio_var = w @ Sigma @ w
    portfolio_vol = np.sqrt(portfolio_var) if portfolio_var > 0 else 1e-8
    
    # Escalar para atingir vol_target
    scale = vol_target / portfolio_vol
    w_scaled = w * scale
    
    return w_scaled


def _optimize_qp_momentum(
    mispricing: np.ndarray,
    Sigma: np.ndarray,
    limits: np.ndarray,
    frictions: dict,
    w_prev: np.ndarray,
    cfg: dict
) -> np.ndarray:
    """
    Otimiza posições usando Quadratic Programming para estratégia momentum.
    
    Problema de otimização (momentum puro):
    maximize: mispricing @ w - (gamma/2) * w @ Sigma @ w
              - lambda_turnover * ||w - w_prev||_1
    
    subject to: -limits <= w <= limits
    
    Parâmetros
    ----------
    mispricing : np.ndarray [M]
    Sigma : np.ndarray [M, M]
    limits : np.ndarray [M]
    frictions : dict
    w_prev : np.ndarray [M]
    cfg : dict
    
    Retorna
    -------
    np.ndarray [M]
    """
    qp_cfg = cfg.get('qp', {})
    gamma = qp_cfg.get('gamma', 3.0)  # Menor aversão ao risco para momentum
    lambda_turnover = qp_cfg.get('lambda_turnover', 0.05)  # Menos penalização de turnover
    
    M = len(mispricing)
    
    def objective(w):
        """Função objetivo para minimização (negativa do problema original)."""
        # Expected return (momentum: maior peso onde mispricing é maior)
        expected_return = mispricing @ w
        
        # Risk penalty (reduzido para momentum)
        risk_penalty = 0.5 * gamma * (w @ Sigma @ w)
        
        # Turnover penalty (reduzido para permitir mais mudanças)
        turnover_penalty = lambda_turnover * np.sum(np.abs(w - w_prev))
        
        # Minimizar o negativo (= maximizar original)
        return -(expected_return - risk_penalty - turnover_penalty)
    
    try:
        # Bounds: -limits <= w <= limits
        bounds = [(-limits[i], limits[i]) for i in range(M)]
        
        # Chute inicial baseado em momentum
        x0 = np.sign(mispricing) * 0.1  # Pequena posição na direção do momentum
        
        # Otimizar
        result = minimize(
            objective,
            x0=x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'disp': False}
        )
        
        if result.success:
            return result.x
        else:
            logger.warning(f"QP momentum não convergiu: {result.message}. Usando vol_target.")
            # Fallback para vol_target simples
            signals = np.sign(mispricing)
            return _size_positions_vol_target(signals, Sigma, cfg, M)
            
    except Exception as e:
        logger.warning(f"Erro no QP momentum: {e}. Usando vol_target.")
        signals = np.sign(mispricing)
        return _size_positions_vol_target(signals, Sigma, cfg, M)


def _apply_limits(target_w: np.ndarray, limits: np.ndarray) -> np.ndarray:
    """
    Aplica limites de posição element-wise.
    
    Parâmetros
    ----------
    target_w : np.ndarray [M]
    limits : np.ndarray [M]
    
    Retorna
    -------
    np.ndarray [M]
    """
    return np.clip(target_w, -limits, limits)


def _build_orders(
    target_w: np.ndarray,
    w_prev: np.ndarray,
    frictions: dict
) -> List[Tuple[int, str, float]]:
    """
    Constrói lista de ordens baseada na diferença target_w - w_prev.
    
    Parâmetros
    ----------
    target_w : np.ndarray [M]
    w_prev : np.ndarray [M]
    frictions : dict
    
    Retorna
    -------
    List[Tuple[int, str, float]]
        Lista de (maturity_idx, 'BUY'|'SELL', qty)
    """
    M = len(target_w)
    orders = []
    
    # Calcular diferença (trade size)
    delta_w = target_w - w_prev
    
    # Threshold mínimo para considerar uma ordem (evitar ruído)
    min_trade_size = 0.001
    
    for i in range(M):
        trade_size = delta_w[i]
        
        if np.abs(trade_size) > min_trade_size:
            if trade_size > 0:
                side = 'BUY'
                qty = trade_size
            else:
                side = 'SELL'
                qty = -trade_size  # Quantidade positiva
            
            orders.append((i, side, qty))
    
    return orders


def _ensure_spd_matrix(Sigma: np.ndarray, min_eig: float = 1e-6) -> np.ndarray:
    """
    Garante que matriz Sigma seja semi-definida positiva.
    
    Parâmetros
    ----------
    Sigma : np.ndarray
        Matriz a ser corrigida.
    min_eig : float
        Valor mínimo para autovalores.
        
    Retorna
    -------
    np.ndarray
        Matriz SPD corrigida.
    """
    # Tornar simétrica
    Sigma = (Sigma + Sigma.T) / 2
    
    # Corrigir autovalores negativos
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    eigvals = np.maximum(eigvals, min_eig)
    
    return eigvecs @ np.diag(eigvals) @ eigvecs.T