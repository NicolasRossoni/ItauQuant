"""
TradingStrategy.py

Implementação completa da estratégia de trading baseada em mispricing do modelo Schwartz-Smith.
Combina preparação de inputs de trading e motor de decisão.

FUNÇÕES PÚBLICAS PRINCIPAIS:
- PrepareTradingInputs(...): prepara mispricing, matriz de risco, limites, etc.
- TradeEngine(...): gera sinais, posições-alvo e ordens executáveis

ATENÇÃO: Este módulo implementa a estratégia conforme definida no help/Teory.txt.
Respeita os mandamentos filosóficos definidos no help/Architecture.md.
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
    logger.info(f"Mispricing calculado: média={np.mean(mispricing):.4f}, "
                f"std={np.std(mispricing):.4f}, "
                f"min={np.min(mispricing):.4f}, max={np.max(mispricing):.4f}")
    
    # 2. Estimar matriz de covariância
    Sigma = _estimate_covariance(
        F_mkt_hist, F_model_hist, cfg.get('risk', {}), M
    )
    logger.info(f"Matriz Sigma estimada: shape={Sigma.shape}, "
                f"eigenvalues min={np.min(np.linalg.eigvals(Sigma)):.6f}, "
                f"max={np.max(np.linalg.eigvals(Sigma)):.6f}")
    
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
    
    Esta é uma das FUNÇÕES PÚBLICAS PRINCIPAIS do módulo TradingStrategy.py.
    Implementa a estratégia de trading baseada em mispricing conforme Teory.txt.
    
    Parâmetros
    ----------
    mispricing : np.ndarray [M]
        DeltaF = F_model - F_mkt por tenor.
    Sigma : np.ndarray [M, M]
        Matriz de covariância de retornos dos futuros.
    limits : np.ndarray [M]
        Limites de posição por tenor (máximo valor absoluto).
    thresh : np.ndarray [M] ou dict
        Limiares de entrada/saída. Se dict: {'z_in': valor, 'z_out': valor}.
    frictions : dict
        Custos: {'tick_value': [M], 'fee': [M], 'slippage': float}.
    method : str
        Método de dimensionamento: "vol_target" ou "qp" (Quadratic Programming).
    topK : int, opcional
        Número máximo de tenores a operar (maiores |z-score|).
    w_prev : np.ndarray [M], opcional
        Posições vigentes (para calcular turnover no QP).
    cfg : dict, opcional
        Configurações adicionais (vol_target, parâmetros QP).
    
    Retorna
    -------
    dict
        {
            'signals': np.ndarray [M] em {-1, 0, +1} (vender, nada, comprar),
            'target_w': np.ndarray [M] posições-alvo normalizadas,
            'orders': list de (maturity_idx, 'BUY'|'SELL', qty),
            'z_scores': np.ndarray [M] (para análise)
        }
    
    Exemplos
    --------
    >>> result = TradeEngine(mispricing, Sigma, limits, thresh, frictions)
    >>> print(f"Sinais: {result['signals']}")
    >>> print(f"Ordens: {len(result['orders'])} operações")
    >>> for order in result['orders']:
    ...     print(f"Tenor {order[0]}: {order[1]} {order[2]} contratos")
    """
    logger.info("=== Iniciando TradeEngine ===")
    
    M = len(mispricing)
    
    if w_prev is None:
        w_prev = np.zeros(M)
    
    if cfg is None:
        cfg = {}
    
    # Validar inputs
    if Sigma.shape != (M, M):
        raise ValueError(f"Sigma shape {Sigma.shape} != ({M}, {M})")
    if len(limits) != M:
        raise ValueError(f"limits length {len(limits)} != {M}")
    
    # 1. Calcular z-scores
    z_scores = _zscore(mispricing, Sigma)
    logger.info(f"Z-scores: min={np.min(z_scores):.2f}, max={np.max(z_scores):.2f}, "
                f"mean={np.mean(z_scores):.2f}, std={np.std(z_scores):.2f}")
    
    # 2. Seleção top-K (opcional)
    active_mask = np.ones(M, dtype=bool)
    if topK is not None and topK < M:
        active_mask = _select_topK_by_abs_z(z_scores, topK)
        logger.info(f"Top-K={topK} selecionados: {np.sum(active_mask)} tenores ativos")
    
    # 3. Aplicar histerese para gerar sinais
    signals = _apply_hysteresis(z_scores, thresh, w_prev, active_mask)
    logger.info(f"Sinais: BUY={np.sum(signals==1)}, SELL={np.sum(signals==-1)}, "
                f"HOLD={np.sum(signals==0)}")
    
    # 4. Dimensionar posições
    if method == "vol_target":
        target_w = _size_positions_vol_target(signals, Sigma, cfg, M)
        logger.info("Dimensionamento: vol_target")
    elif method == "qp":
        target_w = _optimize_qp(mispricing, Sigma, limits, frictions, w_prev, cfg)
        logger.info("Dimensionamento: QP mean-variance")
    else:
        raise ValueError(f"Método '{method}' não suportado. Use 'vol_target' ou 'qp'.")
    
    # 5. Aplicar limites
    target_w = _apply_limits(target_w, limits)
    logger.info(f"Target weights (após limites): soma={np.sum(np.abs(target_w)):.4f}, "
                f"max={np.max(np.abs(target_w)):.4f}")
    
    # 6. Construir ordens
    orders = _build_orders(target_w, w_prev, frictions)
    logger.info(f"Ordens geradas: {len(orders)} operações")
    
    result = {
        'signals': signals,
        'target_w': target_w,
        'orders': orders,
        'z_scores': z_scores  # Para análise posterior
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
        logger.info(f"Estimando covariância usando histórico ({source})")
        
        # Usar janela lookback
        data_window = F_mkt_hist[-lookback:, :]
        
        if source == 'returns':
            # Calcular retornos log
            returns = np.diff(np.log(data_window + 1e-10), axis=0)
        elif source == 'residuals' and F_model_hist is not None:
            # Usar resíduos F_mkt - F_model
            model_window = F_model_hist[-lookback:, :]
            residuals = data_window - model_window
            returns = np.diff(residuals, axis=0)
        else:
            # Fallback para retornos
            returns = np.diff(np.log(data_window + 1e-10), axis=0)
        
        # Remover NaNs
        returns = np.nan_to_num(returns, nan=0.0)
        
        # Estimar covariância
        if use_shrinkage and returns.shape[0] > M:
            try:
                lw = LedoitWolf()
                Sigma = lw.fit(returns).covariance_
            except Exception as e:
                logger.warning(f"Shrinkage falhou: {e}. Usando covariância empírica.")
                Sigma = np.cov(returns, rowvar=False)
        else:
            Sigma = np.cov(returns, rowvar=False)
        
        # Garantir que é SPD
        Sigma = _ensure_spd_matrix(Sigma)
        
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


def _zscore(mispricing: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """
    Calcula z-score por tenor usando desvio-padrão marginal.
    
    z_i = mispricing_i / sqrt(Sigma_ii)
    
    Parâmetros
    ----------
    mispricing : np.ndarray [M]
    Sigma : np.ndarray [M, M]
    
    Retorna
    -------
    np.ndarray [M]
    """
    std_diag = np.sqrt(np.diag(Sigma))
    std_diag = np.where(std_diag < 1e-8, 1e-8, std_diag)  # Evitar divisão por zero
    z = mispricing / std_diag
    return z


def _select_topK_by_abs_z(z_scores: np.ndarray, topK: int) -> np.ndarray:
    """
    Seleciona os K tenores com maiores |z-score|.
    
    Parâmetros
    ----------
    z_scores : np.ndarray [M]
    topK : int
    
    Retorna
    -------
    np.ndarray [M] bool
        Máscara de tenores selecionados.
    """
    abs_z = np.abs(z_scores)
    if topK >= len(abs_z):
        return np.ones(len(abs_z), dtype=bool)
    
    threshold = np.partition(abs_z, -topK)[-topK]
    mask = abs_z >= threshold
    return mask


def _apply_hysteresis(
    z_scores: np.ndarray,
    thresh: Union[dict, np.ndarray],
    w_prev: np.ndarray,
    active_mask: np.ndarray
) -> np.ndarray:
    """
    Aplica histerese para gerar sinais discretos {-1, 0, +1}.
    
    Regras de histerese (para evitar whipsaw):
    - Se |z| > z_in: entrar na direção do mispricing (sinal = sign(z))
    - Se |z| < z_out E há posição existente: sair (sinal = 0)
    - Caso contrário: manter posição anterior
    
    Parâmetros
    ----------
    z_scores : np.ndarray [M]
    thresh : dict com 'z_in' e 'z_out' ou array
    w_prev : np.ndarray [M]
    active_mask : np.ndarray [M] bool
    
    Retorna
    -------
    np.ndarray [M]
        Sinais em {-1, 0, +1}
    """
    M = len(z_scores)
    signals = np.zeros(M)
    
    # Extrair limiares
    if isinstance(thresh, dict):
        z_in = thresh.get('z_in', 1.5)
        z_out = thresh.get('z_out', 0.5)
    else:
        z_in = 1.5
        z_out = 0.5
    
    # Converter para arrays se forem escalares
    if isinstance(z_in, (int, float)):
        z_in = np.ones(M) * z_in
    else:
        z_in = np.array(z_in) if not isinstance(z_in, np.ndarray) else z_in
        
    if isinstance(z_out, (int, float)):
        z_out = np.ones(M) * z_out
    else:
        z_out = np.array(z_out) if not isinstance(z_out, np.ndarray) else z_out
    
    for i in range(M):
        if not active_mask[i]:
            signals[i] = 0
            continue
        
        abs_z = np.abs(z_scores[i])
        
        # Entrada: |z| > z_in
        if abs_z > z_in[i]:
            signals[i] = np.sign(z_scores[i])
        # Saída: |z| < z_out E há posição existente
        elif abs_z < z_out[i] and np.abs(w_prev[i]) > 1e-6:
            signals[i] = 0  # Fechar posição
        # Manter posição anterior
        else:
            signals[i] = np.sign(w_prev[i]) if np.abs(w_prev[i]) > 1e-6 else 0
    
    return signals


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


def _optimize_qp(
    mispricing: np.ndarray,
    Sigma: np.ndarray,
    limits: np.ndarray,
    frictions: dict,
    w_prev: np.ndarray,
    cfg: dict
) -> np.ndarray:
    """
    Otimiza posições usando Quadratic Programming (mean-variance).
    
    Problema de otimização:
    maximize: mispricing @ w - (gamma/2) * w @ Sigma @ w
              - lambda_l1 * ||w||_1
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
        {'sizing': {'qp': {'gamma': float, 'lambda_l1': float, 'lambda_turnover': float}}}
    
    Retorna
    -------
    np.ndarray [M]
    """
    sizing_cfg = cfg.get('sizing', {})
    qp_cfg = sizing_cfg.get('qp', {})
    
    gamma = qp_cfg.get('gamma', 5.0)
    lambda_l1 = qp_cfg.get('lambda_l1', 0.0)
    lambda_turnover = qp_cfg.get('lambda_turnover', 0.1)
    
    M = len(mispricing)
    
    def objective(w):
        """Função objetivo para minimização (negativa do problema original)."""
        # Expected return
        expected_return = mispricing @ w
        
        # Risk penalty
        risk_penalty = 0.5 * gamma * (w @ Sigma @ w)
        
        # L1 penalty
        l1_penalty = lambda_l1 * np.sum(np.abs(w))
        
        # Turnover penalty
        turnover_penalty = lambda_turnover * np.sum(np.abs(w - w_prev))
        
        # Minimizar o negativo (= maximizar original)
        return -(expected_return - risk_penalty - l1_penalty - turnover_penalty)
    
    # Bounds: -limits <= w <= limits
    bounds = [(-limits[i], limits[i]) for i in range(M)]
    
    # Chute inicial: posição anterior
    x0 = w_prev.copy()
    
    # Garantir que x0 está dentro dos bounds
    x0 = np.clip(x0, -limits, limits)
    
    try:
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
            logger.warning(f"QP não convergiu: {result.message}. Usando vol_target.")
            # Fallback para vol_target simples
            signals = np.sign(mispricing)
            return _size_positions_vol_target(signals, Sigma, cfg, M)
            
    except Exception as e:
        logger.warning(f"Erro no QP: {e}. Usando vol_target.")
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