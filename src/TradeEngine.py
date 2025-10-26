"""
TradeEngine.py

Motor de decisão de trading que transforma mispricing + risco + limites em ordens executáveis.

Implementa:
- Cálculo de z-scores
- Histerese (limiares de entrada/saída)
- Seleção top-K
- Dimensionamento de posições (vol-target ou QP mean-variance)
- Aplicação de limites
- Geração de ordens

Função pública: TradeEngine(...)
"""

import numpy as np
import logging
from typing import Union, Optional, Dict, List, Tuple
from scipy.optimize import minimize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    Gera sinais, posições-alvo e ordens a partir de mispricing e risco.
    
    Parâmetros
    ----------
    mispricing : np.ndarray [M]
        DeltaF = F_model - F_mkt por tenor.
    Sigma : np.ndarray [M, M]
        Matriz de covariância de retornos.
    limits : np.ndarray [M]
        Limites de posição por tenor.
    thresh : np.ndarray [M] ou dict
        Limiares {'z_in': [M], 'z_out': [M]} para histerese.
    frictions : dict
        {'tick_value': [M], 'fee': [M], 'slippage': float}
    method : str
        "vol_target" ou "qp" (Quadratic Programming).
    topK : int, opcional
        Número máximo de tenores a operar (maiores |z-score|).
    w_prev : np.ndarray [M], opcional
        Posições vigentes (para calcular turnover).
    cfg : dict, opcional
        Configurações adicionais (vol_target, qp params).
    
    Retorna
    -------
    dict
        {
            'signals': np.ndarray [M] em {-1, 0, +1},
            'target_w': np.ndarray [M] posições-alvo normalizadas,
            'orders': list de (maturity_idx, 'BUY'|'SELL', qty)
        }
    
    Exemplos
    --------
    >>> result = TradeEngine(mispricing, Sigma, limits, thresh, frictions)
    >>> print(result['signals'])
    >>> print(result['orders'])
    """
    logger.info("=== Iniciando TradeEngine ===")
    
    M = len(mispricing)
    
    if w_prev is None:
        w_prev = np.zeros(M)
    
    if cfg is None:
        cfg = {}
    
    # 1. Calcular z-scores
    z_scores = _zscore(mispricing, Sigma)
    logger.info(f"Z-scores: min={np.min(z_scores):.2f}, max={np.max(z_scores):.2f}, "
                f"mean={np.mean(z_scores):.2f}")
    
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
        'z_scores': z_scores  # Adicional para debug/gráficos
    }
    
    logger.info("=== TradeEngine concluído ===")
    return result


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
    
    Regras:
    - Se |z| > z_in: entrar (sinal = sign(z))
    - Se |z| < z_out E posição existente: sair (sinal = 0)
    - Caso contrário: manter posição anterior
    
    Parâmetros
    ----------
    z_scores : np.ndarray [M]
    thresh : dict com 'z_in' e 'z_out' [M] ou escalar
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
        z_in = thresh.get('z_in', np.ones(M) * 1.5)
        z_out = thresh.get('z_out', np.ones(M) * 0.5)
    else:
        z_in = np.ones(M) * 1.5
        z_out = np.ones(M) * 0.5
    
    # Converter para arrays se forem escalares
    if isinstance(z_in, (int, float)):
        z_in = np.ones(M) * z_in
    if isinstance(z_out, (int, float)):
        z_out = np.ones(M) * z_out
    
    for i in range(M):
        if not active_mask[i]:
            signals[i] = 0
            continue
        
        abs_z = np.abs(z_scores[i])
        
        # Entrada: |z| > z_in
        if abs_z > z_in[i]:
            signals[i] = np.sign(z_scores[i])
        # Saída: |z| < z_out E há posição existente
        elif abs_z < z_out[i] and w_prev[i] != 0:
            signals[i] = 0  # Fechar posição
        # Manter posição anterior
        else:
            signals[i] = np.sign(w_prev[i]) if w_prev[i] != 0 else 0
    
    return signals


def _size_positions_vol_target(
    signals: np.ndarray,
    Sigma: np.ndarray,
    cfg: dict,
    M: int
) -> np.ndarray:
    """
    Dimensiona posições para atingir vol-target.
    
    Método simplificado:
    - Aloca peso igual a cada sinal ativo
    - Escala total para atingir volatilidade-alvo de portfolio
    
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
    vol_target = sizing_cfg.get('vol_target', 0.10)
    
    # Pesos iniciais baseados em sinais
    w = signals.copy().astype(float)
    
    # Normalizar para soma unitária (valor absoluto)
    sum_abs_w = np.sum(np.abs(w))
    if sum_abs_w > 0:
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
    
    Objetivo:
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
        {'sizing': {'qp': {'gamma', 'lambda_l1', 'lambda_turnover'}}}
    
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
    
    # Função objetivo (negativo para minimização)
    def objective(w):
        expected_return = mispricing @ w
        risk_penalty = (gamma / 2) * (w @ Sigma @ w)
        l1_penalty = lambda_l1 * np.sum(np.abs(w))
        turnover_penalty = lambda_turnover * np.sum(np.abs(w - w_prev))
        
        return -expected_return + risk_penalty + l1_penalty + turnover_penalty
    
    # Bounds: -limits <= w <= limits
    bounds = [(-limits[i], limits[i]) for i in range(M)]
    
    # Otimização
    result = minimize(
        objective,
        x0=w_prev,
        method='SLSQP',
        bounds=bounds,
        options={'maxiter': 200, 'ftol': 1e-6}
    )
    
    if result.success:
        logger.info(f"QP otimização convergiu: obj={result.fun:.6f}")
        return result.x
    else:
        logger.warning(f"QP otimização falhou: {result.message}. Usando w_prev.")
        return w_prev


def _apply_limits(target_w: np.ndarray, limits: np.ndarray) -> np.ndarray:
    """
    Aplica limites de posição por tenor.
    
    Parâmetros
    ----------
    target_w : np.ndarray [M]
    limits : np.ndarray [M]
    
    Retorna
    -------
    np.ndarray [M]
    """
    w_limited = np.clip(target_w, -limits, limits)
    return w_limited


def _build_orders(
    target_w: np.ndarray,
    w_prev: np.ndarray,
    frictions: dict
) -> List[Tuple[int, str, float]]:
    """
    Constrói lista de ordens executáveis.
    
    Ordem = (maturity_idx, side, qty)
    
    Parâmetros
    ----------
    target_w : np.ndarray [M]
    w_prev : np.ndarray [M]
    frictions : dict
    
    Retorna
    -------
    list de (int, str, float)
        [(maturity_idx, 'BUY'|'SELL', qty), ...]
    """
    M = len(target_w)
    orders = []
    
    delta_w = target_w - w_prev
    
    for i in range(M):
        qty = delta_w[i]
        
        # Arredondar para evitar trades minúsculos
        if np.abs(qty) < 1e-4:
            continue
        
        if qty > 0:
            side = 'BUY'
        else:
            side = 'SELL'
            qty = -qty
        
        orders.append((i, side, qty))
    
    return orders
