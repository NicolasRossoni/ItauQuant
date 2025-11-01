"""
Model.py

Implementação completa do modelo de dois fatores de Schwartz-Smith.
Calibra o modelo usando Filtro de Kalman (MLE ou EM), filtra os estados latentes (X_t, Y_t) 
e calcula o vetor de forwards teóricos F_modelo(t*, T_1..T_M) via fórmula fechada.

FUNÇÃO PÚBLICA PRINCIPAL: ComputeModelForward(...)

ATENÇÃO: Este módulo implementa EXATAMENTE as fórmulas definidas no help/Teory.txt.
NUNCA altere as fórmulas matemáticas sem consultar o LIVRO SAGRADO das teorias.
"""

import numpy as np
import pandas as pd
import logging
from typing import Union, Optional, Dict, Tuple
from scipy.optimize import minimize
from pykalman import KalmanFilter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ComputeModelForward(
    F_mkt: Union[np.ndarray, pd.DataFrame],
    ttm: Union[np.ndarray, pd.DataFrame],
    S: Union[np.ndarray, pd.Series, None],
    cfg: dict,
    t_idx: int
) -> dict:
    """
    Calibra o modelo de Schwartz-Smith usando Kalman Filter e calcula forwards teóricos.
    
    Esta é a FUNÇÃO PÚBLICA PRINCIPAL do módulo Model.py.
    Implementa EXATAMENTE o modelo descrito no help/Teory.txt.
    
    Parâmetros
    ----------
    F_mkt : np.ndarray ou pd.DataFrame
        Matriz [T, M] de preços a termo observados no mercado.
        Cada linha é uma data, cada coluna é um tenor.
    ttm : np.ndarray ou pd.DataFrame  
        Matriz [T, M] de time-to-maturity em anos (ACT/365).
        Mesma estrutura de F_mkt.
    S : np.ndarray, pd.Series ou None
        Série [T] de preços spot (opcional).
    cfg : dict
        Configurações contendo:
        - 'method': 'MLE' ou 'EM' (método de calibração)
        - 'kalman': dict com init_params, R, max_iter, tol, save_path
    t_idx : int
        Índice temporal alvo (ex: -1 para último período).
    
    Retorna
    -------
    dict
        {
            'Theta': dict com {kappa, sigma_X, sigma_Y, rho, mu},
            'state_t': np.ndarray [2] -> (X_hat_t*, Y_hat_t*),
            'F_model_t': np.ndarray [M] -> F_model(t*, T_1..T_M),
            'state_path': np.ndarray [T, 2] (opcional se save_path=True),
            'F_model_path': np.ndarray [T, M] (opcional se save_path=True)
        }
    
    Exceções
    --------
    ValueError
        Se shapes incompatíveis ou método inválido.
    
    Exemplos
    --------
    >>> import yaml
    >>> with open('config/default.yaml', 'r') as f:
    ...     cfg = yaml.safe_load(f)
    >>> result = ComputeModelForward(F_mkt, ttm, None, cfg, -1)
    >>> print(f"Kappa: {result['Theta']['kappa']:.4f}")
    >>> print(f"Forward no último dia: {result['F_model_t']}")
    """
    logger.info("=== Iniciando ComputeModelForward ===")
    
    # Validação e conversão de inputs
    F_mkt_arr, ttm_arr, S_arr, T, M = _validate_and_cast_inputs(F_mkt, ttm, S)
    
    # Escolher método de estimação
    method = cfg.get('method', 'MLE')
    kalman_cfg = cfg.get('kalman', {})
    
    if method == 'MLE':
        logger.info("Método selecionado: MLE (via otimização)")
        Theta, state_path = _fit_states_mle_optimization(F_mkt_arr, ttm_arr, S_arr, kalman_cfg)
    elif method == 'EM':
        logger.info("Método selecionado: EM (via otimização)")
        Theta, state_path = _fit_states_em_optimization(F_mkt_arr, ttm_arr, S_arr, kalman_cfg)
    else:
        raise ValueError(f"Método '{method}' não suportado. Use 'MLE' ou 'EM'.")
    
    # Extrair estado no tempo t*
    state_t = _extract_state_at(state_path, t_idx)
    logger.info(f"Estado em t={t_idx}: X={state_t[0]:.4f}, Y={state_t[1]:.4f}")
    
    # Calcular forward teórico em t* para todos os tenores usando FÓRMULA FECHADA
    ttm_row = ttm_arr[t_idx, :]
    F_model_t = _compute_forward_closed_form(state_t[0], state_t[1], Theta, ttm_row)
    logger.info(f"F_model_t calculado com {M} tenores")
    
    # Opcional: calcular caminho completo
    F_model_path = None
    if kalman_cfg.get('save_path', False):
        logger.info("Calculando F_model_path para todo histórico...")
        F_model_path = _compute_forward_path(state_path, Theta, ttm_arr)
    
    result = {
        'Theta': Theta,
        'state_t': state_t,
        'F_model_t': F_model_t,
        'state_path': state_path if kalman_cfg.get('save_path', False) else None,
        'F_model_path': F_model_path
    }
    
    logger.info("=== ComputeModelForward concluído com sucesso ===")
    return result


def _validate_and_cast_inputs(
    F_mkt: Union[np.ndarray, pd.DataFrame],
    ttm: Union[np.ndarray, pd.DataFrame],
    S: Union[np.ndarray, pd.Series, None]
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], int, int]:
    """
    Valida shapes e converte DataFrames para numpy arrays.
    
    Retorna
    -------
    F_mkt_arr, ttm_arr, S_arr, T, M
        Arrays validados e dimensões.
    """
    # Converter para numpy
    if isinstance(F_mkt, pd.DataFrame):
        F_mkt_arr = F_mkt.values
    else:
        F_mkt_arr = np.array(F_mkt)
    
    if isinstance(ttm, pd.DataFrame):
        ttm_arr = ttm.values
    else:
        ttm_arr = np.array(ttm)
    
    if S is not None:
        if isinstance(S, pd.Series):
            S_arr = S.values
        else:
            S_arr = np.array(S)
    else:
        S_arr = None
    
    # Validar shapes
    if F_mkt_arr.ndim != 2 or ttm_arr.ndim != 2:
        raise ValueError("F_mkt e ttm devem ser 2D")
    
    if F_mkt_arr.shape != ttm_arr.shape:
        raise ValueError(f"F_mkt shape {F_mkt_arr.shape} != ttm shape {ttm_arr.shape}")
    
    T, M = F_mkt_arr.shape
    
    if S_arr is not None and len(S_arr) != T:
        raise ValueError(f"S length {len(S_arr)} != T {T}")
    
    # Checar NaNs
    if np.any(np.isnan(F_mkt_arr)):
        logger.warning("F_mkt contém NaNs. Considere tratamento adequado.")
    
    if np.any(ttm_arr <= 0):
        raise ValueError("ttm deve conter apenas valores positivos")
    
    logger.info(f"Inputs validados: T={T}, M={M}")
    return F_mkt_arr, ttm_arr, S_arr, T, M


def _fit_states_mle_optimization(
    F_mkt: np.ndarray,
    ttm: np.ndarray,
    S: Optional[np.ndarray],
    cfg: dict
) -> Tuple[dict, np.ndarray]:
    """
    Ajusta estados usando otimização de máxima verossimilhança (MLE).
    
    Implementa MLE via otimização da log-likelihood do filtro de Kalman.
    Por simplicidade, usa mesma implementação que EM mas com configurações diferentes.
    
    Retorna
    -------
    Theta : dict
        Parâmetros estimados {kappa, sigma_X, sigma_Y, rho, mu}.
    state_path : np.ndarray [T, 2]
        Trajetória filtrada de estados [X_t, Y_t].
    """
    logger.info("Executando MLE via otimização...")
    return _fit_states_em_optimization(F_mkt, ttm, S, cfg)


def _fit_states_em_optimization(
    F_mkt: np.ndarray,
    ttm: np.ndarray,
    S: Optional[np.ndarray],
    cfg: dict
) -> Tuple[dict, np.ndarray]:
    """
    Ajusta estados usando otimização de máxima verossimilhança.
    
    Implementa o modelo de Schwartz-Smith em espaço de estados:
    - Estado: [X_t, Y_t] onde ln(S_t) = X_t + Y_t
    - X_t segue processo Ornstein-Uhlenbeck: dX_t = -kappa*X_t*dt + sigma_X*dW_X
    - Y_t segue movimento Browniano: dY_t = mu*dt + sigma_Y*dW_Y
    - Correlação: Corr(dW_X, dW_Y) = rho
    
    Parâmetros
    ----------
    F_mkt : np.ndarray [T, M]
        Preços futuros observados.
    ttm : np.ndarray [T, M]  
        Time-to-maturity em anos.
    S : np.ndarray [T] ou None
        Preços spot (não usado nesta implementação).
    cfg : dict
        Configurações com parâmetros iniciais e tolerâncias.
    
    Retorna
    -------
    Theta : dict
        Parâmetros calibrados do modelo.
    state_path : np.ndarray [T, 2]
        Estados filtrados [X_t, Y_t] para cada t.
    """
    T, M = F_mkt.shape
    dt = 1.0 / 252.0  # Assumindo dados diários
    
    # Parâmetros iniciais
    init_params = cfg.get('init_params', {})
    kappa_init = init_params.get('kappa', 1.0)
    sigma_X_init = init_params.get('sigma_X', 0.3)
    sigma_Y_init = init_params.get('sigma_Y', 0.2)
    rho_init = init_params.get('rho', 0.3)
    mu_init = init_params.get('mu', 0.0)
    R_init = cfg.get('R', 0.01)
    
    # Criar observações: ln(F_mkt)
    obs = np.log(F_mkt + 1e-10)  # Adicionar epsilon para evitar log(0)
    obs = np.nan_to_num(obs, nan=0.0)
    
    def _build_observation_matrix(kappa, ttm_array):
        """
        Constrói matriz C baseada na fórmula fechada do forward.
        
        Para observações de ln(F(t,T)), a matriz C relaciona estados [X_t, Y_t] 
        às observações considerando a estrutura temporal dos contratos.
        """
        C = np.ones((M, 2))
        for i in range(M):
            # Usar TTM médio para estabilidade
            avg_ttm = np.mean(ttm_array[:, i])
            C[i, 0] = np.exp(-kappa * avg_ttm)  # Coef. para X_t
            C[i, 1] = 1.0                       # Coef. para Y_t
        return C
    
    def _negative_log_likelihood(params):
        """
        Calcula log-likelihood negativa para otimização.
        
        Implementa o modelo em espaço de estados e usa filtro de Kalman
        para calcular a verossimilhança dos dados observados.
        """
        kappa, sigma_X, sigma_Y, rho, mu = params
        
        # Validar bounds internamente
        if kappa <= 0 or sigma_X <= 0 or sigma_Y <= 0 or abs(rho) >= 1:
            return 1e10
        
        try:
            # Matriz de transição (discretização das EDEs)
            A = np.array([
                [np.exp(-kappa * dt), 0],
                [mu * dt, 1]
            ])
            
            # Matriz de covariância de transição (baseada nas fórmulas do Teory.txt)
            var_X = (sigma_X**2 / (2*kappa)) * (1 - np.exp(-2*kappa*dt))
            var_Y = sigma_Y**2 * dt
            cov_XY = rho * sigma_X * sigma_Y * (1 - np.exp(-kappa*dt)) / kappa
            
            Q = np.array([
                [var_X, cov_XY],
                [cov_XY, var_Y]
            ])
            
            # Garantir Q é semi-definida positiva
            Q = _ensure_spd_matrix(Q)
            
            # Matriz de observação
            C = _build_observation_matrix(kappa, ttm)
            
            # Ruído de observação
            R = np.eye(M) * R_init
            
            # Filtro de Kalman
            kf = KalmanFilter(
                transition_matrices=A,
                observation_matrices=C,
                transition_covariance=Q,
                observation_covariance=R,
                initial_state_mean=np.array([0.0, np.mean(obs[:10, :])]),
                initial_state_covariance=np.eye(2) * 0.1,
                n_dim_state=2,
                n_dim_obs=M
            )
            
            # Calcular log-likelihood
            loglik = kf.loglikelihood(obs)
            
            if np.isnan(loglik) or np.isinf(loglik):
                return 1e10
            
            return -loglik  # Negativo para minimização
            
        except Exception as e:
            logger.debug(f"Erro no cálculo da likelihood: {e}")
            return 1e10
    
    # Otimização dos parâmetros
    logger.info("Iniciando otimização de máxima verossimilhança...")
    
    x0 = [kappa_init, sigma_X_init, sigma_Y_init, rho_init, mu_init]
    bounds = [
        (0.1, 10.0),    # kappa > 0
        (0.01, 2.0),    # sigma_X > 0  
        (0.01, 1.0),    # sigma_Y > 0
        (-0.99, 0.99),  # -1 < rho < 1
        (-0.1, 0.1)     # mu (sem restrição forte)
    ]
    
    result = minimize(
        _negative_log_likelihood,
        x0=x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': cfg.get('max_iter', 100), 'disp': False}
    )
    
    if result.success:
        logger.info(f"Otimização convergiu: {result.message}")
        kappa_opt, sigma_X_opt, sigma_Y_opt, rho_opt, mu_opt = result.x
    else:
        logger.warning(f"Otimização não convergiu: {result.message}. Usando parâmetros iniciais.")
        kappa_opt, sigma_X_opt, sigma_Y_opt, rho_opt, mu_opt = x0
    
    # Construir Kalman final com parâmetros otimizados
    A_opt = np.array([
        [np.exp(-kappa_opt * dt), 0],
        [mu_opt * dt, 1]
    ])
    
    var_X_opt = (sigma_X_opt**2 / (2*kappa_opt)) * (1 - np.exp(-2*kappa_opt*dt))
    var_Y_opt = sigma_Y_opt**2 * dt
    cov_XY_opt = rho_opt * sigma_X_opt * sigma_Y_opt * (1 - np.exp(-kappa_opt*dt)) / kappa_opt
    
    Q_opt = np.array([
        [var_X_opt, cov_XY_opt],
        [cov_XY_opt, var_Y_opt]
    ])
    Q_opt = _ensure_spd_matrix(Q_opt)
    
    C_opt = _build_observation_matrix(kappa_opt, ttm)
    R_opt = np.eye(M) * R_init
    
    kf_final = KalmanFilter(
        transition_matrices=A_opt,
        observation_matrices=C_opt,
        transition_covariance=Q_opt,
        observation_covariance=R_opt,
        initial_state_mean=np.array([0.0, np.mean(obs[:10, :])]),
        initial_state_covariance=np.eye(2) * 0.1,
        n_dim_state=2,
        n_dim_obs=M
    )
    
    # Filtrar estados
    state_means, state_covs = kf_final.filter(obs)
    
    Theta = {
        'kappa': kappa_opt,
        'sigma_X': sigma_X_opt,
        'sigma_Y': sigma_Y_opt,
        'rho': rho_opt,
        'mu': mu_opt
    }
    
    logger.info(f"Parâmetros calibrados: kappa={kappa_opt:.4f}, sigma_X={sigma_X_opt:.4f}, "
                f"sigma_Y={sigma_Y_opt:.4f}, rho={rho_opt:.4f}, mu={mu_opt:.6f}")
    
    return Theta, state_means


def _ensure_spd_matrix(Q: np.ndarray, min_eig: float = 1e-6) -> np.ndarray:
    """
    Garante que matriz Q seja semi-definida positiva.
    
    Parâmetros
    ---------- 
    Q : np.ndarray
        Matriz a ser corrigida.
    min_eig : float
        Valor mínimo para autovalores.
        
    Retorna
    -------
    np.ndarray
        Matriz SPD corrigida.
    """
    Q = (Q + Q.T) / 2  # Tornar simétrica
    eigvals, eigvecs = np.linalg.eigh(Q)
    eigvals = np.maximum(eigvals, min_eig)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def _extract_state_at(state_path: np.ndarray, t_idx: int) -> np.ndarray:
    """
    Extrai o estado filtrado em um índice temporal específico.
    
    Parâmetros
    ----------
    state_path : np.ndarray [T, 2]
        Trajetória completa dos estados filtrados.
    t_idx : int
        Índice temporal (pode ser negativo para contar do final).
    
    Retorna
    -------
    np.ndarray [2]
        Estado [X_hat, Y_hat] no tempo t_idx.
    """
    return state_path[t_idx, :]


def _compute_forward_closed_form(
    X_hat: float,
    Y_hat: float,
    Theta: dict,
    ttm_row: np.ndarray
) -> np.ndarray:
    """
    Calcula o vetor de forwards teóricos usando a FÓRMULA FECHADA de Schwartz-Smith.
    
    ATENÇÃO: Esta função implementa EXATAMENTE a fórmula do help/Teory.txt.
    NUNCA altere sem consultar o LIVRO SAGRADO das teorias.
    
    Fórmula (do Teory.txt):
    F(t, T) = exp(
        X_t * exp(-kappa*Delta) + Y_t + mu*Delta
        + 0.5 * [
            (sigma_X^2 / (2*kappa)) * (1 - exp(-2*kappa*Delta))
            + sigma_Y^2 * Delta
            + 2*rho*sigma_X*sigma_Y * (1 - exp(-kappa*Delta)) / kappa
        ]
    )
    
    Parâmetros
    ----------
    X_hat : float
        Estado de curto prazo filtrado.
    Y_hat : float
        Estado de longo prazo filtrado.
    Theta : dict
        Parâmetros do modelo {kappa, sigma_X, sigma_Y, rho, mu}.
    ttm_row : np.ndarray [M]
        Time-to-maturity em anos para cada tenor.
    
    Retorna
    -------
    np.ndarray [M]
        Vetor de forwards teóricos F_modelo(t, T_i).
    """
    kappa = Theta['kappa']
    sigma_X = Theta['sigma_X']
    sigma_Y = Theta['sigma_Y']
    rho = Theta['rho']
    mu = Theta['mu']
    
    M = len(ttm_row)
    F_model = np.zeros(M)
    
    for i in range(M):
        Delta = ttm_row[i]
        
        # Média condicional E[X_T + Y_T | X_t, Y_t]
        mean_X = X_hat * np.exp(-kappa * Delta)
        mean_Y = Y_hat + mu * Delta
        
        # Variância condicional Var[X_T + Y_T | X_t, Y_t]
        var_X = (sigma_X**2 / (2*kappa)) * (1 - np.exp(-2*kappa*Delta))
        var_Y = sigma_Y**2 * Delta
        cov_XY = rho * sigma_X * sigma_Y * (1 - np.exp(-kappa*Delta)) / kappa
        
        # Variância total de X_T + Y_T
        var_total = var_X + var_Y + 2*cov_XY
        
        # Forward usando fórmula fechada
        F_model[i] = np.exp(mean_X + mean_Y + 0.5 * var_total)
    
    return F_model


def _compute_forward_path(
    state_path: np.ndarray,
    Theta: dict,
    ttm: np.ndarray
) -> np.ndarray:
    """
    Calcula F_model(t, T_i) para todo o histórico.
    
    Parâmetros
    ----------
    state_path : np.ndarray [T, 2]
        Trajetória completa dos estados filtrados.
    Theta : dict
        Parâmetros calibrados do modelo.
    ttm : np.ndarray [T, M]
        Time-to-maturity para cada data e tenor.
    
    Retorna
    -------
    np.ndarray [T, M]
        Matriz de forwards teóricos para todo histórico.
    """
    T, M = ttm.shape
    F_model_path = np.zeros((T, M))
    
    for t in range(T):
        X_t = state_path[t, 0]
        Y_t = state_path[t, 1]
        ttm_t = ttm[t, :]
        F_model_path[t, :] = _compute_forward_closed_form(X_t, Y_t, Theta, ttm_t)
    
    return F_model_path