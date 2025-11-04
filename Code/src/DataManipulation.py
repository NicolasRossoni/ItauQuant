"""
DataManipulation.py

Módulo de manipulação pesada de dados para abstrair a lógica complexa dos arquivos principais.
Contém todas as funções auxiliares de formatação, limpeza, transformações e validações de dados.

OBJETIVO: Deixar os códigos principais (download.py, backtest.py, analysis.py) como fluxogramas
legíveis, enquanto toda a sintaxe pesada fica abstraída neste módulo.

FUNÇÕES PÚBLICAS PRINCIPAIS:
- load_data_from_raw(...): carrega dados estruturados da pasta raw
- save_data_to_processed(...): salva resultados estruturados na pasta processed  
- validate_data_structure(...): valida estrutura de dados
- format_for_model(...): formata dados para input do modelo
- format_for_analysis(...): formata dados para análise
- create_time_series(...): cria séries temporais padronizadas

ATENÇÃO: Este módulo implementa manipulações seguindo princípios de modularidade e clareza.
Mantém compatibilidade com estrutura padronizada de dados definida no projeto.
"""

import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, date
import yaml
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data_from_raw(
    dataset_id: str,
    raw_path: str = "data/raw"
) -> Dict[str, Union[pd.DataFrame, np.ndarray]]:
    """
    Carrega dados estruturados da pasta raw seguindo padrão definido.
    
    Parâmetros
    ----------
    dataset_id : str
        ID do dataset (nome da pasta em raw/).
    raw_path : str
        Caminho base para dados raw.
    
    Retorna
    -------
    dict
        {
            'F_mkt': pd.DataFrame [T x M] com preços futuros,
            'ttm': pd.DataFrame [T x M] com time-to-maturity,
            'S': pd.Series [T] com preços spot (opcional),
            'costs': pd.Series [M] com custos (opcional),
            'dates': pd.DatetimeIndex [T],
            'tenors': List[str] [M]
        }
    
    Exemplos
    --------
    >>> data = load_data_from_raw("WTI_2024")
    >>> print(f"Dados carregados: {data['F_mkt'].shape}")
    >>> print(f"Período: {data['dates'][0]} a {data['dates'][-1]}")
    """
    logger.info(f"=== Carregando dados raw: {dataset_id} ===")
    
    dataset_path = os.path.join(raw_path, dataset_id)
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset não encontrado: {dataset_path}")
    
    # Carregar arquivos obrigatórios
    F_mkt_path = os.path.join(dataset_path, "F_mkt.csv")
    ttm_path = os.path.join(dataset_path, "ttm.csv")
    
    if not os.path.exists(F_mkt_path):
        raise FileNotFoundError(f"F_mkt.csv não encontrado em {dataset_path}")
    if not os.path.exists(ttm_path):
        raise FileNotFoundError(f"ttm.csv não encontrado em {dataset_path}")
    
    # Carregar F_mkt e ttm
    F_mkt = pd.read_csv(F_mkt_path, index_col=0, parse_dates=True)
    ttm = pd.read_csv(ttm_path, index_col=0, parse_dates=True)
    
    logger.info(f"F_mkt carregado: shape {F_mkt.shape}")
    logger.info(f"ttm carregado: shape {ttm.shape}")
    
    # Validar consistência
    if F_mkt.shape != ttm.shape:
        raise ValueError(f"Shapes incompatíveis: F_mkt {F_mkt.shape} != ttm {ttm.shape}")
    
    if not F_mkt.index.equals(ttm.index):
        logger.warning("Índices de F_mkt e ttm não são idênticos. Realizando inner join.")
        common_dates = F_mkt.index.intersection(ttm.index)
        F_mkt = F_mkt.loc[common_dates]
        ttm = ttm.loc[common_dates]
    
    # Carregar arquivos opcionais
    S = None
    S_path = os.path.join(dataset_path, "S.csv")
    if os.path.exists(S_path):
        S = pd.read_csv(S_path, index_col=0, parse_dates=True, squeeze=True)
        logger.info(f"Spot carregado: shape {S.shape}")
        
        # Garantir mesmo índice
        if not S.index.equals(F_mkt.index):
            logger.warning("Índice de S diferente de F_mkt. Realizando reindex.")
            S = S.reindex(F_mkt.index)
    
    costs = None
    costs_path = os.path.join(dataset_path, "costs.csv")
    if os.path.exists(costs_path):
        costs = pd.read_csv(costs_path, index_col=0, squeeze=True)
        logger.info(f"Custos carregados: shape {costs.shape}")
        
        # Garantir mesmas colunas
        if not costs.index.equals(pd.Index(F_mkt.columns)):
            logger.warning("Índice de costs diferente das colunas de F_mkt. Realizando reindex.")
            costs = costs.reindex(F_mkt.columns)
    
    result = {
        'F_mkt': F_mkt,
        'ttm': ttm,
        'S': S,
        'costs': costs,
        'dates': F_mkt.index,
        'tenors': list(F_mkt.columns)
    }
    
    logger.info(f"Dados carregados com sucesso: T={len(result['dates'])}, M={len(result['tenors'])}")
    return result


def save_data_to_processed(
    data: Dict,
    dataset_id: str,
    processed_path: str = "data/processed"
) -> str:
    """
    Salva dados processados seguindo estrutura padrão do projeto.
    
    Parâmetros
    ----------
    data : dict
        Dados a serem salvos (resultados do backtest).
    dataset_id : str
        ID do dataset.
    processed_path : str
        Caminho base para dados processados.
    
    Retorna
    -------
    str
        Caminho do diretório criado.
    
    Exemplos
    --------
    >>> data = {'portfolio_performance': df, 'trades_log': trades_df}
    >>> path = save_data_to_processed(data, "WTI_2024")
    >>> print(f"Dados salvos em: {path}")
    """
    logger.info(f"=== Salvando dados processados: {dataset_id} ===")
    
    # Criar diretório com timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = os.path.join(processed_path, f"{dataset_id}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Criar subdiretórios
    daily_dir = os.path.join(output_dir, "daily_results")
    os.makedirs(daily_dir, exist_ok=True)
    
    # Salvar dados principais
    for key, value in data.items():
        if key == 'daily_results':
            _save_daily_results(value, daily_dir)
        elif isinstance(value, pd.DataFrame):
            filepath = os.path.join(output_dir, f"{key}.csv")
            value.to_csv(filepath)
            logger.info(f"Salvo: {key}.csv")
        elif isinstance(value, dict):
            filepath = os.path.join(output_dir, f"{key}.pkl")
            with open(filepath, 'wb') as f:
                pickle.dump(value, f)
            logger.info(f"Salvo: {key}.pkl")
        else:
            # Tentar salvar como pickle
            filepath = os.path.join(output_dir, f"{key}.pkl")
            with open(filepath, 'wb') as f:
                pickle.dump(value, f)
            logger.info(f"Salvo: {key}.pkl")
    
    logger.info(f"Dados salvos em: {output_dir}")
    return output_dir


def validate_data_structure(
    F_mkt: Union[pd.DataFrame, np.ndarray],
    ttm: Union[pd.DataFrame, np.ndarray],
    S: Optional[Union[pd.Series, np.ndarray]] = None
) -> Tuple[bool, List[str]]:
    """
    Valida estrutura de dados conforme padrões definidos.
    
    Parâmetros
    ----------
    F_mkt : pd.DataFrame ou np.ndarray [T x M]
    ttm : pd.DataFrame ou np.ndarray [T x M]  
    S : pd.Series ou np.ndarray [T], opcional
    
    Retorna
    -------
    tuple
        (is_valid: bool, errors: List[str])
    
    Exemplos
    --------
    >>> is_valid, errors = validate_data_structure(F_mkt, ttm, S)
    >>> if not is_valid:
    ...     for error in errors:
    ...         print(f"Erro: {error}")
    """
    errors = []
    
    # Validar F_mkt
    if isinstance(F_mkt, pd.DataFrame):
        if F_mkt.empty:
            errors.append("F_mkt está vazio")
        if F_mkt.isnull().any().any():
            errors.append("F_mkt contém valores NaN")
        if (F_mkt <= 0).any().any():
            errors.append("F_mkt contém valores não-positivos")
    elif isinstance(F_mkt, np.ndarray):
        if F_mkt.size == 0:
            errors.append("F_mkt está vazio")
        if F_mkt.ndim != 2:
            errors.append("F_mkt deve ser 2D")
        if np.any(np.isnan(F_mkt)):
            errors.append("F_mkt contém valores NaN")
        if np.any(F_mkt <= 0):
            errors.append("F_mkt contém valores não-positivos")
    else:
        errors.append("F_mkt deve ser DataFrame ou ndarray")
    
    # Validar ttm
    if isinstance(ttm, pd.DataFrame):
        if ttm.empty:
            errors.append("ttm está vazio")
        if ttm.isnull().any().any():
            errors.append("ttm contém valores NaN")
        if (ttm <= 0).any().any():
            errors.append("ttm contém valores não-positivos")
    elif isinstance(ttm, np.ndarray):
        if ttm.size == 0:
            errors.append("ttm está vazio")
        if ttm.ndim != 2:
            errors.append("ttm deve ser 2D")
        if np.any(np.isnan(ttm)):
            errors.append("ttm contém valores NaN")
        if np.any(ttm <= 0):
            errors.append("ttm contém valores não-positivos")
    else:
        errors.append("ttm deve ser DataFrame ou ndarray")
    
    # Validar consistência
    if hasattr(F_mkt, 'shape') and hasattr(ttm, 'shape'):
        if F_mkt.shape != ttm.shape:
            errors.append(f"Shapes incompatíveis: F_mkt {F_mkt.shape} != ttm {ttm.shape}")
    
    # Validar S se fornecido
    if S is not None:
        if isinstance(S, pd.Series):
            if S.empty:
                errors.append("S está vazio")
            if S.isnull().any():
                errors.append("S contém valores NaN")
            if (S <= 0).any():
                errors.append("S contém valores não-positivos")
        elif isinstance(S, np.ndarray):
            if S.size == 0:
                errors.append("S está vazio")
            if S.ndim != 1:
                errors.append("S deve ser 1D")
            if np.any(np.isnan(S)):
                errors.append("S contém valores NaN")
            if np.any(S <= 0):
                errors.append("S contém valores não-positivos")
        
        # Validar tamanho de S
        if hasattr(F_mkt, 'shape') and hasattr(S, 'shape'):
            if len(S) != F_mkt.shape[0]:
                errors.append(f"S length {len(S)} != F_mkt rows {F_mkt.shape[0]}")
    
    is_valid = len(errors) == 0
    return is_valid, errors


def format_for_model(
    data: Dict[str, Union[pd.DataFrame, pd.Series]]
) -> Dict[str, np.ndarray]:
    """
    Formata dados para input do modelo (converte para numpy arrays).
    
    Parâmetros
    ----------
    data : dict
        Dados carregados do load_data_from_raw.
    
    Retorna
    -------
    dict
        {
            'F_mkt': np.ndarray [T x M],
            'ttm': np.ndarray [T x M],
            'S': np.ndarray [T] ou None
        }
    
    Exemplos
    --------
    >>> raw_data = load_data_from_raw("WTI_2024")
    >>> model_data = format_for_model(raw_data)
    >>> print(f"F_mkt: {model_data['F_mkt'].shape}, dtype: {model_data['F_mkt'].dtype}")
    """
    logger.info("Formatando dados para o modelo...")
    
    # Validar entrada
    is_valid, errors = validate_data_structure(data['F_mkt'], data['ttm'], data.get('S'))
    if not is_valid:
        raise ValueError(f"Dados inválidos: {'; '.join(errors)}")
    
    # Converter para numpy
    F_mkt_arr = data['F_mkt'].values if isinstance(data['F_mkt'], pd.DataFrame) else data['F_mkt']
    ttm_arr = data['ttm'].values if isinstance(data['ttm'], pd.DataFrame) else data['ttm']
    
    S_arr = None
    if data.get('S') is not None:
        S_arr = data['S'].values if isinstance(data['S'], pd.Series) else data['S']
    
    # Garantir tipos corretos
    F_mkt_arr = F_mkt_arr.astype(np.float64)
    ttm_arr = ttm_arr.astype(np.float64)
    if S_arr is not None:
        S_arr = S_arr.astype(np.float64)
    
    result = {
        'F_mkt': F_mkt_arr,
        'ttm': ttm_arr,
        'S': S_arr
    }
    
    logger.info(f"Dados formatados: F_mkt {F_mkt_arr.shape}, ttm {ttm_arr.shape}")
    if S_arr is not None:
        logger.info(f"S formatado: {S_arr.shape}")
    
    return result


def format_for_analysis(
    processed_data: Dict,
    raw_data: Dict
) -> Dict[str, pd.DataFrame]:
    """
    Formata dados para análise (mantém como DataFrames com índices temporais).
    
    Parâmetros
    ----------
    processed_data : dict
        Dados processados do backtest.
    raw_data : dict
        Dados raw originais.
    
    Retorna
    -------
    dict
        Dados formatados para análise com índices temporais apropriados.
    
    Exemplos
    --------
    >>> analysis_data = format_for_analysis(processed_data, raw_data)
    >>> print(analysis_data.keys())
    """
    logger.info("Formatando dados para análise...")
    
    analysis_data = {}
    
    # Manter dados raw como referência
    analysis_data['F_mkt_original'] = raw_data['F_mkt']
    analysis_data['dates'] = raw_data['dates']
    analysis_data['tenors'] = raw_data['tenors']
    
    # Processar dados do backtest
    for key, value in processed_data.items():
        if isinstance(value, pd.DataFrame):
            analysis_data[key] = value
        elif isinstance(value, dict) and 'daily_results' in key:
            # Expandir resultados diários
            analysis_data[f"{key}_expanded"] = _expand_daily_results(value)
    
    logger.info(f"Dados formatados para análise: {len(analysis_data)} conjuntos")
    return analysis_data


def create_time_series(
    data: np.ndarray,
    dates: pd.DatetimeIndex,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Cria série temporal padronizada a partir de array numpy.
    
    Parâmetros
    ----------
    data : np.ndarray
        Dados [T x M] ou [T].
    dates : pd.DatetimeIndex
        Índice temporal.
    columns : List[str], opcional
        Nomes das colunas.
    
    Retorna
    -------
    pd.DataFrame
        Série temporal com índice de datas.
    
    Exemplos
    --------
    >>> ts = create_time_series(data_array, dates, ['tenor_1', 'tenor_2'])
    >>> print(ts.head())
    """
    if data.ndim == 1:
        # Série univariada
        return pd.Series(data, index=dates, name=columns[0] if columns else 'value')
    else:
        # Série multivariada
        if columns is None:
            columns = [f'col_{i}' for i in range(data.shape[1])]
        return pd.DataFrame(data, index=dates, columns=columns)


def load_data_from_raw(dataset_id: str) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
    """
    Carrega dataset completo da pasta data/raw/.
    
    Parâmetros
    ----------
    dataset_id : str
        ID do dataset a ser carregado.
    
    Retorna
    -------
    dict
        Dados carregados com chaves 'F_mkt', 'ttm', 'S', 'costs', 'dates', 'tenors'.
    
    Exemplos
    --------
    >>> raw_data = load_data_from_raw("WTI_test_250d")
    >>> print(f"Dados carregados: {raw_data['F_mkt'].shape}")
    """
    import os
    
    dataset_path = os.path.join("data/raw", dataset_id)
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset '{dataset_id}' não encontrado em data/raw/")
    
    # Carregar arquivos principais
    F_mkt = pd.read_csv(os.path.join(dataset_path, "F_mkt.csv"), index_col=0, parse_dates=True)
    ttm = pd.read_csv(os.path.join(dataset_path, "ttm.csv"), index_col=0, parse_dates=True)
    
    # Carregar S (opcional)
    S_path = os.path.join(dataset_path, "S.csv")
    if os.path.exists(S_path):
        S = pd.read_csv(S_path, index_col=0, parse_dates=True).squeeze()
    else:
        S = F_mkt.iloc[:, 0].copy()  # Usar primeiro tenor como proxy
    
    # Carregar costs (opcional)
    costs_path = os.path.join(dataset_path, "costs.csv")
    if os.path.exists(costs_path):
        costs_df = pd.read_csv(costs_path)
        # Extrair apenas os valores tick_value como array
        if 'tick_value' in costs_df.columns:
            costs = costs_df['tick_value'].values
        else:
            costs = None
    else:
        costs = None
    
    return {
        'F_mkt': F_mkt,
        'ttm': ttm,
        'S': S,
        'costs': costs,
        'dates': F_mkt.index,
        'tenors': list(F_mkt.columns)
    }


def format_for_model(raw_data: Dict) -> Dict[str, pd.DataFrame]:
    """
    Formata dados brutos para uso no modelo Schwartz-Smith.
    
    Parâmetros
    ----------
    raw_data : dict
        Dados brutos carregados via load_data_from_raw.
    
    Retorna
    -------
    dict
        Dados formatados para o modelo.
    """
    return {
        'F_mkt': raw_data['F_mkt'],
        'ttm': raw_data['ttm'], 
        'S': raw_data.get('S')
    }


def save_data_to_processed(results_data: Dict, dataset_id: str) -> str:
    """
    Salva resultados de backtesting na pasta data/processed/.
    
    Parâmetros
    ----------
    results_data : dict
        Dados de resultado do backtesting.
    dataset_id : str
        ID do dataset.
    
    Retorna
    -------
    str
        Caminho do diretório criado.
    """
    import os
    import shutil
    
    # Criar pasta SEM timestamp (mesmo nome do raw)
    output_dir = os.path.join("data/processed", dataset_id)
    
    # Remover se já existir
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 🔧 Salvar DataFrames das 3 estratégias
    if 'portfolio_performance' in results_data:
        results_data['portfolio_performance'].to_csv(os.path.join(output_dir, "portfolio_performance.csv"))
    
    if 'portfolio_performance_tenor1' in results_data:
        results_data['portfolio_performance_tenor1'].to_csv(os.path.join(output_dir, "portfolio_performance_tenor1.csv"))
    
    if 'portfolio_performance_tenor2' in results_data:
        results_data['portfolio_performance_tenor2'].to_csv(os.path.join(output_dir, "portfolio_performance_tenor2.csv"))
    
    if 'trades_log' in results_data:
        results_data['trades_log'].to_csv(os.path.join(output_dir, "trades_log.csv"))
    
    if 'trades_log_tenor1' in results_data:
        results_data['trades_log_tenor1'].to_csv(os.path.join(output_dir, "trades_log_tenor1.csv"))
    
    if 'trades_log_tenor2' in results_data:
        results_data['trades_log_tenor2'].to_csv(os.path.join(output_dir, "trades_log_tenor2.csv"))
    
    if 'model_evolution' in results_data:
        results_data['model_evolution'].to_csv(os.path.join(output_dir, "model_evolution.csv"))
    
    # Salvar daily_results como JSON
    if 'daily_results' in results_data:
        import json
        daily_results_path = os.path.join(output_dir, "daily_results")
        os.makedirs(daily_results_path, exist_ok=True)
        
        for date_str, day_data in results_data['daily_results'].items():
            day_file = os.path.join(daily_results_path, f"{date_str}.json")
            with open(day_file, 'w') as f:
                json.dump(day_data, f, indent=2, default=str)
    
    return output_dir


def load_config(config_path: str = "config/default.yaml") -> dict:
    """
    Carrega configuração do arquivo YAML.
    
    Parâmetros
    ----------
    config_path : str
        Caminho para arquivo de configuração.
    
    Retorna
    -------
    dict
        Configurações carregadas.
    """
    import yaml
    import os
    
    if not os.path.exists(config_path):
        # Retornar configuração padrão se arquivo não existir
        return {
            'method': 'MLE',  # Método otimizado L-BFGS-B
            'calibration': {  # Nova configuração otimizada
                'max_iter': 500,
                'tol': 1e-6,
                'init_params': {
                    'kappa': 1.0,    # Baseado na literatura
                    'sigma_X': 0.2,  # Volatilidade razoável
                    'sigma_Y': 0.08, # Mínimo significativo
                    'rho': 0.0,      # Fatores independentes
                    'mu': 0.0        # Deriva neutra
                }
            },
            'ttm_adjustment': {  # 🔧 DESABILITAR TTM ajustado temporariamente
                'method': 'none',  # Usar TTM original
                'base_ttm_days': [30, 60],  # 1 mês, 2 meses
                'add_noise': False
            },
            # Manter compatibilidade com código existente
            'kalman': {
                'max_iter': 500,
                'tol': 1e-6,
                'init_params': {
                    'kappa': 1.0,
                    'sigma_X': 0.2,
                    'sigma_Y': 0.08,
                    'rho': 0.0,
                    'mu': 0.0
                }
            },
            'sizing': {
                'method': 'vol_target',
                'vol_target': 0.10
            },
            'thresh': {
                'z_in': 1.5,
                'z_out': 0.5,
                'topK': 4
            },
            'limits': {
                'leverage': 3.0,
                'per_tenor_cap': 0.3
            }
        }
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def format_for_analysis(data: Dict) -> Dict:
    """
    Formata dados para análise visual (placeholder).
    
    Parâmetros
    ---------- 
    data : dict
        Dados brutos.
        
    Retorna
    -------
    dict
        Dados formatados.
    """
    # Placeholder - será usado no analysis.py
    return data


def clean_data(
    F_mkt: pd.DataFrame,
    ttm: pd.DataFrame,
    max_na_ratio: float = 0.1,
    interpolate_method: str = 'linear'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Limpa dados removendo outliers e interpolando valores faltantes.
    
    Parâmetros
    ----------
    F_mkt : pd.DataFrame
        Preços futuros.
    ttm : pd.DataFrame
        Time-to-maturity.
    max_na_ratio : float
        Máxima proporção de NAs permitida por série.
    interpolate_method : str
        Método de interpolação.
    
    Retorna
    -------
    tuple
        (F_mkt_clean, ttm_clean)
    
    Exemplos
    --------
    >>> F_clean, ttm_clean = clean_data(F_mkt, ttm)
    >>> print(f"NAs removidos: {F_mkt.isnull().sum().sum() - F_clean.isnull().sum().sum()}")
    """
    logger.info("Limpando dados...")
    
    F_mkt_clean = F_mkt.copy()
    ttm_clean = ttm.copy()
    
    # Remover colunas com muitos NAs
    na_ratio_F = F_mkt_clean.isnull().sum() / len(F_mkt_clean)
    na_ratio_ttm = ttm_clean.isnull().sum() / len(ttm_clean)
    
    bad_cols_F = na_ratio_F[na_ratio_F > max_na_ratio].index
    bad_cols_ttm = na_ratio_ttm[na_ratio_ttm > max_na_ratio].index
    bad_cols = bad_cols_F.union(bad_cols_ttm)
    
    if len(bad_cols) > 0:
        logger.warning(f"Removendo colunas com muitos NAs: {list(bad_cols)}")
        F_mkt_clean = F_mkt_clean.drop(columns=bad_cols)
        ttm_clean = ttm_clean.drop(columns=bad_cols)
    
    # Interpolação
    F_mkt_clean = F_mkt_clean.interpolate(method=interpolate_method)
    ttm_clean = ttm_clean.interpolate(method=interpolate_method)
    
    # Forward fill e backward fill para extremos
    F_mkt_clean = F_mkt_clean.fillna(method='ffill').fillna(method='bfill')
    ttm_clean = ttm_clean.fillna(method='ffill').fillna(method='bfill')
    
    # Remover outliers (método simples: 3 sigma)
    for col in F_mkt_clean.columns:
        series = F_mkt_clean[col]
        mean_val = series.mean()
        std_val = series.std()
        outlier_mask = np.abs(series - mean_val) > 3 * std_val
        
        if outlier_mask.sum() > 0:
            logger.info(f"Removendo {outlier_mask.sum()} outliers da coluna {col}")
            F_mkt_clean.loc[outlier_mask, col] = np.nan
            F_mkt_clean[col] = F_mkt_clean[col].interpolate()
    
    logger.info(f"Dados limpos: shape final {F_mkt_clean.shape}")
    return F_mkt_clean, ttm_clean


# ==========================================
# FUNÇÕES AUXILIARES (PRIVADAS)
# ==========================================

def _save_daily_results(daily_results: Dict, daily_dir: str) -> None:
    """Salva resultados diários em subpastas por data."""
    for date_str, day_data in daily_results.items():
        day_dir = os.path.join(daily_dir, date_str)
        os.makedirs(day_dir, exist_ok=True)
        
        for key, value in day_data.items():
            if isinstance(value, pd.DataFrame):
                filepath = os.path.join(day_dir, f"{key}.csv")
                value.to_csv(filepath)
            elif isinstance(value, (dict, list, np.ndarray)):
                filepath = os.path.join(day_dir, f"{key}.pkl")
                with open(filepath, 'wb') as f:
                    pickle.dump(value, f)


def _expand_daily_results(daily_results: Dict) -> pd.DataFrame:
    """Expande resultados diários em DataFrame consolidado."""
    # Implementação simplificada - expandir conforme necessário
    all_data = []
    
    for date_str, day_data in daily_results.items():
        try:
            date_obj = pd.to_datetime(date_str)
            row_data = {'date': date_obj}
            
            # Extrair métricas escalares
            for key, value in day_data.items():
                if isinstance(value, (float, int)):
                    row_data[key] = value
                elif isinstance(value, dict):
                    # Expandir dicts de métricas
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, (float, int)):
                            row_data[f"{key}_{subkey}"] = subvalue
            
            all_data.append(row_data)
            
        except Exception as e:
            logger.warning(f"Erro ao expandir data {date_str}: {e}")
    
    if all_data:
        return pd.DataFrame(all_data).set_index('date')
    else:
        return pd.DataFrame()


def _get_default_config() -> dict:
    """Retorna configuração padrão."""
    return {
        'kalman': {
            'method': 'MLE',
            'save_path': True,
            'max_iter': 200,
            'tol': 1e-6,
            'init_params': {
                'kappa': 1.0,
                'sigma_X': 0.3,
                'sigma_Y': 0.2,
                'rho': 0.3,
                'mu': 0.0
            },
            'R': 0.01
        },
        'risk': {
            'source': 'returns',
            'lookback': 60,
            'shrinkage': True
        },
        'sizing': {
            'method': 'vol_target',
            'vol_target': 0.10,
            'qp': {
                'gamma': 5.0,
                'lambda_l1': 0.0,
                'lambda_turnover': 0.1
            }
        },
        'limits': {
            'leverage': 3.0,
            'per_tenor_cap': 0.3
        },
        'thresh': {
            'z_in': 1.5,
            'z_out': 0.5,
            'topK': 4
        },
        'costs': {
            'default_tick_value': 10.0,
            'default_fee': 2.0,
            'slippage': 0.0001
        }
    }