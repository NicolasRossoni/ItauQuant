"""
Download.py

Módulo de download de dados reais de futuros de commodities.
Contém toda a burocracia de downloads de APIs, conexões, autenticação, parsing,
formatação para estrutura padrão e tratamento de erros de rede.

FUNÇÕES PÚBLICAS PRINCIPAIS:
- download_yahoo_wti(...): baixa dados de WTI via Yahoo Finance
- download_cme_data(...): baixa dados via API CME (placeholder)
- format_raw_data(...): formata dados para estrutura padrão
- save_raw_dataset(...): salva dataset no formato padronizado

ATENÇÃO: Este módulo implementa downloads conforme padrões do help/Architecture.md.
Abstrai toda complexidade de conectividade para deixar download.py como fluxograma limpo.
"""

import os
import logging
from typing import List, Tuple, Dict, Optional
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
import requests
import time

# Imports condicionais para dependências opcionais
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    yf = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constantes para futuros
MONTH_CODES = ['F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z']
MONTH_MAP = {
    'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
    'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12
}


def download_yahoo_wti(
    start_date: str,
    end_date: str,
    num_tenors: int = 8,
    contango_monthly: float = 0.004
) -> Dict[str, pd.DataFrame]:
    """
    Baixa dados de WTI via Yahoo Finance e cria estrutura de termo sintética.
    
    Yahoo Finance só tem CL=F (contínuo front month). Esta função baixa o preço
    spot e cria uma curva forward sintética baseada em contango típico.
    
    Parâmetros
    ----------
    start_date : str
        Data inicial no formato 'YYYY-MM-DD'.
    end_date : str
        Data final no formato 'YYYY-MM-DD'.
    num_tenors : int
        Número de tenores (meses) a criar.
    contango_monthly : float
        Contango mensal para criar estrutura de termo (padrão: 0.4%).
    
    Retorna
    -------
    dict
        {
            'F_mkt': pd.DataFrame [T x M] com preços futuros sintéticos,
            'ttm': pd.DataFrame [T x M] com time-to-maturity,
            'S': pd.DataFrame [T x 1] com preços spot,
            'info': dict com metadados
        }
    
    Exceções
    --------
    RuntimeError
        Se yfinance não estiver disponível ou dados não puderem ser baixados.
    
    Exemplos
    --------
    >>> data = download_yahoo_wti("2024-01-01", "2024-12-31", num_tenors=6)
    >>> print(f"Dados baixados: {data['F_mkt'].shape}")
    >>> print(f"Período: {data['F_mkt'].index[0]} a {data['F_mkt'].index[-1]}")
    """
    logger.info("=== Iniciando download Yahoo WTI ===")
    
    if not YFINANCE_AVAILABLE:
        raise RuntimeError("yfinance não está disponível. Execute: pip install yfinance")
    
    logger.info(f"Período: {start_date} → {end_date}")
    logger.info(f"Tenores: {num_tenors}")
    logger.info(f"Contango mensal: {contango_monthly*100:.2f}%")
    
    # Baixar CL=F (WTI front month contínuo)
    try:
        logger.info("Baixando CL=F (WTI front month)...")
        df = yf.download('CL=F', start=start_date, end=end_date, 
                        interval="1d", progress=False, auto_adjust=False)
        
        if df is None or df.empty:
            raise RuntimeError("CL=F não retornou dados válidos do Yahoo Finance")
        
        spot_prices = df['Close'].dropna()
        logger.info(f"✓ CL=F baixado: {len(spot_prices)} dias úteis")
        
    except Exception as e:
        raise RuntimeError(f"Erro ao baixar CL=F via Yahoo Finance: {e}")
    
    # Criar estrutura de termo sintética
    logger.info("Criando estrutura de termo sintética...")
    
    price_data = []
    ttm_data = []
    tenor_names = []
    
    for i in range(1, num_tenors + 1):
        # TTM em anos (fixo por tenor)
        ttm_years = i / 12.0
        
        # Preço forward sintético com contango
        forward_prices = spot_prices * (1 + contango_monthly * i)
        price_data.append(forward_prices)
        
        # TTM constante para cada tenor
        ttm_series = pd.Series(
            [ttm_years] * len(spot_prices),
            index=spot_prices.index,
            name=f'tenor_{i}'
        )
        ttm_data.append(ttm_series)
        tenor_names.append(f'tenor_{i}')
        
        logger.info(f"  Tenor {i}: TTM={ttm_years:.2f} anos, "
                    f"contango={contango_monthly*i*100:.1f}%, "
                    f"preço médio=${forward_prices.mean():.2f}")
    
    # Consolidar DataFrames
    F_mkt = pd.concat(price_data, axis=1)
    F_mkt.columns = tenor_names
    
    ttm = pd.concat(ttm_data, axis=1)
    ttm.columns = tenor_names
    
    # Spot como DataFrame
    S = pd.DataFrame({'S': spot_prices})
    
    # Remover linhas com NaN
    common_index = F_mkt.dropna().index
    F_mkt = F_mkt.loc[common_index]
    ttm = ttm.loc[common_index]
    S = S.loc[common_index]
    
    # Metadados
    info = {
        'source': 'Yahoo Finance',
        'symbol': 'CL=F',
        'start_date': start_date,
        'end_date': end_date,
        'num_tenors': num_tenors,
        'contango_monthly': contango_monthly,
        'days_downloaded': len(common_index),
        'price_range': {
            'min': float(F_mkt.min().min()),
            'max': float(F_mkt.max().max()),
            'mean': float(F_mkt.mean().mean())
        }
    }
    
    result = {
        'F_mkt': F_mkt,
        'ttm': ttm,
        'S': S,
        'info': info
    }
    
    logger.info(f"✓ Estrutura de termo criada: {F_mkt.shape}")
    logger.info(f"  Período final: {F_mkt.index[0].date()} → {F_mkt.index[-1].date()}")
    logger.info(f"  Preço médio: ${info['price_range']['mean']:.2f}")
    
    return result


def download_cme_data(
    commodity: str,
    start_date: str,
    end_date: str,
    api_key: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Baixa dados via API CME Group (PLACEHOLDER).
    
    ATENÇÃO: Esta função é um placeholder para implementação futura.
    Atualmente retorna dados sintéticos para compatibilidade.
    
    Parâmetros
    ----------
    commodity : str
        Código da commodity (ex: 'WTI', 'NG', etc.).
    start_date : str
        Data inicial.
    end_date : str
        Data final.
    api_key : str, opcional
        Chave da API CME.
    
    Retorna
    -------
    dict
        Estrutura similar ao download_yahoo_wti.
    
    Exemplos
    --------
    >>> # Será implementado futuramente
    >>> data = download_cme_data("WTI", "2024-01-01", "2024-12-31")
    """
    logger.warning("download_cme_data é um PLACEHOLDER. Implementação futura.")
    logger.info(f"Gerando dados sintéticos para {commodity}")
    
    # Implementação placeholder - gerar dados sintéticos simples
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    np.random.seed(42)  # Para reprodutibilidade
    
    # Simular preços com random walk
    base_price = 80.0  # Preço base para WTI
    returns = np.random.normal(0, 0.02, len(date_range))  # 2% vol diária
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Criar 6 tenores com contango
    num_tenors = 6
    F_mkt_data = {}
    ttm_data = {}
    
    for i in range(1, num_tenors + 1):
        ttm_years = i / 12.0
        forward_prices = prices * (1 + 0.003 * i)  # 0.3% contango mensal
        
        F_mkt_data[f'tenor_{i}'] = forward_prices
        ttm_data[f'tenor_{i}'] = [ttm_years] * len(date_range)
    
    F_mkt = pd.DataFrame(F_mkt_data, index=date_range)
    ttm = pd.DataFrame(ttm_data, index=date_range)
    S = pd.DataFrame({'S': prices}, index=date_range)
    
    # Remover finais de semana (simulação mais realista)
    business_days = F_mkt.index[F_mkt.index.weekday < 5]
    F_mkt = F_mkt.loc[business_days]
    ttm = ttm.loc[business_days]
    S = S.loc[business_days]
    
    info = {
        'source': 'Synthetic (CME placeholder)',
        'commodity': commodity,
        'start_date': start_date,
        'end_date': end_date,
        'num_tenors': num_tenors,
        'days_generated': len(F_mkt)
    }
    
    logger.info(f"✓ Dados sintéticos gerados: {F_mkt.shape}")
    return {'F_mkt': F_mkt, 'ttm': ttm, 'S': S, 'info': info}


def format_raw_data(
    raw_data: Dict[str, pd.DataFrame],
    validate: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Formata dados brutos para estrutura padrão do projeto.
    
    Parâmetros
    ----------
    raw_data : dict
        Dados brutos com chaves 'F_mkt', 'ttm', 'S', etc.
    validate : bool
        Se deve validar estrutura dos dados.
    
    Retorna
    -------
    dict
        Dados formatados e validados.
    
    Exemplos
    --------
    >>> raw = download_yahoo_wti("2024-01-01", "2024-03-31")
    >>> formatted = format_raw_data(raw)
    >>> print(f"Dados formatados: {formatted['F_mkt'].dtypes}")
    """
    logger.info("Formatando dados para estrutura padrão...")
    
    formatted_data = {}
    
    # Formatar F_mkt
    if 'F_mkt' in raw_data:
        F_mkt = raw_data['F_mkt'].copy()
        # Garantir tipos float64
        F_mkt = F_mkt.astype(np.float64)
        # Ordenar por data
        F_mkt = F_mkt.sort_index()
        formatted_data['F_mkt'] = F_mkt
        logger.info(f"F_mkt formatado: {F_mkt.shape}, dtype: {F_mkt.dtypes.iloc[0]}")
    
    # Formatar ttm
    if 'ttm' in raw_data:
        ttm = raw_data['ttm'].copy()
        ttm = ttm.astype(np.float64)
        ttm = ttm.sort_index()
        formatted_data['ttm'] = ttm
        logger.info(f"ttm formatado: {ttm.shape}, dtype: {ttm.dtypes.iloc[0]}")
    
    # Formatar S
    if 'S' in raw_data:
        S = raw_data['S'].copy()
        if isinstance(S, pd.DataFrame):
            S = S.squeeze()  # Converter para Series se necessário
        S = S.astype(np.float64)
        S = S.sort_index()
        formatted_data['S'] = S
        logger.info(f"S formatado: {S.shape}, dtype: {S.dtype}")
    
    # Carregar info se disponível
    if 'info' in raw_data:
        formatted_data['info'] = raw_data['info']
    
    # Validação opcional
    if validate and 'F_mkt' in formatted_data and 'ttm' in formatted_data:
        from .DataManipulation import validate_data_structure
        
        is_valid, errors = validate_data_structure(
            formatted_data['F_mkt'], 
            formatted_data['ttm'], 
            formatted_data.get('S')
        )
        
        if not is_valid:
            logger.warning(f"Dados têm problemas de validação: {'; '.join(errors)}")
        else:
            logger.info("✓ Dados validados com sucesso")
    
    return formatted_data


def save_raw_dataset(
    formatted_data: Dict[str, pd.DataFrame],
    dataset_id: str,
    output_path: str = "data/raw"
) -> str:
    """
    Salva dataset no formato padronizado para uso no pipeline.
    
    Parâmetros
    ----------
    formatted_data : dict
        Dados formatados (F_mkt, ttm, S, etc.).
    dataset_id : str
        ID único do dataset.
    output_path : str
        Caminho base para salvar dados raw.
    
    Retorna
    -------
    str
        Caminho completo do diretório criado.
    
    Exemplos
    --------
    >>> path = save_raw_dataset(formatted_data, "WTI_2024_yahoo")
    >>> print(f"Dataset salvo em: {path}")
    """
    logger.info(f"=== Salvando dataset: {dataset_id} ===")
    
    # Criar diretório
    dataset_path = os.path.join(output_path, dataset_id)
    os.makedirs(dataset_path, exist_ok=True)
    
    files_saved = []
    
    # Salvar F_mkt (obrigatório)
    if 'F_mkt' in formatted_data:
        F_mkt_path = os.path.join(dataset_path, "F_mkt.csv")
        formatted_data['F_mkt'].to_csv(F_mkt_path)
        files_saved.append("F_mkt.csv")
        logger.info(f"✓ Salvo: F_mkt.csv ({formatted_data['F_mkt'].shape})")
    else:
        raise ValueError("F_mkt é obrigatório mas não foi fornecido")
    
    # Salvar ttm (obrigatório)
    if 'ttm' in formatted_data:
        ttm_path = os.path.join(dataset_path, "ttm.csv")
        formatted_data['ttm'].to_csv(ttm_path)
        files_saved.append("ttm.csv")
        logger.info(f"✓ Salvo: ttm.csv ({formatted_data['ttm'].shape})")
    else:
        raise ValueError("ttm é obrigatório mas não foi fornecido")
    
    # Salvar S (opcional)
    if 'S' in formatted_data:
        S_path = os.path.join(dataset_path, "S.csv")
        if isinstance(formatted_data['S'], pd.Series):
            formatted_data['S'].to_csv(S_path, header=True)
        else:
            formatted_data['S'].to_csv(S_path)
        files_saved.append("S.csv")
        logger.info(f"✓ Salvo: S.csv ({len(formatted_data['S'])} pontos)")
    
    # Salvar costs (opcional, criar padrão se não fornecido)
    costs_path = os.path.join(dataset_path, "costs.csv")
    if 'costs' in formatted_data:
        formatted_data['costs'].to_csv(costs_path, index=False)
    else:
        # Criar costs padrão baseado nos tenores
        M = formatted_data['F_mkt'].shape[1]
        default_costs = pd.DataFrame({
            'tenor': [f'tenor_{i+1}' for i in range(M)],
            'tick_value': [10.0] * M,
            'fee_per_contract': [2.5] * M
        })
        default_costs.to_csv(costs_path, index=False)
    
    files_saved.append("costs.csv")
    logger.info("✓ Salvo: costs.csv (padrão)")
    
    # Salvar metadados se disponíveis
    if 'info' in formatted_data:
        import json
        info_path = os.path.join(dataset_path, "info.json")
        with open(info_path, 'w') as f:
            json.dump(formatted_data['info'], f, indent=2, default=str)
        files_saved.append("info.json")
        logger.info("✓ Salvo: info.json")
    
    logger.info(f"✓ Dataset salvo em: {dataset_path}")
    logger.info(f"  Arquivos: {', '.join(files_saved)}")
    
    return dataset_path


def create_synthetic_dataset(
    dataset_id: str,
    start_date: str,
    end_date: str,
    num_tenors: int = 6,
    base_price: float = 80.0,
    volatility: float = 0.25,
    drift: float = 0.05,
    contango: float = 0.003,
    output_path: str = "data/raw"
) -> str:
    """
    Cria dataset sintético para testes e desenvolvimento.
    
    Útil quando não há acesso a dados reais ou para testes rápidos.
    
    Parâmetros
    ----------
    dataset_id : str
        ID do dataset sintético.
    start_date : str
        Data inicial.
    end_date : str
        Data final.
    num_tenors : int
        Número de tenores.
    base_price : float
        Preço base inicial.
    volatility : float
        Volatilidade anual.
    drift : float
        Deriva anual.
    contango : float
        Contango mensal entre tenores.
    output_path : str
        Caminho para salvar.
    
    Retorna
    -------
    str
        Caminho do dataset criado.
    
    Exemplos
    --------
    >>> path = create_synthetic_dataset("synthetic_test", "2024-01-01", "2024-03-31")
    >>> print(f"Dataset sintético: {path}")
    """
    logger.info(f"=== Criando dataset sintético: {dataset_id} ===")
    
    # Gerar datas (apenas dias úteis)
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    business_dates = all_dates[all_dates.weekday < 5]  # Apenas seg-sex
    T = len(business_dates)
    
    logger.info(f"Gerando {T} dias úteis de dados sintéticos...")
    
    # Seed para reprodutibilidade
    np.random.seed(hash(dataset_id) % 2**32)
    
    # Simular processo de preços (GBM)
    dt = 1/252  # Dias úteis por ano
    daily_vol = volatility / np.sqrt(252)
    daily_drift = drift / 252
    
    # Retornos log-normais
    log_returns = np.random.normal(daily_drift, daily_vol, T)
    log_prices = np.log(base_price) + np.cumsum(log_returns)
    spot_prices = np.exp(log_prices)
    
    # Criar estrutura de termo com contango
    F_mkt_data = {}
    ttm_data = {}
    
    for i in range(1, num_tenors + 1):
        ttm_years = i / 12.0
        # Forward = Spot * exp(contango * T) com ruído adicional
        noise = np.random.normal(0, 0.005, T)  # Ruído de 0.5%
        forward_prices = spot_prices * np.exp(contango * i) * np.exp(noise)
        
        F_mkt_data[f'tenor_{i}'] = forward_prices
        ttm_data[f'tenor_{i}'] = [ttm_years] * T
    
    # Criar DataFrames
    F_mkt = pd.DataFrame(F_mkt_data, index=business_dates)
    ttm = pd.DataFrame(ttm_data, index=business_dates)
    S = pd.Series(spot_prices, index=business_dates, name='S')
    
    # Metadados
    info = {
        'type': 'synthetic',
        'dataset_id': dataset_id,
        'start_date': start_date,
        'end_date': end_date,
        'num_tenors': num_tenors,
        'base_price': base_price,
        'volatility': volatility,
        'drift': drift,
        'contango': contango,
        'days_generated': T,
        'seed': hash(dataset_id) % 2**32
    }
    
    # Formatar e salvar
    data = {
        'F_mkt': F_mkt,
        'ttm': ttm,
        'S': S,
        'info': info
    }
    
    formatted_data = format_raw_data(data, validate=True)
    dataset_path = save_raw_dataset(formatted_data, dataset_id, output_path)
    
    logger.info(f"✓ Dataset sintético criado: {dataset_path}")
    logger.info(f"  Parâmetros: vol={volatility:.1%}, drift={drift:.1%}, contango={contango:.1%}")
    
    return dataset_path


# ==========================================
# FUNÇÕES AUXILIARES (PRIVADAS)
# ==========================================

def _validate_date_format(date_str: str) -> bool:
    """Valida formato de data YYYY-MM-DD."""
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False


def _check_network_connection(url: str = "https://finance.yahoo.com", timeout: int = 5) -> bool:
    """Verifica conectividade de rede."""
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code == 200
    except requests.RequestException:
        return False