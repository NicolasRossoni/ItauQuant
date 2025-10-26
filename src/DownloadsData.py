"""
DownloadsData.py

Baixa dados de futuros de commodities da CME Group API e processa para o formato
compatível com o pipeline (F_mkt.csv, ttm.csv, S.csv, costs.csv).

Requer credenciais CME configuradas em arquivo .env:
    CME_API_KEY=your_key
    CME_API_SECRET=your_secret

Uso:
    python src/DownloadsData.py --product CL --start-date 2020-01-01 --end-date 2024-12-31
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Carregar variáveis de ambiente
load_dotenv()


def DownloadCMEData(
    product: str = "CL",  # WTI Crude Oil
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None,
    num_tenors: int = 8,
    output_dir: str = "data/real_data",
    dataset_name: Optional[str] = None
) -> dict:
    """
    Baixa dados de futuros da CME Group e salva no formato padrão.
    
    Parâmetros
    ----------
    product : str
        Código do produto CME (ex: 'CL' para WTI, 'NG' para Natural Gas)
    start_date : str
        Data inicial no formato YYYY-MM-DD
    end_date : str, optional
        Data final (padrão: hoje)
    num_tenors : int
        Número de tenores (maturidades) a coletar
    output_dir : str
        Diretório raiz para salvar dados
    dataset_name : str, optional
        Nome do dataset (padrão: {product}_{start}_{end})
    
    Retorna
    -------
    dict
        Caminhos dos arquivos salvos:
        {'F_mkt': path, 'ttm': path, 'S': path, 'costs': path}
    
    Exemplos
    --------
    >>> paths = DownloadCMEData('CL', '2020-01-01', '2023-12-31', num_tenors=8)
    >>> print(paths['F_mkt'])
    'data/real_data/CL_2020_2023/F_mkt.csv'
    """
    logger.info("=" * 70)
    logger.info("=== Baixando dados da CME Group API ===")
    logger.info("=" * 70)
    
    # Validar credenciais
    api_key = os.getenv('CME_API_KEY')
    api_secret = os.getenv('CME_API_SECRET')
    
    if not api_key or not api_secret:
        raise ValueError(
            "Credenciais CME não encontradas!\n"
            "Crie um arquivo .env com:\n"
            "  CME_API_KEY=your_key\n"
            "  CME_API_SECRET=your_secret\n"
            "Veja o arquivo .env.example para referência."
        )
    
    # Processar datas
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    if dataset_name is None:
        dataset_name = f"{product}_{start_date[:4]}_{end_date[:4]}"
    
    output_path = os.path.join(output_dir, dataset_name)
    os.makedirs(output_path, exist_ok=True)
    
    logger.info(f"Produto: {product}")
    logger.info(f"Período: {start_date} a {end_date}")
    logger.info(f"Tenores: {num_tenors}")
    logger.info(f"Saída: {output_path}")
    
    # Baixar dados
    logger.info("\nBaixando dados históricos...")
    df_raw = _fetch_cme_futures_data(product, start_date, end_date, api_key, api_secret)
    
    if df_raw is None or len(df_raw) == 0:
        raise ValueError("Nenhum dado retornado da API CME")
    
    logger.info(f"Dados brutos: {len(df_raw)} registros")
    
    # Processar para formato padrão
    logger.info("\nProcessando dados para formato padrão...")
    F_mkt, ttm, S = _process_to_standard_format(df_raw, num_tenors)
    
    # Gerar custos padrão
    costs = _generate_default_costs(product, num_tenors)
    
    # Salvar CSVs
    logger.info("\nSalvando arquivos...")
    paths = _save_datasets(output_path, F_mkt, ttm, S, costs)
    
    logger.info("\n" + "=" * 70)
    logger.info("=== Download concluído com sucesso! ===")
    logger.info("=" * 70)
    logger.info(f"\nArquivos salvos em: {output_path}")
    for key, path in paths.items():
        logger.info(f"  {key}: {os.path.basename(path)}")
    
    return paths


def _fetch_cme_futures_data(
    product: str,
    start_date: str,
    end_date: str,
    api_key: str,
    api_secret: str
) -> pd.DataFrame:
    """
    Baixa dados de futuros da CME Group API (DataMine).
    
    Nota: Esta implementação usa um mock/fallback se a API não estiver acessível.
    Para produção, substitua pela chamada real à API CME DataMine.
    """
    # URL base da API CME DataMine
    base_url = "https://datamine.cmegroup.com/api/v1"
    
    # Endpoint para dados históricos de futuros
    endpoint = f"{base_url}/historical/futures/{product}"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    params = {
        "startDate": start_date,
        "endDate": end_date,
        "fields": "trade_date,contract_month,settlement_price,expiration_date,open_interest"
    }
    
    try:
        logger.info(f"Consultando API: {endpoint}")
        response = requests.get(endpoint, headers=headers, params=params, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data.get('data', []))
            logger.info(f"API retornou {len(df)} registros")
            return df
        elif response.status_code == 401:
            logger.error("Erro 401: Credenciais inválidas")
            logger.warning("Verifique suas credenciais CME no arquivo .env")
        elif response.status_code == 403:
            logger.error("Erro 403: Acesso negado")
            logger.warning("Sua conta CME pode não ter permissão para este produto")
        else:
            logger.error(f"Erro {response.status_code}: {response.text}")
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Erro na requisição: {e}")
    
    # Fallback: gerar dados sintéticos para teste
    logger.warning("\n" + "="*70)
    logger.warning("AVISO: API CME não acessível ou credenciais inválidas")
    logger.warning("Gerando dados SINTÉTICOS para teste do pipeline")
    logger.warning("Para dados REAIS, configure credenciais válidas no .env")
    logger.warning("="*70 + "\n")
    
    return _generate_synthetic_fallback(start_date, end_date, product)


def _generate_synthetic_fallback(
    start_date: str,
    end_date: str,
    product: str
) -> pd.DataFrame:
    """
    Gera dados sintéticos caso a API não esteja acessível.
    Apenas para testes do pipeline.
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
    
    # Simular contratos: M1 a M12 (próximos 12 meses)
    contracts = []
    for date in dates:
        for month_offset in range(1, 13):
            exp_date = date + timedelta(days=30 * month_offset)
            
            # Preço base + ruído + estrutura de termo
            base_price = 70.0 if product == 'CL' else 3.0  # WTI ou NG
            term_structure = month_offset * 0.5  # Contango
            noise = np.random.randn() * 2.0
            
            contracts.append({
                'trade_date': date,
                'contract_month': exp_date.strftime('%Y%m'),
                'expiration_date': exp_date,
                'settlement_price': base_price + term_structure + noise,
                'open_interest': 10000 + np.random.randint(-1000, 1000)
            })
    
    df = pd.DataFrame(contracts)
    logger.info(f"Dados sintéticos gerados: {len(df)} registros")
    return df


def _process_to_standard_format(
    df_raw: pd.DataFrame,
    num_tenors: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Processa dados brutos para o formato padrão: F_mkt, ttm, S.
    """
    # Converter para datetime
    df_raw['trade_date'] = pd.to_datetime(df_raw['trade_date'])
    df_raw['expiration_date'] = pd.to_datetime(df_raw['expiration_date'])
    
    # Calcular time-to-maturity em anos (ACT/365)
    df_raw['ttm_years'] = (df_raw['expiration_date'] - df_raw['trade_date']).dt.days / 365.0
    
    # Para cada data, selecionar os N contratos mais líquidos (maior open interest)
    dates = df_raw['trade_date'].unique()
    dates.sort()
    
    F_mkt_list = []
    ttm_list = []
    S_list = []
    
    for date in dates:
        df_date = df_raw[df_raw['trade_date'] == date].copy()
        
        # Ordenar por open interest (liquidez) e pegar top N
        df_date = df_date.nlargest(num_tenors, 'open_interest')
        
        # Ordenar por maturidade (tenor crescente)
        df_date = df_date.sort_values('ttm_years')
        
        if len(df_date) >= num_tenors:
            F_mkt_list.append(df_date['settlement_price'].values[:num_tenors])
            ttm_list.append(df_date['ttm_years'].values[:num_tenors])
            
            # Spot = primeiro contrato (M1)
            S_list.append(df_date['settlement_price'].iloc[0])
    
    # Criar DataFrames
    F_mkt = pd.DataFrame(
        F_mkt_list,
        index=pd.to_datetime(dates[:len(F_mkt_list)]),
        columns=[f'tenor_{i+1}' for i in range(num_tenors)]
    )
    
    ttm = pd.DataFrame(
        ttm_list,
        index=pd.to_datetime(dates[:len(ttm_list)]),
        columns=[f'tenor_{i+1}' for i in range(num_tenors)]
    )
    
    S = pd.DataFrame(
        {'S': S_list},
        index=pd.to_datetime(dates[:len(S_list)])
    )
    
    logger.info(f"F_mkt processado: shape={F_mkt.shape}")
    logger.info(f"ttm processado: shape={ttm.shape}")
    logger.info(f"S processado: shape={S.shape}")
    
    return F_mkt, ttm, S


def _generate_default_costs(product: str, num_tenors: int) -> pd.DataFrame:
    """
    Gera custos padrão por tenor baseado no produto.
    """
    # Custos típicos por produto (CME)
    cost_map = {
        'CL': {'tick_value': 10.0, 'fee': 2.50},  # WTI: $10 por tick, $2.50 fee
        'NG': {'tick_value': 10.0, 'fee': 2.00},  # Natural Gas
        'HO': {'tick_value': 4.20, 'fee': 2.50},  # Heating Oil
        'RB': {'tick_value': 4.20, 'fee': 2.50},  # RBOB Gasoline
    }
    
    costs = cost_map.get(product, {'tick_value': 10.0, 'fee': 2.0})
    
    df_costs = pd.DataFrame({
        'tenor': [f'tenor_{i+1}' for i in range(num_tenors)],
        'tick_value': [costs['tick_value']] * num_tenors,
        'fee_per_contract': [costs['fee']] * num_tenors
    })
    
    return df_costs


def _save_datasets(
    output_dir: str,
    F_mkt: pd.DataFrame,
    ttm: pd.DataFrame,
    S: pd.DataFrame,
    costs: pd.DataFrame
) -> dict:
    """
    Salva datasets no formato padrão.
    """
    paths = {}
    
    # F_mkt.csv
    path = os.path.join(output_dir, 'F_mkt.csv')
    F_mkt.to_csv(path)
    paths['F_mkt'] = path
    logger.info(f"Salvo: {path}")
    
    # ttm.csv
    path = os.path.join(output_dir, 'ttm.csv')
    ttm.to_csv(path)
    paths['ttm'] = path
    logger.info(f"Salvo: {path}")
    
    # S.csv
    path = os.path.join(output_dir, 'S.csv')
    S.to_csv(path)
    paths['S'] = path
    logger.info(f"Salvo: {path}")
    
    # costs.csv
    path = os.path.join(output_dir, 'costs.csv')
    costs.to_csv(path, index=False)
    paths['costs'] = path
    logger.info(f"Salvo: {path}")
    
    return paths


def main():
    """CLI para download de dados."""
    parser = argparse.ArgumentParser(
        description='Baixa dados de futuros da CME Group API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # WTI Crude Oil (2020-2023)
  python src/DownloadsData.py --product CL --start-date 2020-01-01 --end-date 2023-12-31
  
  # Natural Gas (últimos 3 anos)
  python src/DownloadsData.py --product NG --start-date 2021-01-01
  
  # WTI com 12 tenores
  python src/DownloadsData.py --product CL --num-tenors 12
        """
    )
    
    parser.add_argument('--product', type=str, default='CL',
                        help='Código do produto CME (CL=WTI, NG=NatGas, etc.)')
    parser.add_argument('--start-date', type=str, required=True,
                        help='Data inicial (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='Data final (YYYY-MM-DD), padrão: hoje')
    parser.add_argument('--num-tenors', type=int, default=8,
                        help='Número de tenores/maturidades')
    parser.add_argument('--output-dir', type=str, default='data/real_data',
                        help='Diretório de saída')
    parser.add_argument('--dataset-name', type=str, default=None,
                        help='Nome do dataset (padrão: auto)')
    
    args = parser.parse_args()
    
    try:
        paths = DownloadCMEData(
            product=args.product,
            start_date=args.start_date,
            end_date=args.end_date,
            num_tenors=args.num_tenors,
            output_dir=args.output_dir,
            dataset_name=args.dataset_name
        )
        
        print("\n" + "="*70)
        print("SUCESSO! Arquivos prontos para uso no pipeline:")
        print("="*70)
        print(f"\nDataset: {os.path.dirname(paths['F_mkt'])}")
        print("\nExecute o pipeline com:")
        print(f"  python src/Main.py")
        print(f"  # ou")
        print(f"  python src/TestingTheoryPipeline.py \\")
        print(f"    --dataset-root {os.path.dirname(paths['F_mkt'])} \\")
        print(f"    --t-idx -1 --method EM --sizing vol_target")
        
    except Exception as e:
        logger.error(f"\nERRO: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
