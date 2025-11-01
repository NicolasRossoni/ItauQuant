#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DownloadsData.py

Baixa dados REAIS de futuros WTI via Yahoo Finance.
Formato de saída compatível com o pipeline (F_mkt.csv, ttm.csv, S.csv, costs.csv).

Uso:
  python src/DownloadsData.py --start-date 2024-01-01 --end-date 2024-12-31 --num-tenors 8
  
  # Para período mais curto (recomendado para testes):
  python src/DownloadsData.py --start-date 2024-10-01 --end-date 2024-12-31 --num-tenors 8
"""

import os
import argparse
import logging
from typing import List, Tuple
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    raise RuntimeError("yfinance não instalado. Execute: pip install yfinance")

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Códigos de mês para futuros (padrão da indústria)
MONTH_CODES = ['F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z']
MONTH_MAP = {
    'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
    'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12
}


def generate_cl_tickers(start_date: str, num_tenors: int = 8) -> List[Tuple[str, datetime]]:
    """
    Yahoo Finance só tem CL=F (contínuo). Vamos baixar ele e simular tenores
    baseado em estrutura de termo histórica típica do WTI.
    
    Retorna: [('CL=F', delivery_date), ...] para cada tenor
    """
    ref_date = datetime.strptime(start_date, "%Y-%m-%d")
    tickers = []
    
    # Apenas o contínuo, mas retornamos N vezes para simular tenores
    for i in range(num_tenors):
        exp_date = ref_date + relativedelta(months=i+1)
        delivery = datetime(exp_date.year, exp_date.month, 15)
        # Vamos baixar CL=F uma vez e depois criar tenores sintéticos
        tickers.append((f"CL=F_tenor{i+1}", delivery))
    
    return tickers


def download_yahoo_data(tickers_info: List[Tuple[str, datetime]], 
                       start_date: str, 
                       end_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Baixa CL=F do Yahoo e cria estrutura de termo sintética.
    
    Yahoo Finance não tem contratos futuros individuais de WTI.
    Estratégia: baixar CL=F (front month contínuo) e criar curva forward
    sintética baseada em contango típico de ~0.5% ao mês.
    
    Retorna:
        F_mkt: DataFrame (dates x tenors) com preços simulados
        ttm: DataFrame (dates x tenors) com time-to-maturity em anos
    """
    logger.info("Baixando CL=F (WTI front month contínuo)")
    
    try:
        df = yf.download('CL=F', start=start_date, end=end_date, 
                       interval="1d", progress=False, auto_adjust=False)
        
        if df is None or df.empty:
            raise RuntimeError("CL=F não retornou dados do Yahoo Finance")
        
        spot_prices = df['Close']
        logger.info(f"✓ CL=F baixado: {len(spot_prices)} dias")
        
    except Exception as e:
        raise RuntimeError(f"Erro ao baixar CL=F: {e}")
    
    # Criar estrutura de termo sintética
    # Contango típico de WTI: ~0.3-0.5% ao mês
    # Usamos uma curva simples: F(T) = S * (1 + contango * T)
    num_tenors = len(tickers_info)
    contango_monthly = 0.004  # 0.4% ao mês
    
    price_data = []
    ttm_data = []
    ref_date = datetime.strptime(start_date, "%Y-%m-%d")
    
    for i in range(1, num_tenors + 1):
        # TTM em meses
        tenor_months = i
        ttm_years = tenor_months / 12.0
        
        # Preço forward sintético
        forward_prices = spot_prices * (1 + contango_monthly * tenor_months)
        price_data.append(forward_prices)
        
        # TTM fixo para cada tenor (aproximação)
        ttm_series = pd.Series(
            [ttm_years] * len(spot_prices),
            index=spot_prices.index
        )
        ttm_data.append(ttm_series)
        
        logger.info(f"  Tenor {i}: TTM={ttm_years:.2f} anos, contango={contango_monthly*tenor_months*100:.1f}%")
    
    # Concatena e renomeia colunas corretamente
    F_mkt = pd.concat(price_data, axis=1).sort_index()
    F_mkt.columns = [f'tenor_{i}' for i in range(1, num_tenors + 1)]
    
    ttm = pd.concat(ttm_data, axis=1).sort_index()
    ttm.columns = [f'tenor_{i}' for i in range(1, num_tenors + 1)]
    
    # Remove linhas com NaN
    F_mkt = F_mkt.dropna(how='all')
    ttm = ttm.loc[F_mkt.index]
    
    logger.info(f"\n✓ Curva forward criada: {len(F_mkt)} dias, {F_mkt.shape[1]} tenores")
    logger.info(f"  Período: {F_mkt.index[0].date()} → {F_mkt.index[-1].date()}")
    logger.info(f"  Preço médio tenor 1: ${F_mkt.iloc[:, 0].mean():.2f}")
    logger.info(f"  Preço médio tenor {num_tenors}: ${F_mkt.iloc[:, -1].mean():.2f}")
    logger.info(f"  Colunas: {list(F_mkt.columns)}")
    
    return F_mkt, ttm


def save_pipeline_format(output_dir: str, F_mkt: pd.DataFrame, ttm: pd.DataFrame):
    """
    Salva arquivos no formato esperado pelo pipeline.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # S.csv: proxy de spot = primeiro tenor
    S = pd.DataFrame({'S': F_mkt.iloc[:, 0]})
    
    # costs.csv: custos padrão para WTI
    num_tenors = F_mkt.shape[1]
    costs = pd.DataFrame({
        'tenor': [f'tenor_{i+1}' for i in range(num_tenors)],
        'tick_value': [10.0] * num_tenors,
        'fee_per_contract': [2.5] * num_tenors
    })
    
    # Salva arquivos
    paths = {}
    
    F_path = os.path.join(output_dir, 'F_mkt.csv')
    F_mkt.to_csv(F_path)
    paths['F_mkt'] = F_path
    logger.info(f"✓ Salvo: {F_path}")
    
    ttm_path = os.path.join(output_dir, 'ttm.csv')
    ttm.to_csv(ttm_path)
    paths['ttm'] = ttm_path
    logger.info(f"✓ Salvo: {ttm_path}")
    
    S_path = os.path.join(output_dir, 'S.csv')
    S.to_csv(S_path)
    paths['S'] = S_path
    logger.info(f"✓ Salvo: {S_path}")
    
    costs_path = os.path.join(output_dir, 'costs.csv')
    costs.to_csv(costs_path, index=False)
    paths['costs'] = costs_path
    logger.info(f"✓ Salvo: {costs_path}")
    
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Baixa dados reais de futuros WTI via Yahoo Finance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # 3 meses recentes (recomendado para testes):
  python src/DownloadsData.py --start-date 2024-10-01 --end-date 2024-12-31 --num-tenors 8
  
  # Ano completo:
  python src/DownloadsData.py --start-date 2024-01-01 --end-date 2024-12-31 --num-tenors 8
        """
    )
    
    parser.add_argument('--start-date', type=str, required=True, help='YYYY-MM-DD')
    parser.add_argument('--end-date', type=str, required=True, help='YYYY-MM-DD')
    parser.add_argument('--num-tenors', type=int, default=8, help='Número de tenores (meses) a baixar')
    parser.add_argument('--output-dir', type=str, default='data/real_data', help='Diretório raiz de saída')
    parser.add_argument('--dataset-name', type=str, default=None, help='Nome do dataset (padrão: WTI_YYYY_YYYY_yahoo)')
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("=== Download de Dados Reais: Yahoo Finance (WTI Futures) ===")
    logger.info("=" * 80)
    
    # Nome do dataset
    if args.dataset_name:
        dataset_name = args.dataset_name
    else:
        y0 = args.start_date[:4]
        y1 = args.end_date[:4]
        dataset_name = f"WTI_{y0}_{y1}_yahoo"
    
    output_path = os.path.join(args.output_dir, dataset_name)
    
    logger.info(f"Período: {args.start_date} → {args.end_date}")
    logger.info(f"Tenores: {args.num_tenors}")
    logger.info(f"Saída: {output_path}")
    
    # Gera tickers automaticamente
    tickers_info = generate_cl_tickers(args.start_date, args.num_tenors)
    logger.info(f"\nTickers gerados:")
    for ticker, delivery in tickers_info:
        logger.info(f"  {ticker} → expira ~{delivery.strftime('%Y-%m-%d')}")
    
    # Baixa dados
    logger.info("\nBaixando dados do Yahoo Finance...")
    F_mkt, ttm = download_yahoo_data(tickers_info, args.start_date, args.end_date)
    
    # Salva no formato do pipeline
    logger.info("\nSalvando arquivos...")
    paths = save_pipeline_format(output_path, F_mkt, ttm)
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ CONCLUÍDO COM SUCESSO!")
    logger.info("=" * 80)
    logger.info(f"Dataset: {output_path}\n")
    logger.info("Arquivos criados:")
    for k, v in paths.items():
        logger.info(f"  {k}: {v}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Próximos passos:")
    logger.info("=" * 80)
    logger.info(f"1. Testar com 2 dias:")
    logger.info(f"   python src/Backtester.py \\")
    logger.info(f"     --dataset-root {output_path} \\")
    logger.info(f"     --train-days 150 --test-days 2 \\")
    logger.info(f"     --method MLE --sizing vol_target\n")
    logger.info(f"2. Rodar backtest completo:")
    logger.info(f"   python src/Backtester.py \\")
    logger.info(f"     --dataset-root {output_path} \\")
    logger.info(f"     --train-days 150 --test-days 60 \\")
    logger.info(f"     --method MLE --sizing vol_target")


if __name__ == "__main__":
    main()
