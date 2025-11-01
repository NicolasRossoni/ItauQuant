#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DownloadsData_Free.py

Coleta dados GRATUITOS para o pipeline:
- Fonte A: Nasdaq Data Link (Quandl) CHRIS (CL1..CLM)  [--source chris]
- Fonte B: Yahoo Finance contratos específicos (ex.: CLZ25.NYM, CLF26.NYM)  [--source yahoo]

Saída (contrato do projeto):
  data/free_data/<dataset_name>/
    F_mkt.csv    (datas x tenores)
    ttm.csv      (datas x tenores)  # aproximação: i/12 ano (CHRIS) ou calc. aprox. por ticker (Yahoo)
    S.csv        (proxy de spot = coluna tenor_1)
    costs.csv    (tabela simples por tenor)

Uso (CHRIS contínuos, 8 tenores):
  python src/DownloadsData_Free.py --source chris --start-date 2020-01-01 --end-date 2024-12-31 --num-tenors 8

Uso (Yahoo, contratos que você escolher):
  python src/DownloadsData_Free.py --source yahoo --start-date 2020-01-01 --end-date 2024-12-31 \
      --yahoo-tickers CLZ25.NYM,CLF26.NYM,CLG26.NYM,CLH26.NYM,CLK26.NYM,CLM26.NYM,CLN26.NYM,CLQ26.NYM

Obs.:
- Para CHRIS, configure a chave (grátis) no ambiente OU .env: NASDAQ_DATA_LINK_API_KEY=SUACHAVE
- Para Yahoo, não precisa de chave.
"""

import os
import argparse
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Pacotes opcionais (carregados on-demand)
try:
    import nasdaqdatalink as ndl  # para CHRIS
except Exception:
    ndl = None

try:
    import yfinance as yf          # para Yahoo
except Exception:
    yf = None

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
def _setup_logging(level: str = "INFO") -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Utilidades
# -----------------------------------------------------------------------------
MONTH_CODE = {
    'F': 1,  'G': 2,  'H': 3,  'J': 4,  'K': 5,  'M': 6,
    'N': 7,  'Q': 8,  'U': 9,  'V': 10, 'X': 11, 'Z': 12
}

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _save_contract_csvs(out_dir: str,
                        F_mkt: pd.DataFrame,
                        ttm: pd.DataFrame,
                        S: pd.DataFrame,
                        costs: pd.DataFrame) -> Dict[str, str]:
    paths = {}
    p = os.path.join(out_dir, "F_mkt.csv"); F_mkt.to_csv(p); paths['F_mkt'] = p; logger.info(f"Salvo: {p}")
    p = os.path.join(out_dir, "ttm.csv"); ttm.to_csv(p); paths['ttm'] = p; logger.info(f"Salvo: {p}")
    p = os.path.join(out_dir, "S.csv"); S.to_csv(p); paths['S'] = p; logger.info(f"Salvo: {p}")
    p = os.path.join(out_dir, "costs.csv"); costs.to_csv(p, index=False); paths['costs'] = p; logger.info(f"Salvo: {p}")
    return paths

def _default_costs(num_tenors: int, tick_value: float = 10.0, fee: float = 2.5) -> pd.DataFrame:
    return pd.DataFrame({
        "tenor": [f"tenor_{i+1}" for i in range(num_tenors)],
        "tick_value": [tick_value] * num_tenors,
        "fee_per_contract": [fee] * num_tenors
    })

# -----------------------------------------------------------------------------
# Fonte A — CHRIS (Nasdaq Data Link / Quandl)  CL1..CLM
# -----------------------------------------------------------------------------
def fetch_chris_cl(start_date: str, end_date: str, M: int, api_key: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retorna F_mkt (datas x M) usando 'Settle' de CHRIS/CME_CL{i} e ttm aproximado (i/12).
    """
    if ndl is None:
        raise RuntimeError("nasdaqdatalink não instalado. Faça: pip install nasdaqdatalink")

    if api_key:
        ndl.ApiConfig.api_key = api_key

    frames = []
    for i in range(1, M + 1):
        code = f"CHRIS/CME_CL{i}"
        df = ndl.get(code, start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            raise RuntimeError(f"Sem dados para {code}. Verifique sua chave e as datas.")
        # Coluna padrão de settlement costuma ser 'Settle'
        col = None
        # Tenta algumas variações comuns
        for c in ["Settle", "Settle Price", "Last", "Previous Day Open Interest"]:
            if c in df.columns:
                col = c
                break
        if col is None:
            raise RuntimeError(f"{code}: não encontrei coluna de preço de fechamento/settlement.")
        s = df[col].rename(f"tenor_{i}")
        frames.append(s)

    F_mkt = pd.concat(frames, axis=1).sort_index()
    # Garante intervalo de datas comuns
    F_mkt = F_mkt.dropna(how="all")

    # ttm aproximado: i/12 ano (constante por tenor)
    idx = F_mkt.index
    ttm = pd.DataFrame(
        data=np.tile(np.array([i/12.0 for i in range(1, M + 1)]), (len(idx), 1)),
        index=idx,
        columns=[f"tenor_{i}" for i in range(1, M + 1)]
    )

    return F_mkt, ttm

# -----------------------------------------------------------------------------
# Fonte B — Yahoo Finance (tickers explicitados)
# -----------------------------------------------------------------------------
def _parse_yahoo_month_code(ticker: str) -> Optional[Tuple[int, int]]:
    """
    Tenta ler MM/YY de um ticker tipo 'CLZ25.NYM' -> (12, 2025).
    Retorna None se não reconhecer.
    """
    base = os.path.basename(ticker).split('.')[0]  # 'CLZ25'
    if len(base) < 4:
        return None
    # acha a primeira letra de mês válida
    m_letter = None
    for ch in base[::-1]:  # varre de trás pra frente e pega a primeira letra que é código de mês
        if ch in MONTH_CODE:
            m_letter = ch
            break
    if m_letter is None:
        return None
    # ano = 2 dígitos logo após a letra do mês (heurística simples)
    pos = base.rfind(m_letter)
    yy = base[pos+1:pos+3]
    if len(yy) != 2 or not yy.isdigit():
        return None
    year = 2000 + int(yy) if int(yy) < 80 else 1900 + int(yy)  # 00-79 -> 2000-2079
    month = MONTH_CODE[m_letter]
    return (month, year)

def fetch_yahoo_contracts(tickers: List[str], start_date: str, end_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Baixa contratos indicados (ex.: ['CLZ25.NYM','CLF26.NYM',...]) e retorna:
      - F_mkt (datas x M) com Close
      - ttm (datas x M) aproximado via data-contrato (dia 15 do mês de entrega)
    """
    if yf is None:
        raise RuntimeError("yfinance não instalado. Faça: pip install yfinance")

    data_cols = []
    ttm_cols = []

    idx_union = None
    meta_contracts = []

    for i, tk in enumerate(tickers, start=1):
        logger.info(f"Yahoo: baixando {tk}")
        df = yf.download(tk, start=start_date, end=end_date, interval="1d", progress=False, auto_adjust=False)
        if df is None or df.empty:
            logger.warning(f"Sem dados para {tk}; pulando.")
            continue
        # usa 'Close' (Yahoo nem sempre tem 'Adj Close' útil para futuros)
        s = df["Close"].rename(f"tenor_{i}")
        data_cols.append(s)
        idx_union = s.index if idx_union is None else idx_union.union(s.index)
        meta_contracts.append(tk)

    if not data_cols:
        raise RuntimeError("Nenhum ticker Yahoo trouxe dados. Verifique a lista de --yahoo-tickers.")

    # Alinha por índice união
    F_mkt = pd.concat([s.reindex(idx_union) for s in data_cols], axis=1)

    # ttm aproximado: diferença até o dia 15 do mês/ano inferido do ticker
    ttm_frames = []
    for i, tk in enumerate(meta_contracts, start=1):
        my = _parse_yahoo_month_code(tk)
        if my is None:
            # fallback: usa i/12 se não conseguir inferir
            arr = np.full(shape=(len(idx_union),), fill_value=i/12.0, dtype=float)
            ttm_frames.append(pd.Series(arr, index=idx_union, name=f"tenor_{i}"))
        else:
            m, y = my
            # usa dia 15 como proxy de “meio do mês de entrega”
            delivery = pd.Timestamp(year=y, month=m, day=15)
            # TTM em anos (ACT/365)
            delta_days = (delivery - idx_union).days.values
            arr = np.maximum(delta_days, 1) / 365.0  # não deixa zero/negativo
            ttm_frames.append(pd.Series(arr, index=idx_union, name=f"tenor_{i}"))

    ttm = pd.concat(ttm_frames, axis=1)

    return F_mkt.sort_index(), ttm.sort_index()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Baixa dados gratuitos (CHRIS ou Yahoo) no formato do pipeline.")
    parser.add_argument("--source", choices=["chris", "yahoo"], required=True, help="Fonte dos dados gratuitos")
    parser.add_argument("--start-date", required=True, type=str, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, type=str, help="YYYY-MM-DD")
    parser.add_argument("--num-tenors", type=int, default=8, help="(CHRIS) nº de tenores CL1..CLM (default=8)")
    parser.add_argument("--yahoo-tickers", type=str, default="", help="(Yahoo) lista separada por vírgula de tickers, ex: CLZ25.NYM,CLF26.NYM,...")
    parser.add_argument("--dataset-name", type=str, default=None, help="Nome do dataset (default: CL_<YYYY>_<YYYY>_free)")
    parser.add_argument("--output-root", type=str, default="data/free_data", help="Diretório raiz de saída")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    _setup_logging(args.log_level)
    logger.info("Iniciando…")

    start_date = args.start_date
    end_date = args.end_date

    if args.dataset_name:
        dataset = args.dataset_name
    else:
        dataset = f"CL_{start_date[:4]}_{end_date[:4]}_free"

    out_dir = os.path.join(args.output_root, dataset)
    _ensure_dir(out_dir)

    if args.source == "chris":
        api_key = os.getenv("NASDAQ_DATA_LINK_API_KEY", "")
        if not api_key:
            logger.warning("NASDAQ_DATA_LINK_API_KEY não encontrado; tentar sem chave pode falhar/limitar.")
        logger.info(f"CHRIS: coletando CL1..CL{args.num_tenors}")
        F_mkt, ttm = fetch_chris_cl(start_date, end_date, args.num_tenors, api_key)

    elif args.source == "yahoo":
        if not args.yahoo_tickers.strip():
            logger.info("Sem --yahoo-tickers; usando apenas CL=F como proxy de spot/curto prazo.")
            tickers = ["CL=F"]
        else:
            tickers = [t.strip() for t in args.yahoo_tickers.split(",") if t.strip()]
        F_mkt, ttm = fetch_yahoo_contracts(tickers, start_date, end_date)

    # S proxy = primeira coluna (tenor_1)
    S = pd.DataFrame({"S": F_mkt.iloc[:, 0]})
    # Custos defaults
    costs = _default_costs(num_tenors=F_mkt.shape[1])

    paths = _save_contract_csvs(out_dir, F_mkt, ttm, S, costs)

    logger.info("\nConcluído!")
    logger.info(f"Arquivos: {out_dir}")
    for k, v in paths.items():
        logger.info(f"  {k}: {v}")
    logger.info("\nPronto para rodar o pipeline/TestingTheoryPipeline.py")

if __name__ == "__main__":
    main()
