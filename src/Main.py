"""
Main.py

Script simples para orquestração manual/rápida durante desenvolvimento.

Permite executar o pipeline completo chamando as 3 funções públicas em sequência
e inspecionar resultados rapidamente.

Uso:
    python src/Main.py
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
import logging

# Adicionar diretório src ao path
sys.path.insert(0, os.path.dirname(__file__))

from ComputeModelForward import ComputeModelForward
from PrepareTradingInputs import PrepareTradingInputs
from TradeEngine import TradeEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    Execução rápida do pipeline em um dataset escolhido.
    """
    logger.info("=" * 70)
    logger.info("=== Main.py - Execução Manual do Pipeline ===")
    logger.info("=" * 70)
    
    # Configurações
    dataset_root = "data/fakedata/wti_synth_01"  # Ajustar conforme necessário
    cfg_path = "config/default.yaml"
    t_idx = -1  # Último período
    
    # Verificar se dataset existe
    if not os.path.exists(dataset_root):
        logger.error(f"Dataset não encontrado: {dataset_root}")
        logger.info("Execute primeiro: python src/GenerateFakeData.py --dataset-name wti_synth_01 --T 1500 --M 8")
        return
    
    # Carregar configuração
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    logger.info(f"Dataset: {dataset_root}")
    logger.info(f"Config: {cfg_path}")
    logger.info(f"t_idx: {t_idx}")
    
    # ==============================
    # 1. Carregar Dados
    # ==============================
    logger.info("\n[1/3] Carregando dados...")
    
    F_mkt = pd.read_csv(os.path.join(dataset_root, 'F_mkt.csv'), 
                        index_col=0, parse_dates=True)
    ttm = pd.read_csv(os.path.join(dataset_root, 'ttm.csv'), 
                      index_col=0, parse_dates=True)
    
    S = None
    S_path = os.path.join(dataset_root, 'S.csv')
    if os.path.exists(S_path):
        S = pd.read_csv(S_path, index_col=0, parse_dates=True)['S']
    
    T, M = F_mkt.shape
    logger.info(f"Dados carregados: T={T}, M={M}")
    
    # ==============================
    # 2. ComputeModelForward
    # ==============================
    logger.info("\n[2/3] Executando ComputeModelForward...")
    
    model_result = ComputeModelForward(
        F_mkt=F_mkt,
        ttm=ttm,
        S=S,
        cfg=cfg,
        t_idx=t_idx
    )
    
    Theta = model_result['Theta']
    state_t = model_result['state_t']
    F_model_t = model_result['F_model_t']
    
    logger.info(f"\nParâmetros estimados (Theta):")
    for key, value in Theta.items():
        logger.info(f"  {key}: {value:.6f}")
    
    logger.info(f"\nEstado em t={t_idx}:")
    logger.info(f"  X_hat: {state_t[0]:.6f}")
    logger.info(f"  Y_hat: {state_t[1]:.6f}")
    
    logger.info(f"\nF_model_t: {F_model_t}")
    
    # ==============================
    # 3. PrepareTradingInputs
    # ==============================
    logger.info("\n[3/3] Executando PrepareTradingInputs...")
    
    F_mkt_t = F_mkt.iloc[t_idx, :].values
    ttm_t = ttm.iloc[t_idx, :].values
    
    trading_inputs = PrepareTradingInputs(
        F_mkt_t=F_mkt_t,
        F_model_t=F_model_t,
        ttm_t=ttm_t,
        cost=None,
        cfg=cfg,
        F_mkt_hist=F_mkt.values,
        F_model_hist=model_result.get('F_model_path')
    )
    
    mispricing = trading_inputs['mispricing']
    Sigma = trading_inputs['Sigma']
    limits = trading_inputs['limits']
    thresh = trading_inputs['thresh']
    frictions = trading_inputs['frictions']
    
    logger.info(f"\nMispricing: {mispricing}")
    logger.info(f"Sigma shape: {Sigma.shape}")
    logger.info(f"Limits: {limits}")
    
    # ==============================
    # 4. TradeEngine
    # ==============================
    logger.info("\n[4/3] Executando TradeEngine...")
    
    trade_result = TradeEngine(
        mispricing=mispricing,
        Sigma=Sigma,
        limits=limits,
        thresh=thresh,
        frictions=frictions,
        method=cfg.get('sizing', {}).get('method', 'vol_target'),
        topK=cfg.get('thresh', {}).get('topK'),
        cfg=cfg
    )
    
    signals = trade_result['signals']
    target_w = trade_result['target_w']
    orders = trade_result['orders']
    
    logger.info(f"\nSignals: {signals}")
    logger.info(f"Target weights: {target_w}")
    logger.info(f"\nOrdens geradas ({len(orders)}):")
    for order in orders:
        logger.info(f"  Tenor {order[0]}: {order[1]} {order[2]:.4f}")
    
    # ==============================
    # Resumo Final
    # ==============================
    logger.info("\n" + "=" * 70)
    logger.info("=== PIPELINE EXECUTADO COM SUCESSO ===")
    logger.info("=" * 70)
    logger.info(f"\nResumo:")
    logger.info(f"  - Parâmetros calibrados: kappa={Theta['kappa']:.4f}, "
                f"sigma_X={Theta['sigma_X']:.4f}, sigma_Y={Theta['sigma_Y']:.4f}")
    logger.info(f"  - Estado final: X={state_t[0]:.4f}, Y={state_t[1]:.4f}")
    logger.info(f"  - Mispricing médio: {np.mean(mispricing):.4f}")
    logger.info(f"  - Sinais ativos: BUY={np.sum(signals==1)}, "
                f"SELL={np.sum(signals==-1)}, HOLD={np.sum(signals==0)}")
    logger.info(f"  - Ordens para executar: {len(orders)}")
    
    logger.info("\nPara gerar gráficos e análise completa, execute:")
    logger.info(f"  python src/TestingTheoryPipeline.py --dataset-root {dataset_root}")


if __name__ == '__main__':
    main()
