"""
TestingTheoryPipeline.py

Executa o pipeline completo de ponta a ponta com dados sintéticos:
1. Carrega dados (F_mkt, ttm, S)
2. Executa ComputeModelForward
3. Executa PrepareTradingInputs
4. Executa TradeEngine
5. Salva artefatos (CSVs)
6. Gera e salva 10 gráficos

Função pública: RunTheoryPipeline(...)
Script CLI: python src/TestingTheoryPipeline.py --dataset-root data/fakedata/wti_synth_01
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import yaml
import argparse
from typing import Dict

# Importar módulos do projeto
import sys
sys.path.insert(0, os.path.dirname(__file__))

from ComputeModelForward import ComputeModelForward
from PrepareTradingInputs import PrepareTradingInputs
from TradeEngine import TradeEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurar matplotlib para não mostrar gráficos
plt.ioff()


def RunTheoryPipeline(
    dataset_root: str,
    t_idx: int = -1,
    save_dir: str = "images/fake_data_analysis",
    method: str = "MLE",
    sizing: str = "vol_target",
    topK: int = None,
    cfg_path: str = "config/default.yaml"
) -> dict:
    """
    Executa o pipeline completo de teoria em dados sintéticos.
    
    Parâmetros
    ----------
    dataset_root : str
        Caminho para o diretório do dataset (ex: "data/fakedata/wti_synth_01").
    t_idx : int
        Índice temporal alvo (ex: -1 para último).
    save_dir : str
        Diretório para salvar gráficos.
    method : str
        "MLE" ou "EM" para Kalman.
    sizing : str
        "vol_target" ou "qp".
    topK : int, opcional
        Número máximo de tenores a operar.
    cfg_path : str
        Caminho para arquivo de configuração YAML.
    
    Retorna
    -------
    dict
        Artefatos e caminhos salvos.
    
    Exemplos
    --------
    >>> result = RunTheoryPipeline("data/fakedata/wti_synth_01")
    >>> print(result.keys())
    """
    logger.info("=" * 70)
    logger.info("=== Executando Pipeline Completo de Teoria ===")
    logger.info("=" * 70)
    
    # Carregar configuração
    cfg = _load_config(cfg_path)
    cfg['method'] = method
    cfg['sizing']['method'] = sizing
    if topK is not None:
        cfg['thresh']['topK'] = topK
    
    # 1. Carregar dados
    logger.info("\n[1/6] Carregando dados...")
    F_mkt, ttm, S = _load_data(dataset_root)
    T, M = F_mkt.shape
    logger.info(f"Dados carregados: T={T}, M={M}")
    
    # 2. Executar ComputeModelForward
    logger.info("\n[2/6] Executando ComputeModelForward...")
    model_result = ComputeModelForward(F_mkt, ttm, S, cfg, t_idx)
    Theta = model_result['Theta']
    state_t = model_result['state_t']
    F_model_t = model_result['F_model_t']
    state_path = model_result['state_path']
    F_model_path = model_result['F_model_path']
    
    logger.info(f"Theta estimado: {Theta}")
    logger.info(f"Estado em t={t_idx}: X={state_t[0]:.4f}, Y={state_t[1]:.4f}")
    
    # 3. Executar PrepareTradingInputs
    logger.info("\n[3/6] Executando PrepareTradingInputs...")
    F_mkt_t = F_mkt[t_idx, :] if isinstance(F_mkt, np.ndarray) else F_mkt.iloc[t_idx, :].values
    ttm_t = ttm[t_idx, :] if isinstance(ttm, np.ndarray) else ttm.iloc[t_idx, :].values
    
    trading_inputs = PrepareTradingInputs(
        F_mkt_t, F_model_t, ttm_t, None, cfg,
        F_mkt_hist=F_mkt if isinstance(F_mkt, np.ndarray) else F_mkt.values,
        F_model_hist=F_model_path
    )
    
    mispricing = trading_inputs['mispricing']
    Sigma = trading_inputs['Sigma']
    limits = trading_inputs['limits']
    thresh = trading_inputs['thresh']
    frictions = trading_inputs['frictions']
    
    # 4. Executar TradeEngine
    logger.info("\n[4/6] Executando TradeEngine...")
    trade_result = TradeEngine(
        mispricing, Sigma, limits, thresh, frictions,
        method=sizing, topK=topK, cfg=cfg
    )
    
    signals = trade_result['signals']
    target_w = trade_result['target_w']
    orders = trade_result['orders']
    z_scores = trade_result.get('z_scores', np.zeros(M))
    
    logger.info(f"Ordens geradas: {len(orders)}")
    
    # 5. Salvar artefatos
    logger.info("\n[5/6] Salvando artefatos...")
    os.makedirs(save_dir, exist_ok=True)
    
    artifacts = _save_artifacts(
        save_dir, t_idx, orders, signals, target_w, mispricing, z_scores
    )
    
    # 6. Gerar e salvar gráficos
    logger.info("\n[6/6] Gerando 10 gráficos...")
    plot_paths = _generate_plots(
        save_dir, t_idx, F_mkt, ttm, F_model_t, F_model_path,
        state_path, mispricing, z_scores, Sigma, signals, target_w, M
    )
    
    logger.info("\n" + "=" * 70)
    logger.info("=== Pipeline concluído com sucesso! ===")
    logger.info("=" * 70)
    
    result = {
        'Theta': Theta,
        'state_t': state_t,
        'F_model_t': F_model_t,
        'artifacts': artifacts,
        'plots': plot_paths
    }
    
    return result


def _load_config(cfg_path: str) -> dict:
    """Carrega arquivo de configuração YAML."""
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Configuração carregada de: {cfg_path}")
    return cfg


def _load_data(dataset_root: str) -> tuple:
    """
    Carrega F_mkt.csv, ttm.csv, S.csv.
    
    Retorna
    -------
    F_mkt, ttm, S (DataFrames ou arrays)
    """
    F_mkt_path = os.path.join(dataset_root, 'F_mkt.csv')
    ttm_path = os.path.join(dataset_root, 'ttm.csv')
    S_path = os.path.join(dataset_root, 'S.csv')
    
    F_mkt = pd.read_csv(F_mkt_path, index_col=0, parse_dates=True)
    ttm = pd.read_csv(ttm_path, index_col=0, parse_dates=True)
    
    S = None
    if os.path.exists(S_path):
        S = pd.read_csv(S_path, index_col=0, parse_dates=True)['S']
    
    logger.info(f"F_mkt: {F_mkt.shape}")
    logger.info(f"ttm: {ttm.shape}")
    if S is not None:
        logger.info(f"S: {S.shape}")
    
    return F_mkt, ttm, S


def _save_artifacts(
    save_dir: str,
    t_idx: int,
    orders: list,
    signals: np.ndarray,
    target_w: np.ndarray,
    mispricing: np.ndarray,
    z_scores: np.ndarray
) -> dict:
    """Salva artefatos em CSVs."""
    
    # orders_t.csv
    orders_df = pd.DataFrame(orders, columns=['maturity_idx', 'side', 'qty'])
    orders_path = os.path.join(save_dir, f'orders_t{t_idx}.csv')
    orders_df.to_csv(orders_path, index=False)
    logger.info(f"Salvo: {orders_path}")
    
    # signals_t.csv
    signals_df = pd.DataFrame({'signals': signals})
    signals_path = os.path.join(save_dir, f'signals_t{t_idx}.csv')
    signals_df.to_csv(signals_path, index=False)
    logger.info(f"Salvo: {signals_path}")
    
    # target_w_t.csv
    target_w_df = pd.DataFrame({'target_w': target_w})
    target_w_path = os.path.join(save_dir, f'target_w_t{t_idx}.csv')
    target_w_df.to_csv(target_w_path, index=False)
    logger.info(f"Salvo: {target_w_path}")
    
    # mispricing_t.csv
    mispricing_df = pd.DataFrame({'mispricing': mispricing, 'z_scores': z_scores})
    mispricing_path = os.path.join(save_dir, f'mispricing_t{t_idx}.csv')
    mispricing_df.to_csv(mispricing_path, index=False)
    logger.info(f"Salvo: {mispricing_path}")
    
    return {
        'orders': orders_path,
        'signals': signals_path,
        'target_w': target_w_path,
        'mispricing': mispricing_path
    }


def _generate_plots(
    save_dir: str,
    t_idx: int,
    F_mkt: pd.DataFrame,
    ttm: pd.DataFrame,
    F_model_t: np.ndarray,
    F_model_path: np.ndarray,
    state_path: np.ndarray,
    mispricing: np.ndarray,
    z_scores: np.ndarray,
    Sigma: np.ndarray,
    signals: np.ndarray,
    target_w: np.ndarray,
    M: int
) -> dict:
    """
    Gera gráficos de análise por tenor.
    
    Para cada tenor, cria um gráfico mostrando:
    - Série temporal de F_model vs F_mkt
    - Marcações visuais de ordens BUY/SELL
    - Estatísticas relevantes (mispricing, z-score, sinal)
    
    Além disso, gera gráficos globais:
    - Covariance heatmap
    - State trajectories
    - Forward surface mispricing
    """
    
    plot_paths = {}
    
    F_mkt_arr = F_mkt.values if isinstance(F_mkt, pd.DataFrame) else F_mkt
    ttm_arr = ttm.values if isinstance(ttm, pd.DataFrame) else ttm
    F_mkt_t = F_mkt_arr[t_idx, :]
    
    tenor_labels = [f'Tenor_{i+1}' for i in range(M)]
    T_total = len(F_mkt_arr)
    time_axis = np.arange(T_total)
    t_actual = t_idx if t_idx >= 0 else T_total + t_idx
    
    # =================================================================
    # GRÁFICOS POR TENOR: F_model vs F_mkt + Sinais de Trading
    # =================================================================
    logger.info(f"Gerando {M} gráficos por tenor...")
    
    for i in range(M):
        tenor_label = tenor_labels[i]
        
        # Criar figura
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Plot F_model e F_mkt ao longo do tempo
        ax.plot(time_axis, F_model_path[:, i], label='F_model', 
                linewidth=2.5, color='#2E86AB', alpha=0.9)
        ax.plot(time_axis, F_mkt_arr[:, i], label='F_mkt', 
                linewidth=2.5, color='#A23B72', alpha=0.9, linestyle='--')
        
        # Marcar ponto atual t*
        ax.scatter([t_actual], [F_model_path[t_actual, i]], 
                   s=150, c='#2E86AB', marker='o', edgecolors='black', 
                   linewidth=2, zorder=5, label=f'F_model em t*')
        ax.scatter([t_actual], [F_mkt_arr[t_actual, i]], 
                   s=150, c='#A23B72', marker='s', edgecolors='black', 
                   linewidth=2, zorder=5, label=f'F_mkt em t*')
        
        # Adicionar área sombreada para mostrar spread
        ax.fill_between(time_axis, F_model_path[:, i], F_mkt_arr[:, i], 
                        alpha=0.2, color='gray', label='Spread')
        
        # Marcar sinais de trading
        signal_at_t = signals[i]
        if signal_at_t == 1:  # BUY
            ax.axvline(t_actual, color='green', linewidth=3, alpha=0.7, 
                      linestyle=':', label='SINAL: BUY')
            ax.text(t_actual, ax.get_ylim()[1]*0.95, 'BUY ▲', 
                   fontsize=14, fontweight='bold', color='green',
                   ha='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        elif signal_at_t == -1:  # SELL
            ax.axvline(t_actual, color='red', linewidth=3, alpha=0.7, 
                      linestyle=':', label='SINAL: SELL')
            ax.text(t_actual, ax.get_ylim()[1]*0.95, 'SELL ▼', 
                   fontsize=14, fontweight='bold', color='red',
                   ha='center', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        else:  # HOLD
            ax.axvline(t_actual, color='gray', linewidth=2, alpha=0.5, 
                      linestyle=':', label='SINAL: HOLD')
        
        # Estatísticas no gráfico
        stats_text = f"Mispricing: {mispricing[i]:.3f}\nZ-score: {z_scores[i]:.2f}\nTarget Weight: {target_w[i]:.2f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Configurações visuais
        ax.set_xlabel('Tempo (dias)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Forward Price (USD)', fontsize=13, fontweight='bold')
        ax.set_title(f'Forward Model vs Market - {tenor_label}\n(t* = {t_actual})', 
                    fontsize=15, fontweight='bold', pad=15)
        ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Salvar
        path = os.path.join(save_dir, f'{i+1:02d}_tenor_{i+1}_timeseries.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        plot_paths[f'tenor_{i+1}'] = path
        logger.info(f"Gráfico {i+1}/{M} salvo: {path}")
    
    # Gráficos globais adicionais
    logger.info("\nGerando gráficos globais...")
    
    # Estados latentes X_t e Y_t
    if state_path is not None:
        plt.figure(figsize=(14, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(time_axis, state_path[:, 0], linewidth=2, color='#06A77D')
        plt.ylabel('X_t (curto prazo)', fontsize=12, fontweight='bold')
        plt.title('Trajetórias dos Estados Latentes', fontsize=15, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.axvline(t_actual, color='red', linestyle='--', linewidth=2, 
                   label=f't*={t_actual}', alpha=0.7)
        plt.legend(fontsize=10)
        
        plt.subplot(2, 1, 2)
        plt.plot(time_axis, state_path[:, 1], linewidth=2, color='#F18F01')
        plt.xlabel('Tempo (dias)', fontsize=12, fontweight='bold')
        plt.ylabel('Y_t (longo prazo)', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.axvline(t_actual, color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        plt.tight_layout()
        path = os.path.join(save_dir, f'{M+1:02d}_state_trajectories.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        plot_paths['states'] = path
        logger.info(f"Gráfico estados salvos: {path}")
    
    # Matriz de covariância
    plt.figure(figsize=(10, 8))
    sns.heatmap(Sigma, annot=True, fmt='.4f', cmap='coolwarm', center=0,
                xticklabels=tenor_labels, yticklabels=tenor_labels,
                cbar_kws={'label': 'Covariance'}, linewidths=0.5)
    plt.title('Matriz de Covariância (Σ)', fontsize=15, fontweight='bold', pad=15)
    plt.tight_layout()
    path = os.path.join(save_dir, f'{M+2:02d}_covariance_heatmap.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    plot_paths['covariance'] = path
    logger.info(f"Gráfico covariância salvo: {path}")
    
    logger.info(f"\nTodos os {M+2} gráficos salvos em: {save_dir}")
    return plot_paths


def main():
    """CLI para execução do pipeline."""
    parser = argparse.ArgumentParser(description='Executa pipeline completo de teoria')
    parser.add_argument('--dataset-root', type=str, required=True,
                        help='Caminho para dataset (ex: data/fakedata/wti_synth_01)')
    parser.add_argument('--t-idx', type=int, default=-1,
                        help='Índice temporal alvo')
    parser.add_argument('--save-dir', type=str, default='images/fake_data_analysis',
                        help='Diretório para salvar gráficos')
    parser.add_argument('--method', type=str, default='MLE',
                        choices=['MLE', 'EM'],
                        help='Método de estimação Kalman')
    parser.add_argument('--sizing', type=str, default='vol_target',
                        choices=['vol_target', 'qp'],
                        help='Método de dimensionamento')
    parser.add_argument('--topK', type=int, default=None,
                        help='Número máximo de tenores a operar')
    parser.add_argument('--cfg-path', type=str, default='config/default.yaml',
                        help='Caminho para configuração YAML')
    
    args = parser.parse_args()
    
    result = RunTheoryPipeline(
        dataset_root=args.dataset_root,
        t_idx=args.t_idx,
        save_dir=args.save_dir,
        method=args.method,
        sizing=args.sizing,
        topK=args.topK,
        cfg_path=args.cfg_path
    )
    
    print("\n" + "=" * 70)
    print("=== PIPELINE CONCLUÍDO COM SUCESSO ===")
    print("=" * 70)
    print(f"\nTheta estimado: {result['Theta']}")
    print(f"Estado final: X={result['state_t'][0]:.4f}, Y={result['state_t'][1]:.4f}")
    print(f"\nArtefatos salvos: {result['artifacts']}")
    print(f"\nGráficos salvos em: {args.save_dir}")


if __name__ == '__main__':
    main()
