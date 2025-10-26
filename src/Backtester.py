"""
Backtester.py

Backtesting completo da estratégia Schwartz-Smith com recalibração rolling.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

from ComputeModelForward import ComputeModelForward
from PrepareTradingInputs import PrepareTradingInputs
from TradeEngine import TradeEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def RunBacktest(
    dataset_root: str,
    train_days: int = 1500,
    test_days: int = 1500,
    output_root: str = "output",
    cfg_path: str = "config/default.yaml",
    method: str = "EM",
    sizing: str = "vol_target",
    topK: int = None
) -> dict:
    """Executa backtesting completo."""
    
    logger.info("=" * 80)
    logger.info("=== BACKTESTING SCHWARTZ-SMITH MODEL ===")
    logger.info("=" * 80)
    
    # Criar output dir
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = os.path.join(output_root, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    
    logger.info(f"\nOutput: {output_dir}")
    
    # Carregar config e dados
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['kalman']['method'] = method
    
    F_mkt, ttm, S = _load_data(dataset_root)
    T_total, M = F_mkt.shape
    
    logger.info(f"Dataset: T={T_total}, M={M}")
    logger.info(f"Treino: {train_days} | Teste: {test_days}")
    
    # Backtesting loop
    pnl_history = []
    trade_history = []
    position = np.zeros(M)
    cash = 0.0
    
    test_start = train_days
    test_end = train_days + test_days
    
    logger.info(f"\n[1/5] Executando backtest ({test_days} dias)...")
    
    import time
    times = []
    
    for t in range(test_start, test_end):
        iter_start = time.time()
        
        if (t - test_start) % 1 == 0 or (t - test_start) < 5:
            logger.info(f"  Dia {t - test_start + 1}/{test_days}")
        
        try:
            # Pipeline completo
            model_result = ComputeModelForward(
                F_mkt=F_mkt.iloc[:t+1].values,
                ttm=ttm.iloc[:t+1].values,
                S=S.iloc[:t+1].values if S is not None else None,
                cfg=cfg,
                t_idx=-1
            )
            
            trading_inputs = PrepareTradingInputs(
                F_mkt_t=F_mkt.iloc[t].values,
                F_model_t=model_result['F_model_t'],
                ttm_t=ttm.iloc[t].values,
                cost=None,
                cfg=cfg,
                F_mkt_hist=F_mkt.iloc[:t+1].values,
                F_model_hist=model_result.get('F_model_path')
            )
            
            trade_result = TradeEngine(
                mispricing=trading_inputs['mispricing'],
                Sigma=trading_inputs['Sigma'],
                limits=trading_inputs['limits'],
                thresh=trading_inputs['thresh'],
                frictions=trading_inputs['frictions'],
                method=sizing,
                topK=topK,
                w_prev=position,
                cfg=cfg
            )
            
            target_w = trade_result['target_w']
            
        except Exception as e:
            logger.warning(f"Erro dia {t}: {e}")
            target_w = position.copy()
        
        # Executar trades
        pnl_day, position, trades = _execute_trades(
            t, position, target_w,
            F_mkt.iloc[t-1].values if t > 0 else F_mkt.iloc[t].values,
            F_mkt.iloc[t].values,
            trading_inputs.get('frictions', {})
        )
        
        cash += pnl_day
        
        pnl_history.append({
            'date': F_mkt.index[t],
            't': t,
            'pnl_day': pnl_day,
            'pnl_cum': cash
        })
        
        for trade in trades:
            trade['date'] = F_mkt.index[t]
            trade['t'] = t
            trade_history.append(trade)
        
        # Medição de tempo
        iter_time = time.time() - iter_start
        times.append(iter_time)
        
        if (t - test_start) < 5 or (t - test_start) % 1 == 0:
            avg_time = np.mean(times)
            remaining_days = test_end - t - 1
            eta_minutes = (remaining_days * avg_time) / 60.0
            logger.info(f"    Tempo: {iter_time:.2f}s | Média: {avg_time:.2f}s | ETA: {eta_minutes:.2f} min")
    
    df_pnl = pd.DataFrame(pnl_history)
    df_trades = pd.DataFrame(trade_history) if trade_history else pd.DataFrame()
    
    logger.info(f"\n[2/5] Calculando benchmarks...")
    benchmarks = _calculate_benchmarks(F_mkt, S, train_days, test_days)
    
    logger.info(f"\n[3/5] Calculando métricas...")
    metrics = _calculate_metrics(df_pnl, df_trades, benchmarks)
    
    logger.info(f"\n[4/5] Salvando resultados...")
    _save_results(output_dir, df_pnl, df_trades, metrics, benchmarks)
    
    logger.info(f"\n[5/5] Gerando gráficos...")
    _generate_plots(output_dir, F_mkt, df_pnl, df_trades, benchmarks, train_days, test_days, M)
    
    logger.info("\n" + "=" * 80)
    logger.info(f"P&L Total: ${metrics['strategy']['total_pnl']:.2f}")
    logger.info(f"Sharpe: {metrics['strategy']['sharpe']:.3f}")
    logger.info(f"Output: {output_dir}")
    
    return {'pnl': df_pnl, 'trades': df_trades, 'metrics': metrics, 'output_dir': output_dir}


def _load_data(dataset_root):
    F_mkt = pd.read_csv(f"{dataset_root}/F_mkt.csv", index_col=0, parse_dates=True)
    ttm = pd.read_csv(f"{dataset_root}/ttm.csv", index_col=0, parse_dates=True)
    S_path = f"{dataset_root}/S.csv"
    S = pd.read_csv(S_path, index_col=0, parse_dates=True) if os.path.exists(S_path) else None
    return F_mkt, ttm, S


def _execute_trades(t, position, target_w, F_prev, F_curr, frictions):
    price_change = F_curr - F_prev
    mtm_pnl = np.sum(position * price_change)
    delta_w = target_w - position
    
    fee = frictions.get('fee', np.ones(len(position)) * 2.0)
    transaction_costs = 0.0
    trades = []
    
    for i in range(len(position)):
        if np.abs(delta_w[i]) > 1e-6:
            cost = np.abs(delta_w[i]) * (fee[i] if hasattr(fee, '__len__') else fee)
            transaction_costs += cost
            trades.append({'tenor': i, 'side': 'BUY' if delta_w[i] > 0 else 'SELL',
                          'quantity': np.abs(delta_w[i]), 'price': F_curr[i], 'cost': cost})
    
    pnl_day = mtm_pnl - transaction_costs
    return pnl_day, target_w.copy(), trades


def _calculate_benchmarks(F_mkt, S, train_days, test_days):
    test_start = train_days
    test_end = train_days + test_days
    F_test = F_mkt.iloc[test_start:test_end]
    
    benchmarks = {}
    
    # Buy & Hold M1
    m1_returns = F_test.iloc[:, 0].pct_change().fillna(0)
    benchmarks['Buy&Hold M1'] = {
        'pnl_daily': m1_returns * 100,
        'pnl_cum': ((1 + m1_returns).cumprod() - 1) * 100
    }
    
    # CDI 13.75%
    cdi_rate = 0.1375 / 252
    benchmarks['CDI (13.75%)'] = {
        'pnl_daily': pd.Series([cdi_rate * 100] * len(F_test), index=F_test.index),
        'pnl_cum': pd.Series(((1 + cdi_rate) ** np.arange(1, len(F_test) + 1) - 1) * 100, index=F_test.index)
    }
    
    # Roll Strategy
    roll_returns = F_test.iloc[:, :2].mean(axis=1).pct_change().fillna(0)
    benchmarks['Roll Strategy'] = {
        'pnl_daily': roll_returns * 100,
        'pnl_cum': ((1 + roll_returns).cumprod() - 1) * 100
    }
    
    return benchmarks


def _calculate_metrics(df_pnl, df_trades, benchmarks):
    returns = df_pnl['pnl_day'].values
    pnl_cum = df_pnl['pnl_cum'].values
    
    def sharpe(rets):
        if len(rets) == 0 or rets.std() == 0:
            return 0.0
        return (rets.mean() - 0.1375/252) / rets.std() * np.sqrt(252)
    
    def max_dd(cum):
        if len(cum) == 0:
            return 0.0
        running_max = np.maximum.accumulate(cum)
        dd = (cum - running_max) / (running_max + 1e-10)
        return dd.min()
    
    strategy_metrics = {
        'total_pnl': pnl_cum[-1] if len(pnl_cum) > 0 else 0,
        'sharpe': sharpe(returns),
        'max_drawdown': max_dd(pnl_cum),
        'win_rate': (returns > 0).sum() / len(returns) if len(returns) > 0 else 0,
        'num_trades': len(df_trades)
    }
    
    benchmark_metrics = {}
    for name, bench in benchmarks.items():
        rets = bench['pnl_daily'].values
        cum = bench['pnl_cum'].values
        benchmark_metrics[name] = {
            'total_pnl': cum[-1] if len(cum) > 0 else 0,
            'sharpe': sharpe(rets),
            'max_drawdown': max_dd(cum)
        }
    
    return {'strategy': strategy_metrics, 'benchmarks': benchmark_metrics}


def _save_results(output_dir, df_pnl, df_trades, metrics, benchmarks):
    df_pnl.to_csv(f"{output_dir}/pnl_daily.csv", index=False)
    if len(df_trades) > 0:
        df_trades.to_csv(f"{output_dir}/trades.csv", index=False)
    
    metrics_df = pd.DataFrame({
        'Metric': ['Total P&L', 'Sharpe', 'Max DD', 'Win Rate', 'Trades'],
        'Strategy': [f"${metrics['strategy']['total_pnl']:.2f}",
                    f"{metrics['strategy']['sharpe']:.3f}",
                    f"{metrics['strategy']['max_drawdown']:.2%}",
                    f"{metrics['strategy']['win_rate']:.2%}",
                    str(metrics['strategy']['num_trades'])]
    })
    
    for name in benchmarks:
        m = metrics['benchmarks'][name]
        metrics_df[name] = [f"${m['total_pnl']:.2f}", f"{m['sharpe']:.3f}",
                           f"{m['max_drawdown']:.2%}", '-', '-']
    
    metrics_df.to_csv(f"{output_dir}/metrics.csv", index=False)


def _generate_plots(output_dir, F_mkt, df_pnl, df_trades, benchmarks, train_days, test_days, M):
    images_dir = f"{output_dir}/images"
    
    # Gráfico principal: P&L
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(df_pnl['t'] - train_days, df_pnl['pnl_cum'], linewidth=3, label='Model', color='#2E86AB')
    
    colors = ['#A23B72', '#F18F01', '#06A77D']
    for i, (name, bench) in enumerate(benchmarks.items()):
        ax.plot(range(len(bench['pnl_cum'])), bench['pnl_cum'], linewidth=2.5,
               label=name, color=colors[i], linestyle='--', alpha=0.8)
    
    ax.axhline(0, color='black', linewidth=1, linestyle=':', alpha=0.5)
    ax.set_xlabel('Dias', fontsize=14, fontweight='bold')
    ax.set_ylabel('P&L ($)', fontsize=14, fontweight='bold')
    ax.set_title('Backtest: Estratégia vs Benchmarks', fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{images_dir}/00_pnl_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Gráficos por tenor
    test_start = train_days
    test_end = train_days + test_days
    
    for tenor_idx in range(M):
        fig, ax = plt.subplots(figsize=(16, 7))
        time_axis = np.arange(len(F_mkt))
        ax.plot(time_axis, F_mkt.iloc[:, tenor_idx], linewidth=1.5, color='#34495e', alpha=0.6)
        
        ax.axvspan(test_start, test_end, alpha=0.1, color='green', label='Teste')
        ax.axvspan(0, test_start, alpha=0.05, color='gray', label='Treino')
        
        if len(df_trades) > 0:
            trades_tenor = df_trades[df_trades['tenor'] == tenor_idx]
            for _, trade in trades_tenor.iterrows():
                t = trade['t']
                color = 'green' if trade['side'] == 'BUY' else 'red'
                marker = '^' if trade['side'] == 'BUY' else 'v'
                ax.scatter(t, F_mkt.iloc[t, tenor_idx], s=150, marker=marker,
                          color=color, edgecolors='black', linewidth=2, zorder=5)
        
        ax.axvline(test_start, color='black', linewidth=2, linestyle='--', alpha=0.7)
        ax.set_xlabel('Tempo (dias)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Preço ($)', fontsize=13, fontweight='bold')
        ax.set_title(f'Sinais - Tenor {tenor_idx+1}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{images_dir}/{tenor_idx+1:02d}_tenor_{tenor_idx+1}_signals.png", dpi=150)
        plt.close()
    
    logger.info(f"  Gráficos salvos em {images_dir}")


def main():
    parser = argparse.ArgumentParser(description='Backtesting Schwartz-Smith')
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--train-days', type=int, default=1500)
    parser.add_argument('--test-days', type=int, default=1500)
    parser.add_argument('--method', default='EM', choices=['MLE', 'EM'])
    parser.add_argument('--sizing', default='vol_target')
    
    args = parser.parse_args()
    
    RunBacktest(
        dataset_root=args.dataset_root,
        train_days=args.train_days,
        test_days=args.test_days,
        method=args.method,
        sizing=args.sizing
    )


if __name__ == '__main__':
    main()
