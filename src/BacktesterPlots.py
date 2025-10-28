"""
BacktesterPlots.py

Gera gráficos e análises para resultados de backtesting já salvos.
Lê o output mais recente (ou especificado) e cria visualizações.

Uso:
    python src/BacktesterPlots.py                    # Usa output mais recente
    python src/BacktesterPlots.py --output-dir path  # Usa output específico
"""

import os
import sys
import glob
import argparse
import logging
from typing import Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def GeneratePlots(output_dir: str) -> None:
    """
    Gera todos os gráficos para um resultado de backtesting salvo.
    
    Parâmetros
    ----------
    output_dir : str
        Diretório com resultados salvos (pnl_daily.csv, trades.csv, etc.)
    """
    logger.info("=" * 80)
    logger.info("=== Gerando Gráficos de Backtesting ===")
    logger.info("=" * 80)
    logger.info(f"Output dir: {output_dir}")
    
    # Carregar dados
    logger.info("\n[1/4] Carregando dados...")
    data = _load_backtest_data(output_dir)
    
    # Criar diretório de imagens
    img_dir = os.path.join(output_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)
    
    # Gerar gráficos
    logger.info("\n[2/4] Gerando gráfico de P&L comparativo...")
    _plot_pnl_comparison(data, img_dir)
    
    logger.info("\n[3/4] Gerando gráficos de sinais por tenor...")
    _plot_signals_by_tenor(data, img_dir)
    
    logger.info("\n[4/4] Gerando gráfico de evolução do erro...")
    _plot_prediction_error_evolution(data, img_dir)
    
    logger.info("\n" + "=" * 80)
    logger.info(f"✓ Gráficos salvos em: {img_dir}")
    logger.info("=" * 80)


def _find_latest_output() -> str:
    """Encontra o diretório de output mais recente."""
    output_base = "output"
    if not os.path.exists(output_base):
        raise FileNotFoundError(f"Diretório {output_base} não encontrado")
    
    dirs = glob.glob(f"{output_base}/202*")  # Padrão: 2025-10-26_19-01-51
    if not dirs:
        raise FileNotFoundError(f"Nenhum resultado encontrado em {output_base}")
    
    # Ordenar por data/hora no nome
    latest = sorted(dirs)[-1]
    logger.info(f"Output mais recente encontrado: {latest}")
    return latest


def _load_backtest_data(output_dir: str) -> Dict:
    """Carrega todos os dados do backtesting."""
    data = {}
    
    # Arquivos obrigatórios
    pnl_path = os.path.join(output_dir, 'pnl_daily.csv')
    if not os.path.exists(pnl_path):
        raise FileNotFoundError(f"Arquivo pnl_daily.csv não encontrado em {output_dir}")
    
    data['pnl'] = pd.read_csv(pnl_path)
    data['pnl']['date'] = pd.to_datetime(data['pnl']['date'])
    logger.info(f"  ✓ P&L: {len(data['pnl'])} dias")
    
    # Trades
    trades_path = os.path.join(output_dir, 'trades.csv')
    if os.path.exists(trades_path):
        data['trades'] = pd.read_csv(trades_path)
        data['trades']['date'] = pd.to_datetime(data['trades']['date'])
        logger.info(f"  ✓ Trades: {len(data['trades'])} operações")
    else:
        data['trades'] = pd.DataFrame()
        logger.warning("  ⚠ Arquivo trades.csv não encontrado")
    
    # Métricas
    metrics_path = os.path.join(output_dir, 'metrics.csv')
    if os.path.exists(metrics_path):
        df_metrics = pd.read_csv(metrics_path)
        # Converter CSV para estrutura dict
        data['metrics'] = {
            'strategy': {
                'total_pnl': float(df_metrics.loc[df_metrics['Metric'] == 'Total P&L', 'Strategy'].values[0].replace('$', '')),
                'sharpe': float(df_metrics.loc[df_metrics['Metric'] == 'Sharpe', 'Strategy'].values[0]),
                'max_drawdown': float(df_metrics.loc[df_metrics['Metric'] == 'Max DD', 'Strategy'].values[0].replace('%', '')) / 100,
                'win_rate': float(df_metrics.loc[df_metrics['Metric'] == 'Win Rate', 'Strategy'].values[0].replace('%', '')) / 100 if 'Win Rate' in df_metrics['Metric'].values else 0,
                'num_trades': int(df_metrics.loc[df_metrics['Metric'] == 'Trades', 'Strategy'].values[0]) if 'Trades' in df_metrics['Metric'].values else 0
            },
            'benchmarks': {}
        }
        # Adicionar benchmarks
        for col in df_metrics.columns[2:]:  # Pular 'Metric' e 'Strategy'
            try:
                data['metrics']['benchmarks'][col] = {
                    'total_pnl': float(df_metrics.loc[df_metrics['Metric'] == 'Total P&L', col].values[0].replace('$', '')),
                    'sharpe': float(df_metrics.loc[df_metrics['Metric'] == 'Sharpe', col].values[0]),
                    'max_drawdown': float(df_metrics.loc[df_metrics['Metric'] == 'Max DD', col].values[0].replace('%', '')) / 100
                }
            except:
                pass  # Pular se houver erro
        logger.info(f"  ✓ Métricas carregadas")
    
    # Previsões do modelo (se existir)
    predictions_path = os.path.join(output_dir, 'model_predictions.csv')
    if os.path.exists(predictions_path):
        data['predictions'] = pd.read_csv(predictions_path)
        data['predictions']['date'] = pd.to_datetime(data['predictions']['date'])
        logger.info(f"  ✓ Previsões: {len(data['predictions'])} dias")
    else:
        data['predictions'] = None
        logger.warning("  ⚠ Arquivo model_predictions.csv não encontrado")
    
    # Market data (se existir)
    market_path = os.path.join(output_dir, 'market_data.csv')
    if os.path.exists(market_path):
        data['market'] = pd.read_csv(market_path, index_col=0, parse_dates=True)
        logger.info(f"  ✓ Market data: {len(data['market'])} dias")
    else:
        data['market'] = None
        logger.warning("  ⚠ Arquivo market_data.csv não encontrado")
    
    return data


def _plot_pnl_comparison(data: Dict, img_dir: str) -> None:
    """Gráfico de P&L comparativo com CDI e tabela de métricas."""
    df_pnl = data['pnl']
    metrics = data['metrics']
    
    # Criar figura com layout customizado
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[4, 1.2, 0.8], width_ratios=[2, 1], 
                          hspace=0.35, wspace=0.25)
    ax_main = fig.add_subplot(gs[0, :])
    ax_table = fig.add_subplot(gs[1, :])
    ax_legend = fig.add_subplot(gs[2, :])
    
    # ===== GRÁFICO PRINCIPAL =====
    # P&L da estratégia
    ax_main.plot(df_pnl['date'], df_pnl['pnl_cum'], 
                linewidth=4.5, label='Estratégia (Modelo Schwartz-Smith)', 
                color='#2E86AB', zorder=5, marker='o', markersize=3, markevery=10)
    
    # CDI 13.75% (taxa Selic atual)
    num_days = len(df_pnl)
    cdi_daily_rate = 0.1375 / 252
    cdi_pnl = [100 * ((1 + cdi_daily_rate)**i - 1) for i in range(num_days)]
    ax_main.plot(df_pnl['date'], cdi_pnl, 
                linewidth=3.5, label='CDI (13.75% a.a.)', 
                color='#27AE60', linestyle='-.', alpha=0.85, zorder=4)
    
    # Linha zero
    ax_main.axhline(0, color='black', linewidth=2, linestyle=':', alpha=0.6, zorder=1)
    
    ax_main.set_xlabel('Data', fontsize=15, fontweight='bold')
    ax_main.set_ylabel('P&L Acumulado ($)', fontsize=15, fontweight='bold')
    ax_main.set_title('Performance: Estratégia vs CDI (Benchmark Livre de Risco)', 
                     fontsize=18, fontweight='bold', pad=25)
    ax_main.legend(loc='upper left', fontsize=13, framealpha=0.95, 
                  shadow=True, fancybox=True, edgecolor='black')
    ax_main.grid(True, alpha=0.35, linestyle='--', linewidth=0.8)
    
    # Formatar eixo x
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax_main.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax_main.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=11)
    plt.setp(ax_main.yaxis.get_majorticklabels(), fontsize=11)
    
    # ===== TABELA DE MÉTRICAS =====
    strat = metrics['strategy']
    cdi_final = cdi_pnl[-1]
    
    # Calcular retorno anualizado
    days = len(df_pnl)
    strat_return_pct = (strat['total_pnl'] / 100) * 100  # % do capital inicial
    strat_annual = ((1 + strat_return_pct/100) ** (252/days) - 1) * 100
    cdi_annual = 13.75
    
    table_data = [
        ['', 'P&L Total', 'Retorno (%)', 'Anualizado', 'Sharpe', 'Max DD', 'Win Rate', 'Trades'],
        ['Estratégia SS', 
         f"${strat['total_pnl']:.2f}",
         f"{strat_return_pct:.2f}%",
         f"{strat_annual:.2f}%",
         f"{strat['sharpe']:.2f}", 
         f"{strat['max_drawdown']:.1%}",
         f"{strat['win_rate']:.1%}", 
         f"{strat['num_trades']}"],
        ['CDI (13.75%)', 
         f"${cdi_final:.2f}", 
         f"{cdi_final:.2f}%",
         f"{cdi_annual:.2f}%",
         "N/A",
         "0.0%", 
         "-", "-"]
    ]
    
    # Adicionar benchmarks se existirem
    bench_names = list(metrics['benchmarks'].keys())
    for name in bench_names[:3]:  # Máximo 3 benchmarks
        if name in metrics['benchmarks']:
            m = metrics['benchmarks'][name]
            bench_return_pct = (m['total_pnl'] / 100) * 100
            bench_annual = ((1 + bench_return_pct/100) ** (252/days) - 1) * 100
            table_data.append([
                name, 
                f"${m['total_pnl']:.2f}",
                f"{bench_return_pct:.2f}%",
                f"{bench_annual:.2f}%",
                f"{m['sharpe']:.2f}",
                f"{m['max_drawdown']:.1%}", 
                "-", "-"
            ])
    
    ax_table.axis('tight')
    ax_table.axis('off')
    
    table = ax_table.table(cellText=table_data, cellLoc='center',
                          bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Estilizar tabela
    for i, cell in enumerate(table.get_celld().values()):
        if i < 8:  # Header
            cell.set_facecolor('#34495E')
            cell.set_text_props(weight='bold', color='white', fontsize=11)
        else:
            row = i // 8
            cell.set_facecolor('#ECF0F1' if row % 2 == 0 else 'white')
            cell.set_text_props(fontsize=10)
    
    # ===== LEGENDA EXPLICATIVA =====
    ax_legend.axis('off')
    legend_text = (
        "\n📈 ESTRATÉGIA: Modelo Schwartz-Smith com recalibração rolling (janela de treino) + sinais baseados em mispricing entre modelo e mercado"
        "\n\n💵 CDI: Taxa livre de risco (13.75% a.a.) - Benchmark padrão do mercado brasileiro"
        "\n\n📉 BENCHMARKS: Estratégias passivas Buy & Hold (comprar e segurar) em diferentes tenores de futuros"
    )
    ax_legend.text(0.02, 0.98, legend_text, transform=ax_legend.transAxes,
                  fontsize=11, verticalalignment='top', horizontalalignment='left',
                  bbox=dict(boxstyle='round', facecolor='#FFFACD', alpha=0.6, 
                           edgecolor='gray', linewidth=1.5),
                  wrap=True)
    
    plt.savefig(os.path.join(img_dir, '00_pnl_comparison.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  ✓ 00_pnl_comparison.png (com CDI, métricas e explicações)")


def _plot_signals_by_tenor(data: Dict, img_dir: str) -> None:
    """Gráficos de sinais por tenor com previsões do modelo e preços de mercado."""
    df_trades = data['trades']
    df_predictions = data.get('predictions')
    df_market = data.get('market')
    
    if len(df_trades) == 0:
        logger.warning("  ⚠ Sem trades para plotar sinais")
        return
    
    # Agrupar por tenor
    num_tenors = df_trades['tenor'].nunique()
    
    for tenor in range(num_tenors):
        df_tenor = df_trades[df_trades['tenor'] == tenor].copy()
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Plotar linha de mercado
        if df_market is not None:
            market_col = f'tenor_{tenor+1}'
            if market_col in df_market.columns:
                ax.plot(df_market.index, df_market[market_col], 
                       linewidth=2.5, color='#2C3E50', label='Mercado (Real)', 
                       alpha=0.8, zorder=1)
        
        # Plotar linha de previsão do modelo
        if df_predictions is not None and len(df_predictions) > 0:
            model_col = f'F_model_{tenor}'
            if model_col in df_predictions.columns:
                ax.plot(df_predictions['date'], df_predictions[model_col], 
                       linewidth=2.5, color='#E74C3C', label='Modelo (Previsão)', 
                       alpha=0.7, linestyle='--', zorder=2)
        
        # Separar BUY e SELL
        if len(df_tenor) > 0:
            df_buy = df_tenor[df_tenor['side'] == 'BUY']
            df_sell = df_tenor[df_tenor['side'] == 'SELL']
            
            if len(df_buy) > 0:
                ax.scatter(df_buy['date'], df_buy['price'], 
                          color='#27AE60', marker='^', s=150, 
                          label='BUY', alpha=0.9, edgecolors='darkgreen', 
                          linewidth=2, zorder=5)
            
            if len(df_sell) > 0:
                ax.scatter(df_sell['date'], df_sell['price'], 
                          color='#C0392B', marker='v', s=150, 
                          label='SELL', alpha=0.9, edgecolors='darkred', 
                          linewidth=2, zorder=5)
        
        ax.set_xlabel('Data', fontsize=13, fontweight='bold')
        ax.set_ylabel('Preço ($)', fontsize=13, fontweight='bold')
        ax.set_title(f'Tenor {tenor+1}: Previsão do Modelo vs Mercado Real', 
                    fontsize=15, fontweight='bold', pad=15)
        ax.legend(loc='best', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Formatar eixo x
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(img_dir, f'{tenor+1:02d}_tenor_{tenor+1}_signals.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    logger.info(f"  ✓ Sinais por tenor: {num_tenors} gráficos")


def _plot_prediction_error_evolution(data: Dict, img_dir: str) -> None:
    """
    Gráfico de evolução do erro de previsão ao longo do tempo.
    
    Para cada dia de previsão, calcula o erro em D+1, D+2, D+3, etc.
    e faz a média para ver como o erro evolui.
    """
    predictions = data.get('predictions')
    market = data.get('market')
    
    if predictions is None or market is None:
        logger.warning("  ⚠ Dados de previsões ou market não disponíveis")
        logger.warning("     Execute Backtester.py novamente para gerar esses dados")
        return
    
    # Calcular erros para cada horizonte
    max_horizon = 30  # Até 30 dias à frente
    num_tenors = len([col for col in predictions.columns if col.startswith('F_model_')])
    
    errors_by_horizon = {h: [] for h in range(1, max_horizon + 1)}
    
    for idx in range(len(predictions)):
        pred_date = predictions.iloc[idx]['date']
        
        # Buscar índice no market data
        if pred_date not in market.index:
            continue
        
        market_idx = market.index.get_loc(pred_date)
        
        for horizon in range(1, max_horizon + 1):
            future_idx = market_idx + horizon
            
            if future_idx >= len(market):
                break
            
            # Calcular erro médio entre todos os tenores
            errors = []
            for t in range(num_tenors):
                pred_col = f'F_model_{t}'
                market_col = f'tenor_{t+1}'
                
                if pred_col in predictions.columns and market_col in market.columns:
                    pred_val = predictions.iloc[idx][pred_col]
                    market_val = market.iloc[future_idx][market_col]
                    
                    if not np.isnan(pred_val) and not np.isnan(market_val):
                        error = np.abs(pred_val - market_val)
                        errors.append(error)
            
            if errors:
                errors_by_horizon[horizon].append(np.mean(errors))
    
    # Calcular média e desvio padrão por horizonte
    horizons = []
    mean_errors = []
    std_errors = []
    
    for h in range(1, max_horizon + 1):
        if errors_by_horizon[h]:
            horizons.append(h)
            mean_errors.append(np.mean(errors_by_horizon[h]))
            std_errors.append(np.std(errors_by_horizon[h]))
    
    if not horizons:
        logger.warning("  ⚠ Não foi possível calcular erros")
        return
    
    # Plotar
    fig, ax = plt.subplots(figsize=(12, 6))
    
    horizons = np.array(horizons)
    mean_errors = np.array(mean_errors)
    std_errors = np.array(std_errors)
    
    ax.plot(horizons, mean_errors, 'o-', linewidth=2, markersize=6, 
            color='#E63946', label='Erro Médio')
    
    # Banda de confiança (±1 std)
    ax.fill_between(horizons, 
                     mean_errors - std_errors, 
                     mean_errors + std_errors,
                     alpha=0.3, color='#E63946', label='±1 Std Dev')
    
    ax.set_xlabel('Horizonte (dias após previsão)', fontsize=12)
    ax.set_ylabel('Erro Médio Absoluto ($)', fontsize=12)
    ax.set_title('Evolução do Erro de Previsão ao Longo do Tempo', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Adicionar linha de tendência
    z = np.polyfit(horizons, mean_errors, 1)
    p = np.poly1d(z)
    ax.plot(horizons, p(horizons), "--", alpha=0.5, color='gray', 
            label=f'Tendência (slope={z[0]:.3f})')
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, '99_prediction_error_evolution.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info("  ✓ 99_prediction_error_evolution.png")
    logger.info(f"     Erro D+1: ${mean_errors[0]:.2f} | Erro D+30: ${mean_errors[-1]:.2f}")


def main():
    """CLI para geração de gráficos."""
    parser = argparse.ArgumentParser(
        description='Gera gráficos para resultados de backtesting salvos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Usa output mais recente
  python src/BacktesterPlots.py
  
  # Usa output específico
  python src/BacktesterPlots.py --output-dir output/2025-10-26_19-01-51
        """
    )
    
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Diretório com resultados (padrão: mais recente)')
    
    args = parser.parse_args()
    
    try:
        # Determinar output dir
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = _find_latest_output()
        
        if not os.path.exists(output_dir):
            raise FileNotFoundError(f"Diretório não encontrado: {output_dir}")
        
        # Gerar plots
        GeneratePlots(output_dir)
        
        print("\n" + "="*80)
        print("✓ SUCESSO! Gráficos gerados")
        print("="*80)
        print(f"\nVisualize os gráficos em:")
        print(f"  {output_dir}/images/")
        
    except Exception as e:
        logger.error(f"\nERRO: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
