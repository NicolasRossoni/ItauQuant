"""
backtest.py

Script principal de backtesting do modelo Schwartz-Smith.
Este é um FLUXOGRAMA LIMPO que abstrai toda complexidade nos módulos src/.

CONFIGURAÇÕES (início do arquivo - modificar aqui):
"""

# ==========================================
# CONFIGURAÇÕES - MODIFICAR AQUI
# ==========================================

DATASET_ID = "WTI_test_380d"             # Nome/ID da pasta em data/raw/ que vamos usar (mesmo do download)
TRAIN_WINDOW_DAYS = 150                 # Janela de treinamento em dias úteis (150 dias para calibração)
TEST_DAYS = 150                         # Dias de teste (150 dias com janela crescente: 150+150=300)

# Configurações do modelo (opcional - deixar padrão se não souber)
MODEL_METHOD = "MLE"                     # Método de calibração: "MLE" ou "EM"
SIZING_METHOD = "vol_target"             # Método de dimensionamento: "vol_target" ou "qp"
ROLLING_RECALIBRATION = False           # Se deve recalibrar a cada dia (False = calibra apenas 1x)
EXPANDING_WINDOW = True                # Se deve usar janela crescente (True = adiciona 1 dia a cada iteração)

# ==========================================
# FLUXOGRAMA PRINCIPAL
# ==========================================

import sys
import os
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
from src.DataManipulation import (
    load_data_from_raw, 
    save_data_to_processed,
    format_for_model,
    load_config
)
from src.Model import ComputeModelForward
from src.TradingStrategy import PrepareTradingInputs, TradeEngine

def main():
    """
    Fluxograma principal de backtesting.
    
    Input: Configurações definidas no início do arquivo
    Output:
    - Print minimalista no terminal mostrando o status do código
    - Arquivos salvos em data/processed/{DATASET_ID}/ com:
      * portfolio_performance.csv - evolução diária do valor da carteira
      * trades_log.csv - log de todas as operações executadas
      * model_evolution.csv - evolução dos parâmetros do modelo ao longo do tempo
    """
    
    # Iniciar cronômetro
    start_time = time.time()
    
    print("=" * 80)
    print("📈 BACKTESTING MODELO SCHWARTZ-SMITH - ITAU QUANT")
    print("=" * 80)
    print(f"🔧 CONFIGURAÇÕES DETECTADAS:")
    print(f"   • Dataset ID: {DATASET_ID}")
    print(f"   • Janela de treino: {TRAIN_WINDOW_DAYS} dias úteis")
    print(f"   • Dias de teste: {TEST_DAYS} dias")
    print(f"   • Método de calibração: {MODEL_METHOD}")
    print(f"   • Sizing method: {SIZING_METHOD}")
    print(f"   • Recalibração rolante: {'Sim' if ROLLING_RECALIBRATION else 'Não (apenas 1x)'}")
    print()
    
    # PASSO 1: Carregar dados e configurações
    print("📥 PASSO 1: CARREGANDO DADOS E CONFIGURAÇÕES")
    print(f"   🔄 Procurando dataset {DATASET_ID} em data/raw/...")
    
    try:
        # Carregar dataset
        raw_data = load_data_from_raw(DATASET_ID)
        model_data = format_for_model(raw_data)
        
        # Carregar configurações
        config = load_config("config/default.yaml")
        config['method'] = MODEL_METHOD
        config['sizing']['method'] = SIZING_METHOD
        
        # Diagnósticos detalhados dos dados carregados
        dates = raw_data['dates']
        total_days = len(dates)
        tenores = raw_data['tenors'] 
        F_mkt = model_data['F_mkt']
        
        print("✅ PASSO 1: Dados carregados com sucesso!")
        print(f"   📊 Período disponível: {dates[0].date()} → {dates[-1].date()}")
        print(f"   📅 Total de dias úteis: {total_days}")
        print(f"   🎯 Tenores disponíveis: {len(tenores)} ({', '.join(tenores)})")
        print(f"   💰 Faixa de preços: ${F_mkt.min().min():.2f} - ${F_mkt.max().max():.2f}")
        
        # Verificação de suficiência dos dados
        if total_days < TRAIN_WINDOW_DAYS + TEST_DAYS:
            required_days = TRAIN_WINDOW_DAYS + TEST_DAYS
            print(f"   ❌ ERRO: Dados insuficientes!")
            print(f"      • Disponível: {total_days} dias")
            print(f"      • Necessário: {required_days} dias ({TRAIN_WINDOW_DAYS} treino + {TEST_DAYS} teste)")
            sys.exit(1)
        else:
            available_for_test = total_days - TRAIN_WINDOW_DAYS
            print(f"   ✅ Dados suficientes: {available_for_test} dias disponíveis para teste")
        
        # Configurações do modelo carregadas
        print(f"   🔧 Configurações do modelo:")
        print(f"      • Método Kalman: {config['method']}")
        print(f"      • Vol target: {config['sizing']['vol_target']:.1%}")
        print(f"      • Z-in/out: {config['thresh']['z_in']}/{config['thresh']['z_out']}")
        print(f"      • TopK: {config['thresh']['topK']}")
        
    except Exception as e:
        print(f"❌ ERRO CRÍTICO no PASSO 1: {e}")
        print(f"   🔍 Tipo do erro: {type(e).__name__}")
        if "FileNotFoundError" in str(type(e)):
            print(f"   💡 SOLUÇÃO: Execute primeiro 'python Code/download.py'")
        sys.exit(1)
    
    # PASSO 2: Preparar período de teste
    print()
    print("🎯 PASSO 2: DEFININDO JANELAS DE TREINO E TESTE")
    print(f"   🔄 Configurando janela de {TRAIN_WINDOW_DAYS} dias de treino + {TEST_DAYS} dias de teste...")
    
    try:
        # Definir janelas baseado nos dados disponíveis
        all_dates = dates  # Usando dates já definida no PASSO 1
        total_available = len(all_dates)
        
        # Janela de treino: últimos TRAIN_WINDOW_DAYS dias (excluindo período de teste)
        train_end_idx = total_available - TEST_DAYS
        train_start_idx = train_end_idx - TRAIN_WINDOW_DAYS
        
        if train_start_idx < 0:
            raise ValueError(f"Dados insuficientes: precisa de {TRAIN_WINDOW_DAYS + TEST_DAYS} dias, mas só há {total_available}")
        
        # Datas de treino e teste
        train_dates = all_dates[train_start_idx:train_end_idx]
        test_dates = all_dates[train_end_idx:train_end_idx + TEST_DAYS]
        
        print("✅ PASSO 2: Janelas definidas com sucesso!")
        print(f"   📚 Período de treino: {train_dates[0].date()} → {train_dates[-1].date()} ({len(train_dates)} dias)")
        print(f"   🧪 Período de teste: {test_dates[0].date()} → {test_dates[-1].date()} ({len(test_dates)} dias)")
        
        # Validação das janelas
        gap_days = (test_dates[0] - train_dates[-1]).days
        if gap_days != 1:
            print(f"   ⚠️  ALERTA: Gap de {gap_days} dias entre treino e teste")
        else:
            print(f"   ✅ Janelas contíguas: sem gaps entre treino e teste")
        
        # Estatísticas do período de treino
        F_train = F_mkt[train_start_idx:train_end_idx]
        train_returns = F_train.pct_change().dropna()
        train_vol = train_returns.std().mean() * np.sqrt(252)  # Anualizada
        print(f"   📊 Estatísticas do treino:")
        print(f"      • Volatilidade média anual: {train_vol:.1%}")
        print(f"      • Preço inicial: ${F_train.iloc[0, 0]:.2f}")
        print(f"      • Preço final: ${F_train.iloc[-1, 0]:.2f}")
        
    except Exception as e:
        print(f"❌ ERRO CRÍTICO no PASSO 2: {e}")
        print(f"   🔍 Tipo do erro: {type(e).__name__}")
        sys.exit(1)
    
    # PASSO 3: Executar backtesting dia por dia
    print()
    print("🔄 PASSO 3: EXECUTANDO BACKTESTING DIA-A-DIA")
    print(f"   🎯 Estratégia: {TEST_DAYS} dias de teste com janela de treino de {TRAIN_WINDOW_DAYS} dias")
    calibration_mode = "Recalibração rolante" if ROLLING_RECALIBRATION else ("Janela crescente" if EXPANDING_WINDOW else "Calibração única")
    print(f"   🔧 Modo: {calibration_mode}")
    print()
    
    # Inicializar estruturas para salvar resultados
    daily_results = {}
    portfolio_performance = []
    trades_log = []
    model_evolution = []
    
    # Posições anteriores para continuidade
    w_prev = None
    portfolio_value = 100000.0  # Valor inicial da carteira ($100k)
    calibrated_params = None    # Parâmetros calibrados (reutilizar se não rolante)
    
    print(f"💰 CONFIGURAÇÕES DE EXECUÇÃO:")
    print(f"   • Valor inicial da carteira: ${portfolio_value:,.0f}")
    print(f"   • Método de calibração: {MODEL_METHOD}")
    calibration_description = "A cada dia (rolante)" if ROLLING_RECALIBRATION else ("A cada dia (crescente)" if EXPANDING_WINDOW else "Apenas uma vez")
    print(f"   • Recalibração: {calibration_description}")
    print()
    
    for i, current_date in enumerate(test_dates):
        date_str = current_date.strftime('%Y-%m-%d')
        day_of_week = current_date.strftime('%A')[:3]  # Seg, Ter, etc
        
        # Progress report detalhado
        progress = (i + 1) / len(test_dates) * 100
        print(f"   📅 PROCESSANDO DIA {i+1}/{len(test_dates)}: {date_str} ({day_of_week}) - {progress:.1f}%")
        
        try:
            # Determinar índices para este dia de teste
            current_test_idx = train_end_idx + i  # Posição no dataset completo
            current_F_mkt_t = F_mkt.iloc[current_test_idx]
            current_ttm_t = model_data['ttm'].iloc[current_test_idx]
            
            print(f"      🔍 Dados do dia: preço_spot=${current_F_mkt_t.iloc[0]:.2f}, tenores={len(current_F_mkt_t)}")
            
            # Calibrar modelo (sempre calibra se janela crescente, só uma vez se fixa)
            should_recalibrate = (calibrated_params is None or ROLLING_RECALIBRATION or EXPANDING_WINDOW)
            
            if should_recalibrate:
                recalibration_type = "inicial" if calibrated_params is None else ("rolante" if ROLLING_RECALIBRATION else "crescente")
                print(f"      🧠 Calibrando modelo Schwartz-Smith ({MODEL_METHOD}) - modo {recalibration_type}...")
                
                # Dados de treino baseado no tipo de janela
                if EXPANDING_WINDOW:
                    # Janela crescente: do início até o dia atual (dia 1: 150 dias, dia 2: 151 dias, etc.)
                    actual_train_start = train_start_idx
                    actual_train_end = train_end_idx + i  # Adiciona 1 dia a cada iteração
                    window_size = actual_train_end - actual_train_start
                elif ROLLING_RECALIBRATION:
                    # Janela rolante: últimos TRAIN_WINDOW_DAYS antes do dia atual
                    actual_train_end = current_test_idx
                    actual_train_start = max(0, actual_train_end - TRAIN_WINDOW_DAYS)
                    window_size = actual_train_end - actual_train_start
                else:
                    # Janela fixa: período de treino definido inicialmente (150 dias sempre)
                    actual_train_start = train_start_idx
                    actual_train_end = train_end_idx
                    window_size = TRAIN_WINDOW_DAYS
                
                F_mkt_train = F_mkt.iloc[actual_train_start:actual_train_end]
                ttm_train = model_data['ttm'].iloc[actual_train_start:actual_train_end]
                S_train = model_data['S'].iloc[actual_train_start:actual_train_end] if model_data.get('S') is not None else None
                
                print(f"         📊 Janela de treino: {len(F_mkt_train)} dias ({F_mkt_train.index[0].date()} → {F_mkt_train.index[-1].date()})")
                
                # Log de progresso temporal
                elapsed_time = time.time() - start_time
                elapsed_minutes = elapsed_time / 60
                days_completed = i  # dias já processados
                if days_completed > 0:
                    avg_time_per_day = elapsed_time / days_completed
                    remaining_days = TEST_DAYS - days_completed
                    estimated_remaining_time = (avg_time_per_day * remaining_days) / 60
                else:
                    estimated_remaining_time = 0
                
                print(f"         ⏱️  Tempo decorrido: {elapsed_minutes:.1f}min | Tempo estimado restante: {estimated_remaining_time:.1f}min")
                
                # SUBSTEP 3.1: Calibrar modelo
                model_result = ComputeModelForward(
                    F_mkt=F_mkt_train,
                    ttm=ttm_train,
                    S=S_train,
                    cfg=config,
                    t_idx=-1  # Último dia dos dados de treino
                )
                
                # Salvar parâmetros calibrados
                calibrated_params = model_result['Theta']
                print(f"         ✅ Calibração concluída!")
                print(f"            • kappa={calibrated_params.get('kappa', 0):.3f}")
                print(f"            • sigma_X={calibrated_params.get('sigma_X', 0):.3f}")
                print(f"            • sigma_Y={calibrated_params.get('sigma_Y', 0):.3f}")
                print(f"            • rho={calibrated_params.get('rho', 0):.3f}")
                
            else:
                print(f"      ♻️  Reutilizando parâmetros calibrados (kappa={calibrated_params.get('kappa', 0):.3f})")
                # Ainda precisamos gerar F_model_t para o dia atual mesmo reutilizando parâmetros
                print(f"         🔄 Gerando predições com parâmetros reutilizados...")
                
                # Log de progresso temporal
                elapsed_time = time.time() - start_time
                elapsed_minutes = elapsed_time / 60
                days_completed = i  # dias já processados
                if days_completed > 0:
                    avg_time_per_day = elapsed_time / days_completed
                    remaining_days = TEST_DAYS - days_completed
                    estimated_remaining_time = (avg_time_per_day * remaining_days) / 60
                else:
                    estimated_remaining_time = 0
                
                print(f"         ⏱️  Tempo decorrido: {elapsed_minutes:.1f}min | Tempo estimado restante: {estimated_remaining_time:.1f}min")
                
                # Usar dados de treino atualizados para predição
                model_result = ComputeModelForward(
                    F_mkt=F_mkt_train,
                    ttm=ttm_train,
                    S=S_train,
                    cfg=config,
                    t_idx=-1  # Último dia dos dados de treino
                )
                # Manter os parâmetros calibrados originais
                model_result['Theta'] = calibrated_params
            
            # SUBSTEP 3.2: Gerar predições para o dia atual
            print(f"      🔮 Gerando predições do modelo para {date_str}...")
            F_mkt_t = current_F_mkt_t
            ttm_t = current_ttm_t
            
            # SUBSTEP 3.3: Preparar inputs de trading
            print(f"      📊 Preparando sinais de trading...")
            # Converter DataFrames para arrays numpy
            F_mkt_hist_array = F_mkt_train.values if 'F_mkt_train' in locals() else None
            F_model_hist_array = None
            if model_result.get('F_model_path') is not None:
                F_model_hist_array = model_result['F_model_path'].values if hasattr(model_result['F_model_path'], 'values') else model_result['F_model_path']
            
            trading_inputs = PrepareTradingInputs(
                F_mkt_t=F_mkt_t.values if hasattr(F_mkt_t, 'values') else F_mkt_t,
                F_model_t=model_result['F_model_t'] if model_result['F_model_t'] is not None else F_mkt_t.values,  # Fallback
                ttm_t=ttm_t.values if hasattr(ttm_t, 'values') else ttm_t,
                cost=raw_data.get('costs'),
                cfg=config,
                F_mkt_hist=F_mkt_hist_array,
                F_model_hist=F_model_hist_array
            )
            
            mispricing = trading_inputs['mispricing']
            print(f"         💹 Mispricing detectado: min={mispricing.min():.3f}, max={mispricing.max():.3f}, mean={mispricing.mean():.3f}")
            
            # SUBSTEP 3.4: Gerar decisões de trading
            print(f"      🎯 Executando motor de trading ({SIZING_METHOD})...")
            trading_result = TradeEngine(
                mispricing=trading_inputs['mispricing'],
                Sigma=trading_inputs['Sigma'],
                limits=trading_inputs['limits'],
                thresh=trading_inputs['thresh'],
                frictions=trading_inputs['frictions'],
                method=SIZING_METHOD,
                w_prev=w_prev,
                cfg=config
            )
            
            # Diagnóstico das decisões de trading
            signals = trading_result['signals']
            target_w = trading_result['target_w']
            orders = trading_result['orders']
            
            active_positions = (target_w != 0).sum()
            total_exposure = abs(target_w).sum()
            net_exposure = target_w.sum()
            
            print(f"         🎲 Decisões: {active_positions} posições ativas, exposição={total_exposure:.2f}, net={net_exposure:.2f}")
            print(f"         📋 Ordens: {len(orders)} ordens geradas")
            
            if len(orders) > 0:
                buy_orders = sum(1 for _, side, _ in orders if side == 'BUY')
                sell_orders = len(orders) - buy_orders
                print(f"            • Compras: {buy_orders}, Vendas: {sell_orders}")
            
            # SUBSTEP 3.5: Calcular P&L simulado (simplificado)
            print(f"      💰 Calculando P&L do dia...")
            if w_prev is not None and i > 0:
                # Simular retorno baseado na diferença de preços
                prev_test_idx = train_end_idx + i - 1
                prev_prices = F_mkt.iloc[prev_test_idx]
                price_change = (F_mkt_t / prev_prices) - 1
                daily_pnl = (w_prev * price_change * portfolio_value).sum()
                portfolio_value += daily_pnl
                
                print(f"         📊 P&L diário: ${daily_pnl:+,.2f}")
                print(f"         💼 Valor da carteira: ${portfolio_value:,.2f}")
                
                # Mostrar contribuição por tenor (se houver posições)
                if abs(w_prev).sum() > 0:
                    contributions = w_prev * price_change * portfolio_value
                    top_contributor = contributions.abs().idxmax()
                    print(f"         🏆 Maior contribuição: {top_contributor} (${contributions[top_contributor]:+.2f})")
            else:
                daily_pnl = 0.0
                print(f"         ℹ️  Primeiro dia ou sem posições anteriores: P&L = $0.00")
            
            # Atualizar posições
            w_prev = trading_result['target_w'].copy()
            print(f"      🔄 Posições atualizadas: {(w_prev != 0).sum()} ativos, max_weight={abs(w_prev).max():.2f}")
            print()  # Linha em branco para separar dias
            
            # SUBSTEP 3.6: Salvar resultados do dia
            day_data = {
                'model_params': model_result['Theta'],
                'predictions': model_result['F_model_t'],
                'trading_decisions': {
                    'signals': trading_result['signals'],
                    'target_weights': trading_result['target_w'],
                    'orders': trading_result['orders'],
                    'z_scores': trading_result.get('z_scores', [])
                },
                'market_data': {
                    'F_mkt_t': F_mkt_t,
                    'ttm_t': ttm_t,
                    'mispricing': trading_inputs['mispricing']
                },
                'performance': {
                    'portfolio_value': portfolio_value,
                    'daily_pnl': daily_pnl,
                    'num_trades': len(trading_result['orders'])
                }
            }
            
            daily_results[date_str] = day_data
            
            # Logs consolidados
            portfolio_performance.append({
                'date': current_date,
                'portfolio_value': portfolio_value,
                'daily_pnl': daily_pnl,
                'num_positions': (trading_result['target_w'] != 0).sum()
            })
            
            # Log de trades
            for tenor_idx, side, qty in trading_result['orders']:
                trades_log.append({
                    'date': current_date,
                    'tenor': f'tenor_{tenor_idx+1}',
                    'side': side,
                    'quantity': qty,
                    'price': F_mkt_t[tenor_idx]
                })
            
            # Evolução do modelo
            model_params = model_result['Theta'].copy()
            model_params['date'] = current_date
            model_evolution.append(model_params)
            
        except Exception as e:
            print(f"   ❌ ERRO em {date_str}: {e}")
            print(f"      🔍 Tipo: {type(e).__name__}")
            print(f"      📝 Detalhes: {str(e)}")
            
            # Adicionar registro de erro para debug
            daily_results[date_str] = {
                'error': str(e),
                'error_type': type(e).__name__,
                'portfolio_value': portfolio_value
            }
            continue
    
    print()
    print("✅ PASSO 3: BACKTESTING CONCLUÍDO!")
    
    # Estatísticas de execução
    successful_days = len([d for d in daily_results.values() if 'error' not in d])
    error_days = len(daily_results) - successful_days
    initial_value = 100000.0
    total_return = (portfolio_value / initial_value - 1) * 100
    
    print(f"   📊 ESTATÍSTICAS DE EXECUÇÃO:")
    print(f"      • Dias processados com sucesso: {successful_days}/{len(test_dates)}")
    if error_days > 0:
        print(f"      • Dias com erro: {error_days}")
    print(f"      • Valor inicial: ${initial_value:,.0f}")
    print(f"      • Valor final: ${portfolio_value:,.2f}")
    print(f"      • Retorno total: {total_return:+.2f}%")
    print(f"      • Total de operações: {len(trades_log)}")
    
    # Preparar dados para salvamento
    print(f"   📊 Dados consolidados:")
    print(f"      • Portfolio performance: {len(portfolio_performance)} registros")
    print(f"      • Trades log: {len(trades_log)} registros")
    print(f"      • Model evolution: {len(model_evolution)} registros")
    print(f"      • Daily results: {len(daily_results)} dias")
    
    # Converter para DataFrames
    portfolio_df = pd.DataFrame(portfolio_performance)
    trades_df = pd.DataFrame(trades_log) if trades_log else pd.DataFrame()
    model_df = pd.DataFrame(model_evolution) if model_evolution else pd.DataFrame()
    
    results_data = {
        'daily_results': daily_results,
        'portfolio_performance': portfolio_df,
        'trades_log': trades_df,
        'model_evolution': model_df
    }
    
    # Salvar usando função do DataManipulation
    output_path = save_data_to_processed(results_data, DATASET_ID)
    
    print("✅ PASSO 4: Resultados salvos com sucesso!")
    print(f"   📁 Localização: {output_path}")
    
    # Verificar arquivos salvos
    import os
    saved_files = [f for f in os.listdir(output_path) if f.endswith('.csv')]
    print(f"   📄 Arquivos criados: {', '.join(saved_files)}")
    
    # RESUMO FINAL COM DIAGNÓSTICOS DETALHADOS
    print()
    print("=" * 80)
    print("🎉 BACKTESTING EXECUTADO COM SUCESSO!")
    print("=" * 80)
    print(f"📁 Resultados salvos em: {output_path}")
    
    # Estatísticas finais detalhadas
    if len(portfolio_performance) > 0:
        initial_value = portfolio_performance[0]['portfolio_value']
        final_value = portfolio_performance[-1]['portfolio_value']
        total_return = (final_value / initial_value - 1) * 100
        
        # Calcular métricas adicionais
        daily_pnls = [p['daily_pnl'] for p in portfolio_performance if p['daily_pnl'] != 0]
        if daily_pnls:
            avg_daily_pnl = np.mean(daily_pnls)
            volatility = np.std(daily_pnls) * np.sqrt(252)  # Anualizada
            win_rate = sum(1 for pnl in daily_pnls if pnl > 0) / len(daily_pnls)
            sharpe = avg_daily_pnl * np.sqrt(252) / volatility if volatility > 0 else 0
            
            print(f"📊 MÉTRICAS FINAIS DE PERFORMANCE:")
            print(f"   💰 VALORES:")
            print(f"      • Valor inicial: ${initial_value:,.0f}")
            print(f"      • Valor final: ${final_value:,.2f}")
            print(f"      • P&L total: ${final_value - initial_value:+,.2f}")
            print(f"      • Retorno total: {total_return:+.2f}%")
            
            print(f"   📈 ESTATÍSTICAS:")
            print(f"      • Dias testados: {len(portfolio_performance)}")
            print(f"      • P&L médio diário: ${avg_daily_pnl:+.2f}")
            print(f"      • Volatilidade anual: {volatility:.2f}")
            print(f"      • Win rate: {win_rate:.1%}")
            print(f"      • Sharpe ratio: {sharpe:.2f}")
            print(f"      • Melhor dia: ${max(daily_pnls):+,.2f}")
            print(f"      • Pior dia: ${min(daily_pnls):+,.2f}")
        
        print(f"   🔄 ATIVIDADE:")
        print(f"      • Total de operações: {len(trades_log)}")
        if len(trades_log) > 0:
            avg_trades_per_day = len(trades_log) / len(portfolio_performance)
            print(f"      • Operações por dia: {avg_trades_per_day:.1f}")
            
            # Análise por tipo de operação
            buy_trades = sum(1 for _, row in trades_df.iterrows() if row['side'] == 'BUY')
            sell_trades = len(trades_log) - buy_trades
            print(f"      • Compras/Vendas: {buy_trades}/{sell_trades}")
    
    # Status de calibração do modelo
    if calibrated_params:
        print(f"   🧠 PARÂMETROS FINAIS DO MODELO:")
        for param, value in calibrated_params.items():
            if isinstance(value, (int, float)):
                print(f"      • {param}: {value:.4f}")
    
    print()
    print("🚀 PRÓXIMOS PASSOS RECOMENDADOS:")
    if successful_days == len(test_dates):
        print(f"   ✅ TUDO OK! Agora execute com mais dias de teste:")
        print(f"      1. 📊 Análise visual: python Code/analysis.py")
        print(f"      2. 🔬 Teste com 50 dias: Alterar TEST_DAYS=50 em backtest.py")
    else:
        print(f"   ⚠️  Alguns erros detectados ({error_days} dias). Verifique:")
        print(f"      1. 🔍 Logs de erro acima")
        print(f"      2. 🛠️  Ajustar configurações se necessário") 
        print(f"      3. 📊 Análise parcial: python Code/analysis.py")
    
    print(f"")
    print(f"💡 DICAS DE DEBUG:")
    print(f"   • Resultados detalhados em: {output_path}")
    print(f"   • Logs de cada dia salvos em daily_results/")
    print(f"   • Performance diária em portfolio_performance.csv")
    
    # Tempo total de execução
    end_time = time.time()
    total_time = end_time - start_time
    print()
    print(f"⏱️  TEMPO TOTAL DE EXECUÇÃO: {total_time:.1f} segundos ({total_time/60:.1f} minutos)")
    print()


if __name__ == "__main__":
    main()