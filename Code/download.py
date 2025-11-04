"""
download.py

DEPRECATED - Este arquivo não é mais usado na pipeline principal.
O projeto agora utiliza dados Bloomberg de alta qualidade.

Este arquivo permanece apenas como referência histórica da pipeline original
que começava com downloads do Yahoo Finance. A estrutura atual do projeto
utiliza dados Bloomberg pré-processados que são carregados diretamente
no backtest.py.

Para referência: este era o início da pipeline original:
download.py → backtest.py → analysis.py

CONFIGURAÇÕES HISTÓRICAS (não utilizadas):
"""

# ==========================================
# CONFIGURAÇÕES - MODIFICAR AQUI
# ==========================================

DATASET_ID = "WTI_test_380d"             # ID do dataset a ser criado (18 meses)
DATA_SOURCE = "yahoo"                   # Fonte dos dados: "yahoo" ou "cme" 
START_DATE = "2023-07-01"               # Data inicial dos dados (YYYY-MM-DD) - ~380 dias úteis
END_DATE = "2024-12-31"                 # Data final dos dados (YYYY-MM-DD)
NUM_TENORS = 6                          # Número de tenores (meses) a baixar
COMMODITY = "WTI"                       # Commodity (para CME): "WTI", "NG", etc.

# Configurações avançadas (opcional)
CONTANGO_MONTHLY = 0.004                # Contango mensal para Yahoo Finance (0.4%)

# ==========================================
# FLUXOGRAMA PRINCIPAL
# ==========================================

import sys
import os
import numpy as np
from src.Download import (
    download_yahoo_wti, 
    download_cme_data, 
    format_raw_data,
    save_raw_dataset
)

def main():
    """
    Fluxograma principal de download de dados.
    
    Input: Configurações definidas no início do arquivo
    Output: 
    - Print minimalista no terminal mostrando progresso
    - Arquivos salvos em data/raw/{DATASET_ID}/ com estrutura padronizada:
      * F_mkt.csv: preços futuros [T x M]
      * ttm.csv: time-to-maturity [T x M]  
      * S.csv: preços spot [T] (opcional)
      * costs.csv: custos por tenor [M]
      * info.json: metadados do dataset
    """
    
    print("=" * 80)
    print("🚀 DOWNLOAD DE DADOS - ITAU QUANT")
    print("=" * 80)
    print(f"🔧 CONFIGURAÇÕES DETECTADAS:")
    print(f"   • Dataset ID: {DATASET_ID}")
    print(f"   • Fonte de dados: {DATA_SOURCE.upper()}")
    print(f"   • Período solicitado: {START_DATE} → {END_DATE}")
    print(f"   • Número de tenores: {NUM_TENORS}")
    print(f"   • Contango mensal: {CONTANGO_MONTHLY:.3%}")
    print(f"   • Commodity (CME): {COMMODITY}")
    
    # Validação inicial das configurações
    from datetime import datetime
    try:
        start_dt = datetime.strptime(START_DATE, '%Y-%m-%d')
        end_dt = datetime.strptime(END_DATE, '%Y-%m-%d')
        days_span = (end_dt - start_dt).days
        print(f"   • Período em dias: {days_span} dias")
        print(f"   • Período em anos: {days_span/365.25:.2f} anos")
        
        if days_span < 100:
            print(f"   ⚠️  AVISO: Período curto ({days_span} dias) - pode impactar calibração")
        elif days_span > 1000:
            print(f"   ℹ️  NOTA: Período longo ({days_span} dias) - download pode demorar")
        
    except Exception as e:
        print(f"   ❌ ERRO na validação de datas: {e}")
    
    print()
    
    # PASSO 1: Baixar dados conforme fonte selecionada
    print("📥 PASSO 1: EXECUTANDO DOWNLOAD DE DADOS")
    print(f"   🔄 Iniciando download via {DATA_SOURCE.upper()}...")
    
    try:
        if DATA_SOURCE.lower() == "yahoo":
            print(f"   📡 Conectando ao Yahoo Finance para símbolo CL=F...")
            print(f"   🏗️  Criando estrutura de termo sintética com contango {CONTANGO_MONTHLY:.3%}...")
            
            raw_data = download_yahoo_wti(
                start_date=START_DATE,
                end_date=END_DATE, 
                num_tenors=NUM_TENORS,
                contango_monthly=CONTANGO_MONTHLY
            )
            
        elif DATA_SOURCE.lower() == "cme":
            print(f"   📡 Conectando à API CME para commodity {COMMODITY}...")
            print(f"   ⚠️  ATENÇÃO: CME é placeholder - gerando dados sintéticos")
            
            raw_data = download_cme_data(
                commodity=COMMODITY,
                start_date=START_DATE,
                end_date=END_DATE
            )
            
        else:
            raise ValueError(f"Fonte '{DATA_SOURCE}' não suportada. Use: yahoo ou cme")
        
        # Diagnóstico dos dados baixados
        F_mkt_shape = raw_data['F_mkt'].shape
        info = raw_data.get('info', {})
        
        print("✅ PASSO 1: Download concluído com sucesso!")
        print(f"   📊 Dimensões dos dados: {F_mkt_shape[0]} dias × {F_mkt_shape[1]} tenores")
        print(f"   💰 Faixa de preços: ${info.get('price_range', {}).get('min', 0):.2f} - ${info.get('price_range', {}).get('max', 0):.2f}")
        print(f"   📈 Preço médio: ${info.get('price_range', {}).get('mean', 0):.2f}")
        
        # Verificação de qualidade dos dados
        nan_count = raw_data['F_mkt'].isna().sum().sum()
        if nan_count > 0:
            print(f"   ⚠️  ALERTA: {nan_count} valores NaN encontrados nos dados")
        else:
            print(f"   ✅ Qualidade: Sem valores NaN detectados")
        
    except Exception as e:
        print(f"❌ ERRO CRÍTICO no PASSO 1: {e}")
        print(f"   🔍 Tipo do erro: {type(e).__name__}")
        print(f"   📝 Detalhes: {str(e)}")
        sys.exit(1)
    
    # PASSO 2: Formatar dados para estrutura padrão
    print()
    print("🔧 PASSO 2: FORMATANDO E VALIDANDO DADOS")
    print(f"   🔄 Convertendo para estrutura padrão do projeto...")
    
    try:
        formatted_data = format_raw_data(raw_data, validate=True)
        
        # Diagnóstico detalhado da formatação
        F_mkt = formatted_data['F_mkt']
        ttm = formatted_data['ttm']
        S = formatted_data.get('S')
        
        print("✅ PASSO 2: Formatação concluída com sucesso!")
        print(f"   📊 F_mkt: {F_mkt.shape} | dtype: {F_mkt.dtypes.iloc[0]}")
        print(f"   📊 ttm: {ttm.shape} | dtype: {ttm.dtypes.iloc[0]}")
        if S is not None:
            print(f"   📊 S (spot): {S.shape} | dtype: {S.dtype}")
        
        # Verificações de integridade
        if F_mkt.index.equals(ttm.index):
            print(f"   ✅ Índices temporais: F_mkt e ttm perfeitamente alinhados")
        else:
            print(f"   ⚠️  ALERTA: Desalinhamento entre índices F_mkt e ttm")
        
        # Estatísticas de volatilidade por tenor
        returns = F_mkt.pct_change().dropna()
        vol_by_tenor = returns.std() * np.sqrt(252)  # Anualizada
        print(f"   📈 Volatilidade anual por tenor: {vol_by_tenor.min():.1%} - {vol_by_tenor.max():.1%}")
        
    except Exception as e:
        print(f"❌ ERRO CRÍTICO no PASSO 2: {e}")
        print(f"   🔍 Tipo do erro: {type(e).__name__}")
        print(f"   📝 Contexto: Formatação de dados brutos para estrutura padrão")
        sys.exit(1)
    
    # PASSO 3: Salvar dataset na estrutura padronizada
    print()
    print("💾 PASSO 3: SALVANDO DATASET NO SISTEMA DE ARQUIVOS")
    print(f"   🗂️  Criando estrutura em data/raw/{DATASET_ID}/...")
    
    try:
        dataset_path = save_raw_dataset(
            formatted_data=formatted_data,
            dataset_id=DATASET_ID,
            output_path="data/raw"
        )
        
        # Verificação dos arquivos salvos
        import os
        files_created = [f for f in os.listdir(dataset_path) if f.endswith(('.csv', '.json'))]
        total_size = sum(os.path.getsize(os.path.join(dataset_path, f)) for f in files_created)
        
        print("✅ PASSO 3: Dataset salvo com sucesso!")
        print(f"   📁 Localização: {dataset_path}")
        print(f"   📄 Arquivos criados: {len(files_created)} ({', '.join(files_created)})")
        print(f"   💾 Tamanho total: {total_size/1024:.1f} KB")
        
    except Exception as e:
        print(f"❌ ERRO CRÍTICO no PASSO 3: {e}")
        print(f"   🔍 Tipo do erro: {type(e).__name__}")
        print(f"   📝 Contexto: Salvamento do dataset formatado")
        sys.exit(1)
    
    # RESUMO FINAL COM DIAGNÓSTICOS COMPLETOS
    print()
    print("=" * 80)
    print("🎉 EXECUÇÃO CONCLUÍDA COM SUCESSO!")
    print("=" * 80)
    print(f"📁 Dataset criado: {dataset_path}")
    
    # Estatísticas detalhadas para debugging
    F_mkt = formatted_data['F_mkt']
    returns = F_mkt.pct_change().dropna()
    
    print(f"📊 DIAGNÓSTICOS FINAIS:")
    print(f"   • Período efetivo: {F_mkt.index[0].date()} → {F_mkt.index[-1].date()}")
    print(f"   • Dias úteis disponíveis: {len(F_mkt)}")
    print(f"   • Tenores configurados: {F_mkt.shape[1]}")
    print(f"   • Preço spot inicial: ${F_mkt.iloc[0, 0]:.2f}")
    print(f"   • Preço spot final: ${F_mkt.iloc[-1, 0]:.2f}")
    print(f"   • Retorno total (tenor 1): {(F_mkt.iloc[-1, 0] / F_mkt.iloc[0, 0] - 1):.2%}")
    print(f"   • Volatilidade diária média: {returns.std().mean():.3%}")
    print(f"   • Correlação média entre tenores: {F_mkt.corr().values[np.triu_indices_from(F_mkt.corr().values, k=1)].mean():.3f}")
    
    # Verificação da estrutura de termo
    final_prices = F_mkt.iloc[-1]
    contango_check = ((final_prices.iloc[1:].values / final_prices.iloc[:-1].values - 1) * 12).mean()  # Anualizado
    print(f"   • Contango médio final: {contango_check:.2%} (esperado ~{CONTANGO_MONTHLY*12:.2%})")
    
    print()
    print("🚀 PRÓXIMOS PASSOS RECOMENDADOS:")
    print(f"   1. 🧪 Teste básico (5 dias): Modificar backtest.py com TEST_DAYS=5")
    print(f"   2. 🔬 Teste completo (50 dias): Após validação do teste básico")
    print(f"   3. 📊 Análise completa: python Code/analysis.py")
    print(f"")
    print(f"💡 DICA DE DEBUG: Se algum passo falhar, verifique:")
    print(f"   • Conectividade de rede (Yahoo Finance)")
    print(f"   • Permissões de escrita na pasta data/")
    print(f"   • Datas válidas e em ordem cronológica")
    print()


if __name__ == "__main__":
    main()