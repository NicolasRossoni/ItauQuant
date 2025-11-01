"""
download.py

Script principal de download de dados de futuros de commodities.
Este é um FLUXOGRAMA LIMPO que abstrai toda complexidade no módulo src/Download.py.

CONFIGURAÇÕES (início do arquivo - modificar aqui):
"""

# ==========================================
# CONFIGURAÇÕES - MODIFICAR AQUI
# ==========================================

DATASET_ID = "WTI_2024_yahoo"           # ID do dataset a ser criado
DATA_SOURCE = "yahoo"                   # Fonte dos dados: "yahoo", "cme" ou "synthetic"
START_DATE = "2024-01-01"               # Data inicial dos dados (YYYY-MM-DD)
END_DATE = "2024-12-31"                 # Data final dos dados (YYYY-MM-DD)
NUM_TENORS = 8                          # Número de tenores (meses) a baixar
COMMODITY = "WTI"                       # Commodity (para CME): "WTI", "NG", etc.

# Configurações avançadas (opcional)
CONTANGO_MONTHLY = 0.004                # Contango mensal para Yahoo Finance (0.4%)
BASE_PRICE = 80.0                       # Preço base para dados sintéticos
VOLATILITY = 0.25                       # Volatilidade anual para sintéticos (25%)

# ==========================================
# FLUXOGRAMA PRINCIPAL
# ==========================================

import sys
import os
from src.Download import (
    download_yahoo_wti, 
    download_cme_data, 
    create_synthetic_dataset,
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
    
    print("=" * 60)
    print("🚀 DOWNLOAD DE DADOS - ITAU QUANT")
    print("=" * 60)
    print(f"Dataset ID: {DATASET_ID}")
    print(f"Fonte: {DATA_SOURCE.upper()}")
    print(f"Período: {START_DATE} → {END_DATE}")
    print(f"Tenores: {NUM_TENORS}")
    print()
    
    # PASSO 1: Baixar dados conforme fonte selecionada
    print("📥 Passo 1: Baixando dados...")
    
    try:
        if DATA_SOURCE.lower() == "yahoo":
            raw_data = download_yahoo_wti(
                start_date=START_DATE,
                end_date=END_DATE, 
                num_tenors=NUM_TENORS,
                contango_monthly=CONTANGO_MONTHLY
            )
            
        elif DATA_SOURCE.lower() == "cme":
            raw_data = download_cme_data(
                commodity=COMMODITY,
                start_date=START_DATE,
                end_date=END_DATE
            )
            
        elif DATA_SOURCE.lower() == "synthetic":
            dataset_path = create_synthetic_dataset(
                dataset_id=DATASET_ID,
                start_date=START_DATE,
                end_date=END_DATE,
                num_tenors=NUM_TENORS,
                base_price=BASE_PRICE,
                volatility=VOLATILITY
            )
            print("✅ Download concluído com sucesso!")
            print(f"📁 Dados salvos em: {dataset_path}")
            return
            
        else:
            raise ValueError(f"Fonte '{DATA_SOURCE}' não suportada. Use: yahoo, cme ou synthetic")
            
        print("✅ Download concluído!")
        print(f"   Dados baixados: {raw_data['F_mkt'].shape}")
        
    except Exception as e:
        print(f"❌ Erro no download: {e}")
        sys.exit(1)
    
    # PASSO 2: Formatar dados para estrutura padrão
    print()
    print("🔧 Passo 2: Formatando dados...")
    
    try:
        formatted_data = format_raw_data(raw_data, validate=True)
        print("✅ Formatação concluída!")
        
    except Exception as e:
        print(f"❌ Erro na formatação: {e}")
        sys.exit(1)
    
    # PASSO 3: Salvar dataset na estrutura padronizada
    print()
    print("💾 Passo 3: Salvando dataset...")
    
    try:
        dataset_path = save_raw_dataset(
            formatted_data=formatted_data,
            dataset_id=DATASET_ID,
            output_path="data/raw"
        )
        print("✅ Dataset salvo com sucesso!")
        
    except Exception as e:
        print(f"❌ Erro ao salvar: {e}")
        sys.exit(1)
    
    # RESUMO FINAL
    print()
    print("=" * 60)
    print("🎉 DOWNLOAD CONCLUÍDO COM SUCESSO!")
    print("=" * 60)
    print(f"📁 Dataset: {dataset_path}")
    
    # Mostrar estatísticas básicas
    F_mkt = formatted_data['F_mkt']
    print(f"📊 Estatísticas:")
    print(f"   • Período: {F_mkt.index[0].date()} → {F_mkt.index[-1].date()}")
    print(f"   • Dias úteis: {len(F_mkt)}")
    print(f"   • Tenores: {F_mkt.shape[1]}")
    print(f"   • Preço médio: ${F_mkt.mean().mean():.2f}")
    print(f"   • Volatilidade diária média: {F_mkt.pct_change().std().mean():.1%}")
    
    print()
    print("🚀 Próximos passos:")
    print(f"   1. Teste rápido: python Code/backtest.py")
    print(f"   2. Análise: python Code/analysis.py")
    print()


if __name__ == "__main__":
    main()