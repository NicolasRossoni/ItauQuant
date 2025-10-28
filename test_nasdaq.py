#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de teste para validar conexão com Nasdaq Data Link
"""

import os
from dotenv import load_dotenv

# Carrega .env
load_dotenv()
api_key = os.getenv("NASDAQ_DATA_LINK_API_KEY")

print("="*80)
print("TESTANDO NASDAQ DATA LINK (ex-Quandl)")
print("="*80)
print()

# Verifica se API key existe
if not api_key:
    print("❌ ERRO: API KEY não encontrada!")
    print()
    print("Solução:")
    print("1. Crie conta em: https://data.nasdaq.com/sign-up")
    print("2. Copie sua API Key em: https://data.nasdaq.com/account/profile")
    print("3. Crie arquivo .env com:")
    print("   NASDAQ_DATA_LINK_API_KEY=sua_chave_aqui")
    print()
    exit(1)

print(f"✅ API Key encontrada: {api_key[:10]}...{api_key[-4:]}")
print()

# Tenta importar biblioteca
try:
    import nasdaqdatalink
except ImportError:
    print("❌ ERRO: Biblioteca não instalada!")
    print()
    print("Solução:")
    print("   pip install nasdaq-data-link")
    print()
    exit(1)

print("✅ Biblioteca nasdaq-data-link instalada")
print()

# Configura API
nasdaqdatalink.ApiConfig.api_key = api_key

# Testa conexão
print("🔍 Testando download de dados (WTI Crude Oil - Janeiro 2024)...")
try:
    df = nasdaqdatalink.get('CHRIS/CME_CL1', start_date='2024-01-01', end_date='2024-01-31')
    
    print("✅ SUCESSO!")
    print()
    print(f"📊 Dados baixados: {len(df)} dias")
    print(f"📅 Período: {df.index[0].date()} → {df.index[-1].date()}")
    print(f"📈 Colunas: {list(df.columns)}")
    print()
    print(f"💰 Estatísticas do preço (Settle):")
    print(f"   Média: ${df['Settle'].mean():.2f}")
    print(f"   Mínimo: ${df['Settle'].min():.2f}")
    print(f"   Máximo: ${df['Settle'].max():.2f}")
    print()
    print("="*80)
    print("✅ CONFIGURAÇÃO OK! Você pode usar o Nasdaq Data Link")
    print("="*80)
    print()
    print("Próximo passo:")
    print("   python src/DownloadsData_backup.py \\")
    print("     --source chris \\")
    print("     --start-date 2020-01-01 \\")
    print("     --end-date 2024-12-31 \\")
    print("     --num-tenors 8")
    
except Exception as e:
    print(f"❌ ERRO ao baixar dados: {e}")
    print()
    print("Possíveis causas:")
    print("1. API Key inválida - verifique se copiou corretamente")
    print("2. Limite de requisições excedido (50/dia)")
    print("3. Dataset não existe ou foi removido")
    print()
    exit(1)
