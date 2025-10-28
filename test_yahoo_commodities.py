#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testa disponibilidade de commodities no Yahoo Finance
"""

import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd

# Commodities disponíveis no Yahoo Finance
COMMODITIES = {
    # Energéticas (EUA)
    'CL=F': 'WTI Crude Oil (front month)',
    'NG=F': 'Natural Gas',
    'RB=F': 'RBOB Gasoline',
    'HO=F': 'Heating Oil',
    'BZ=F': 'Brent Crude Oil',
    
    # Metais (EUA)
    'GC=F': 'Gold',
    'SI=F': 'Silver',
    'HG=F': 'Copper',
    'PL=F': 'Platinum',
    'PA=F': 'Palladium',
    
    # Agrícolas (EUA - CME)
    'ZC=F': 'Corn',
    'ZS=F': 'Soybeans',
    'ZW=F': 'Wheat',
    'KC=F': 'Coffee',
    'SB=F': 'Sugar',
    'CC=F': 'Cocoa',
    'CT=F': 'Cotton',
    'LBS=F': 'Lumber',
}

def test_ticker(ticker, name):
    """Testa se um ticker tem dados disponíveis"""
    # Usar período fixo em 2024 que sabemos que tem dados
    start_date = '2024-01-01'
    end_date = '2024-10-01'
    
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
        if df is not None and not df.empty and len(df) > 10:
            return {
                'ticker': ticker,
                'name': name,
                'status': '✅ OK',
                'days': len(df),
                'last_price': f"${df['Close'].iloc[-1]:.2f}",
                'mean_volume': f"{df['Volume'].mean():.0f}"
            }
    except Exception as e:
        pass
    
    return {
        'ticker': ticker,
        'name': name,
        'status': '❌ Falhou',
        'days': 0,
        'last_price': 'N/A',
        'mean_volume': 'N/A'
    }

if __name__ == "__main__":
    print("="*80)
    print("TESTANDO COMMODITIES DISPONÍVEIS NO YAHOO FINANCE")
    print("="*80)
    print()
    
    results = []
    for ticker, name in COMMODITIES.items():
        print(f"Testando {ticker:10s} ({name})...", end=' ')
        result = test_ticker(ticker, name)
        results.append(result)
        print(result['status'])
    
    print("\n" + "="*80)
    print("RESUMO")
    print("="*80)
    
    df_results = pd.DataFrame(results)
    
    # Commodities disponíveis
    available = df_results[df_results['status'] == '✅ OK']
    print(f"\n✅ DISPONÍVEIS: {len(available)}/{len(results)}")
    print("\nDetalhes:")
    for _, row in available.iterrows():
        print(f"  {row['ticker']:10s} | {row['name']:30s} | {row['days']} dias | Último: {row['last_price']}")
    
    # Recomendações
    print("\n" + "="*80)
    print("RECOMENDAÇÕES PARA SEU MODELO")
    print("="*80)
    
    print("\n🔥 ENERGÉTICAS (alta volatilidade, bom para Schwartz-Smith):")
    energy = available[available['ticker'].isin(['CL=F', 'NG=F', 'RB=F', 'HO=F', 'BZ=F'])]
    for _, row in energy.iterrows():
        print(f"  • {row['ticker']:8s} - {row['name']}")
    
    print("\n🥇 METAIS (volatilidade média):")
    metals = available[available['ticker'].isin(['GC=F', 'SI=F', 'HG=F'])]
    for _, row in metals.iterrows():
        print(f"  • {row['ticker']:8s} - {row['name']}")
    
    print("\n🌾 AGRÍCOLAS (volatilidade sazonal):")
    agri = available[available['ticker'].isin(['ZC=F', 'ZS=F', 'ZW=F', 'KC=F', 'SB=F'])]
    for _, row in agri.iterrows():
        print(f"  • {row['ticker']:8s} - {row['name']}")
