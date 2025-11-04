#!/usr/bin/env python3
"""
Script para executar backtest + analysis em sequência.
Configurado para teste 2011-2012 com 6 meses de janela.
"""

import subprocess
import sys
import time
import os

def run_command(command, description):
    """Executa comando e reporta resultado."""
    print(f"🚀 {description}")
    print(f"   Comando: {command}")
    start_time = time.time()
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        elapsed = time.time() - start_time
        print(f"   ✅ Concluído em {elapsed:.1f}s")
        print(f"   📄 Output: {len(result.stdout.splitlines())} linhas")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"   ❌ ERRO após {elapsed:.1f}s")
        print(f"   💥 Código de saída: {e.returncode}")
        if e.stderr:
            print(f"   📋 Stderr: {e.stderr[:200]}...")
        return False

def main():
    print("="*80)
    print("🎯 EXECUÇÃO SEQUENCIAL: BACKTEST → ANALYSIS")
    print("="*80)
    print("📊 Teste: WTI 2011-2013 (2 ANOS EXATOS + 6 meses treino)")
    print("🎯 ID: WTI2011_2013")
    print()
    
    # Verificar se estamos no diretório correto
    if not os.path.exists("Code/backtest.py"):
        print("❌ ERRO: Execute este script no diretório raiz do projeto ItauQuant")
        sys.exit(1)
    
    start_total = time.time()
    
    # PASSO 1: Backtest
    success = run_command("python Code/backtest.py", "EXECUTANDO BACKTESTING")
    
    if not success:
        print("❌ BACKTESTING FALHOU - Abortando execução")
        sys.exit(1)
    
    print()
    
    # PASSO 2: Analysis
    success = run_command("python Code/analysis.py", "EXECUTANDO ANÁLISE VISUAL")
    
    if not success:
        print("❌ ANÁLISE FALHOU")
        sys.exit(1)
    
    # Resumo final
    total_elapsed = time.time() - start_total
    print()
    print("="*80)
    print("🎉 EXECUÇÃO COMPLETA!")
    print("="*80)
    print(f"⏱️  Tempo total: {total_elapsed/60:.1f} minutos")
    print()
    print("📁 Resultados salvos em:")
    print("   • data/processed/WTI2011_2013/")
    print("   • data/analysis/WTI2011_2013/")
    print()
    print("🎯 Próximo passo: Analisar os resultados gerados!")

if __name__ == "__main__":
    main()
