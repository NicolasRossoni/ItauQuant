# 📊 Backtesting - Guia Completo

## 🎯 O Que Foi Implementado

Sistema completo de backtesting da estratégia Schwartz-Smith com recalibração rolling.

### Estrutura

```
Dados: 3000 dias totais
├─ Treino: 1500 dias (calibração inicial)
└─ Teste: 1500 dias (atuação do modelo)
    └─ A cada dia: recalibra → prevê → trade
```

### Outputs

```
output/YYYY-MM-DD_HH-MM-SS/
├── pnl_daily.csv           # P&L diário da estratégia
├── trades.csv              # Histórico de todas as ordens
├── metrics.csv             # Métricas vs benchmarks
└── images/
    ├── 00_pnl_comparison.png        # GRÁFICO PRINCIPAL: P&L comparativo
    ├── 01_tenor_1_signals.png       # Sinais BUY/SELL por tenor
    ├── 02_tenor_2_signals.png
    └── ...
```

---

## 📈 Benchmarks Comparativos

### 1. Buy & Hold M1
- Comprar e segurar o front-month (primeiro tenor)
- Benchmark mais simples

### 2. CDI/Selic (13.75% aa)
- Taxa livre de risco brasileira
- Equivalente a renda fixa

### 3. Roll Strategy
- Rolar contratos sistematicamente (M1 → M2)
- Estratégia passiva de futuros

---

## 🚀 Como Usar

### 1. Gerar Dados (3000 dias)

```bash
python src/GenerateFakeData.py \
  --dataset-name wti_synth_backtest \
  --T 3000 \
  --M 8 \
  --seed 42
```

### 2. Rodar Backtesting

```bash
python src/Backtester.py \
  --dataset-root data/fakedata/wti_synth_backtest \
  --train-days 1500 \
  --test-days 1500 \
  --method EM \
  --sizing vol_target
```

**Tempo estimado**: 20-30 minutos (1500 otimizações Kalman)

### 3. Analisar Resultados

```bash
# Ver métricas
cat output/YYYY-MM-DD_HH-MM-SS/metrics.csv

# Ver gráficos
open output/YYYY-MM-DD_HH-MM-SS/images/00_pnl_comparison.png
```

---

## 📊 Gráficos Gerados

### Gráfico Principal: P&L Comparativo

![](images/00_pnl_comparison.png)

**Mostra**:
- Linha azul sólida: Estratégia Schwartz-Smith
- Linha rosa tracejada: Buy & Hold M1
- Linha laranja tracejada: CDI (13.75%)
- Linha verde tracejada: Roll Strategy

**Métricas exibidas**:
- P&L Total ($)
- Sharpe Ratio
- Max Drawdown (%)
- Win Rate (%)

### Gráficos por Tenor (8 total)

Cada tenor tem um gráfico mostrando:

1. **Área cinza**: Período de treino (1500 dias)
2. **Área verde**: Período de teste (1500 dias)
3. **Linha vertical preta**: Divisor treino/teste
4. **Triângulos verdes ▲**: Sinais BUY
5. **Triângulos vermelhos ▼**: Sinais SELL

---

## 📄 CSVs Salvos

### pnl_daily.csv

| date | t | pnl_day | pnl_cum |
|------|---|---------|---------|
| 2023-01-01 | 1500 | 12.50 | 12.50 |
| 2023-01-02 | 1501 | -5.30 | 7.20 |
| ... | ... | ... | ... |

### trades.csv

| date | t | tenor | side | quantity | price | cost |
|------|---|-------|------|----------|-------|------|
| 2023-01-01 | 1500 | 0 | SELL | 0.9 | 70.5 | 1.8 |
| ... | ... | ... | ... | ... | ... | ... |

### metrics.csv

| Metric | Strategy | Buy&Hold M1 | CDI | Roll |
|--------|----------|-------------|-----|------|
| Total P&L | $1234.56 | $987.65 | $543.21 | $876.54 |
| Sharpe | 1.234 | 0.987 | 0.123 | 0.654 |
| Max DD | -12.3% | -18.9% | 0.0% | -15.4% |
| Win Rate | 58.2% | - | - | - |
| Trades | 1234 | - | - | - |

---

## 🔧 Parâmetros Configuráveis

```bash
--dataset-root      # Caminho dos dados
--train-days 1500   # Dias para treino
--test-days 1500    # Dias para teste
--method EM         # MLE ou EM
--sizing vol_target # vol_target ou qp
```

---

## 📝 Interpretação dos Resultados

### ✅ Boa Performance

- **Sharpe > 1.0**: Retorno ajustado por risco superior
- **Max DD < 20%**: Perdas controladas
- **Win Rate > 50%**: Mais trades lucrativos que perdedores
- **P&L > Benchmarks**: Estratégia supera passivos

### ⚠️ Performance Ruim

- **Sharpe < 0.5**: Retorno não compensa risco
- **Max DD > 40%**: Perdas muito grandes
- **P&L < CDI**: Não vale o risco vs renda fixa

---

## 🎓 Metodologia

### Recalibração Rolling

A cada dia no período de teste:

1. **Usa todos os dados até aquele dia** para calibrar Kalman
2. **Gera previsão** F_model para aquele dia
3. **Calcula mispricing** = F_model - F_mkt
4. **Gera sinais** (BUY/SELL/HOLD)
5. **Executa trades** e registra P&L
6. **Avança** para o próximo dia

Isso simula condições **reais**: no dia T, só sabemos dados até T-1.

### Cálculo de P&L

```python
# P&L do dia
P&L = (posição_anterior × variação_preço) - custos_transação

# Custos
custo = |quantidade_tradada| × fee_por_contrato
```

### Benchmarks

**CDI**:
```python
retorno_diário = 0.1375 / 252  # 13.75% aa ÷ 252 dias úteis
P&L_acum = (1 + retorno_diário)^dias - 1
```

**Buy & Hold M1**:
```python
retorno = (preço_final - preço_inicial) / preço_inicial
```

**Roll Strategy**:
```python
retorno = média(M1, M2).pct_change()
```

---

## 🐛 Troubleshooting

### Erro: Dataset muito pequeno

```
ValueError: Dataset muito pequeno! Tem 1500 dias, precisa de 3000
```

**Solução**: Gere dados com T ≥ train_days + test_days

```bash
python src/GenerateFakeData.py --T 3000
```

### Backtesting muito lento

**Causa**: Recalibração Kalman a cada dia (1500 otimizações)

**Soluções**:
1. Reduzir `--test-days` (ex: 500 em vez de 1500)
2. Usar `--method MLE` (um pouco mais rápido que EM)
3. Aumentar tolerância no `config/default.yaml`:
   ```yaml
   kalman:
     tol: 1e-4  # Antes: 1e-6
     max_iter: 50  # Antes: 200
   ```

### P&L sempre zero

**Causa**: Nenhum sinal gerado (z-scores < threshold)

**Solução**: Reduzir `z_in` no config:
```yaml
thresh:
  z_in: 1.0  # Antes: 1.5
```

---

## 💡 Próximos Passos

1. **Dados Reais**: Rodar com dados CME (via `DownloadsData.py`)
2. **Otimização**: Testar diferentes hiperparâmetros
3. **Walk-Forward**: Recalibrar a cada N dias (não todo dia)
4. **Transaction Costs**: Ajustar fees reais da corretora
5. **Risk Management**: Adicionar stop-loss, position sizing dinâmico

---

## 📚 Referências

- **Modelo**: Schwartz & Smith (2000) - Short-Term Long-Term Model
- **Kalman Filter**: Welch & Bishop (2006)
- **Backtesting**: Prado (2018) - Advances in Financial ML
