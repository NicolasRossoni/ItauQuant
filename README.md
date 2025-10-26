# ItauQuant

This is the code of the Alpha Fisher Grup in the Itaú Asset Quant AI 2025 Challenge.

## 📋 Descrição

Implementação do modelo de dois fatores de Schwartz-Smith para trading de futuros de commodities, incluindo:
- Calibração via Filtro de Kalman (MLE/EM)
- Cálculo de forwards teóricos com fórmula fechada
- Motor de decisão de trading com gestão de risco
- Pipeline completo de análise e geração de gráficos

## 🚀 Setup do Ambiente

### 1. Criar Virtual Environment

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

### 2. Instalar Dependências

```bash
pip install -r requirements.txt
```


## 📁 Estrutura do Projeto

```
ItauQuant/
├── src/
│   ├── ComputeModelForward.py      # Kalman + fórmula fechada
│   ├── PrepareTradingInputs.py     # Mispricing, Sigma, limits
│   ├── TradeEngine.py              # Sinais e ordens
│   ├── GenerateFakeData.py         # Dados sintéticos
│   ├── TestingTheoryPipeline.py    # Pipeline + gráficos
│   ├── DownloadsData.py            # Placeholder API
│   └── Main.py                     # Execução rápida
├── config/
│   └── default.yaml                # Configurações
├── data/fakedata/                  # Datasets sintéticos
├── images/fake_data_analysis/      # Gráficos gerados
├── requirements.txt                # Dependências Python
└── README.md                       # Este arquivo
```
