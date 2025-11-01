# ItauQuant

Modelo de dois fatores de Schwartz-Smith para trading de futuros de commodities.

## Setup

1. **Criar ambiente virtual:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou venv\Scripts\activate  # Windows
```

2. **Instalar dependências:**
```bash
pip install -r requirements.txt
```

## Como rodar

Configure as variaveis no inicio de cada arquivo e no arquivo config/default.yaml

1. **Download de dados:**
```bash
python Code/download.py
```

2. **Backtesting:**
```bash
python Code/backtest.py
```

3. **Análise:**
```bash
python Code/analysis.py
```

## Estrutura

```
ItauQuant/
├── README.md
├── requirements.txt
├── Code/
│   ├── download.py          # Download de dados
│   ├── backtest.py          # Backtesting 
│   ├── analysis.py          # Análise de resultados
│   └── src/                 # Módulos internos
│       ├── Download.py
│       ├── Model.py
│       ├── TradingStrategy.py
│       └── DataManipulation.py
├── help/                    # Documentação
│   ├── Architecture.md
│   ├── Teory.txt
│   └── prompts.txt
├── data/                    # Dados
│   ├── raw/
│   └── processed/
└── Analysis/                # Análises visuais
```