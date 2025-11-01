# Architecture.md - BIBLIA ESTRUTURAL E FILOSÓFICA DO PROJETO

> **ATENÇÃO**: Este é o documento SAGRADO do projeto. Sempre consulte este arquivo antes de qualquer modificação.
> Este README é o "prompt de implementação" para gerar **toda a estrutura de código em Python** do projeto.  
> Use-o junto com o **documento Teory.txt** (teoria e fórmulas). Aqui tratamos **apenas** de engenharia de software: arquivos, funções, I/O, formatos, gráficos e passos de execução.

---

## 1) MANDAMENTOS ESTRUTURAIS E FILOSÓFICOS (NUNCA QUEBRAR)

### 1.1 Princípios de Código (IMUTÁVEIS):
- **Documentação impecável**: cada função deve ter docstring clara (contexto, parâmetros com tipos e shapes, retorno com tipos e shapes, exceções, exemplos).  
- **Nomes autoexplicativos**: variáveis e funções com nomes óbvios, sem abreviações crípticas.  
- **Modularidade**: cada arquivo expõe **uma função pública principal**; auxiliares ficam internas e **começam com `_`**.  
- **I/O determinístico**: funções recebem tudo que precisam como argumentos e retornam resultados claros; efeitos colaterais (leitura/escrita em disco) apenas quando explicitamente indicado.  
- **Eficiência e clareza**: código enxuto, sem redundâncias, mas com comentários estratégicos explicando decisões e fórmulas implementadas.  
- **Tipos e validações**: use `typing` (e.g., `np.ndarray`, `pd.DataFrame`) e valide shapes/NaNs.  
- **Logs**: `logging` em nível INFO/DEBUG para fases críticas (ajuste do estado/EM/MLE, métricas de risco, sizing, geração de gráficos).

### 1.2 Filosofia de Execução (SAGRADA):
- **Configurações intuitivas**: strings definidas no início de cada arquivo com explicações claras ao lado
- **Execução "seca"**: python arquivo.py deve funcionar diretamente no terminal
- **Separação de responsabilidades**: src/ contém toda implementação complexa, arquivos principais são apenas fluxogramas legíveis
- **Lógica abstraída**: códigos principais parecem fluxogramas para facilitar entendimento do pipeline
- **Dados completos**: sempre salvar tudo que pode ser útil para análise posterior (nunca precisar reprocessar)

---

## 2) ESTRATÉGIA DE REFATORAMENTO

### 2.1 Pipeline Principal:
1. **Download**: `download.py` → baixa dados e salva em `data/raw/{dataset_id}/`
2. **Backtest**: `backtest.py` → executa modelo para período especificado e salva em `data/processed/{dataset_id}/`  
3. **Analysis**: `analysis.py` → gera análises visuais em `Analysis/{dataset_id}/`

### 2.2 Módulos de Suporte (src/):
- **Download.py**: toda burocracia de downloads
- **DataManipulation.py**: manipulação pesada de dados, abstraída dos arquivos principais
- **Model.py**: implementação completa do modelo Schwartz-Smith
- **TradingStrategy.py**: estratégia de trading completa

### 2.3 Pipeline Detalhada:
1. **ComputeModelForward**: calibra modelo (Kalman MLE/EM), filtra estados, calcula forwards teóricos via fórmula fechada
2. **PrepareTradingInputs**: computa mispricing, matriz de risco Σ, limites, limiares, fricções  
3. **TradeEngine**: transforma desvio+risco+limites em ordens (buy/sell/qty), signals, target weights
4. **Backtesting**: execução rolling do pipeline completo com recalibração
5. **Analysis**: geração de gráficos e métricas de performance

---

## 3) NOVA ESTRUTURA DE DIRETÓRIOS E ARQUIVOS

```
ItauQuant/
├── README.md                        # Minimalista: título, setup, estrutura
├── requirements.txt                 # Dependências do projeto
├── Code/
│   ├── download.py                  # Script principal de download (fluxograma)
│   ├── backtest.py                  # Script principal de backtesting (fluxograma)  
│   ├── analysis.py                  # Script principal de análise (fluxograma)
│   └── src/                         # Módulos de implementação
│       ├── Download.py              # Toda burocracia de downloads
│       ├── DataManipulation.py      # Manipulação pesada de dados
│       ├── Model.py                 # Modelo Schwartz-Smith completo
│       └── TradingStrategy.py       # Estratégia de trading completa
├── help/                            # Documentação sagrada do projeto
│   ├── Architecture.md              # Este arquivo (BIBLIA)
│   ├── Teory.txt                    # Teorias matemáticas (LIVRO SAGRADO)
│   └── prompts.txt                  # Backup dos prompts originais
├── data/                            # Dados do projeto
│   ├── raw/                         # Dados brutos baixados
│   │   └── {dataset_id}/           # Um dataset por pasta
│   └── processed/                   # Dados processados do backtest
│       └── {dataset_id}/           # Resultados de cada execução
└── Analysis/                        # Análises visuais
    └── {dataset_id}/               # Análises por dataset
        ├── daily_predictions/       # Gráfico por dia
        ├── performance/            # Performance geral vs benchmarks
        ├── by_tenor/              # Análise por tenor
        └── others/                # Outras análises
```

---

## 4) ESPECIFICAÇÃO DE CADA ARQUIVO/MÓDULO

### 4.1 `Code/download.py` (FLUXOGRAMA PRINCIPAL)
```python
# CONFIGURAÇÕES (início do arquivo)
DATASET_ID = "WTI_2024"              # ID do dataset a ser criado
START_DATE = "2020-01-01"            # Data inicial dos dados
END_DATE = "2024-12-31"              # Data final dos dados
DATA_SOURCE = "CME"                  # Fonte dos dados

# FLUXOGRAMA (código simples, legível)
def main():
    print("Iniciando download...")
    dados = baixar_dados(DATASET_ID, START_DATE, END_DATE, DATA_SOURCE)
    salvar_dados(dados, f"data/raw/{DATASET_ID}/")  
    print("Download concluído")
```

### 4.2 `Code/backtest.py` (FLUXOGRAMA PRINCIPAL) 
```python
# CONFIGURAÇÕES
DATASET_ID = "WTI_2024"              # Dataset a ser usado
TRAIN_START = "2020-01-01"           # Início dos dados de treino
TEST_START = "2023-01-01"            # Início dos testes
TEST_END = "2024-12-31"              # Fim dos testes

# FLUXOGRAMA
def main():
    for dia in periodo_teste:
        print(f"Processando {dia}...")
        modelo = calibrar_modelo(dados_historicos_ate(dia))
        decisoes = gerar_decisoes_trading(modelo, dia)
        salvar_resultados_dia(dia, modelo, decisoes)
    
    salvar_resultado_geral()
```

### 4.3 `Code/analysis.py` (FLUXOGRAMA PRINCIPAL)
```  
# CONFIGURAÇÕES
DATASET_ID = "WTI_2024"              # Dataset a ser analisado

# FLUXOGRAMA
def main():
    resultados = carregar_resultados(f"data/processed/{DATASET_ID}/")
    
    # 3 grandes categorias de análise
    gerar_predicoes_diarias(resultados)      # Analysis/{ID}/daily_predictions/
    gerar_analise_performance(resultados)    # Analysis/{ID}/performance/  
    gerar_analise_por_tenor(resultados)     # Analysis/{ID}/by_tenor/
    gerar_outras_analises(resultados)       # Analysis/{ID}/others/
```

### 4.4 `Code/src/Model.py` (IMPLEMENTAÇÃO PESADA)
- Função principal: `ComputeModelForward(...)`
- Implementa modelo Schwartz-Smith completo
- Kalman Filter (MLE/EM)
- Fórmula fechada dos forwards
- Toda matemática pesada fica aqui

### 4.5 `Code/src/TradingStrategy.py` (IMPLEMENTAÇÃO PESADA)  
- Função principal: `TradeEngine(...)`
- `PrepareTradingInputs(...)`
- Cálculo de mispricing, matriz de risco
- Decisões de trading, position sizing
- Toda lógica de estratégia fica aqui

### 4.6 `Code/src/DataManipulation.py` (IMPLEMENTAÇÃO PESADA)
- Todas manipulações de dados complexas
- Formatação, limpeza, transformações
- Funções auxiliares para todos os módulos
- Deixa códigos principais limpos

### 4.7 `Code/src/Download.py` (IMPLEMENTAÇÃO PESADA)
- Toda burocracia de download de APIs
- Conexões, autenticação, parsing
- Formatação para estrutura padrão
- Tratamento de erros de rede

---

## 5) ESTRUTURA DE DADOS PADRONIZADA

### 5.1 Raw Data (`data/raw/{dataset_id}/`):
- `F_mkt.csv`: Preços futuros [T x M] 
- `ttm.csv`: Time-to-maturity [T x M]
- `S.csv`: Preços spot [T] (opcional)
- `costs.csv`: Custos por tenor [M] (opcional)

### 5.2 Processed Data (`data/processed/{dataset_id}/`):
```
├── daily_results/                   # Resultados por dia
│   ├── 2023-01-01/
│   │   ├── model_params.csv         # Parâmetros calibrados
│   │   ├── predictions.csv          # Predições do modelo  
│   │   ├── trading_decisions.csv    # Decisões de trading
│   │   └── market_data.csv          # Dados de mercado do dia
│   └── ...
├── portfolio_performance.csv        # Performance da carteira
├── trades_log.csv                  # Log de todas as operações
└── model_evolution.csv             # Evolução dos parâmetros
```

---

## 6) REGRAS DE EXECUÇÃO E SAÍDAS

### 6.1 Outputs Obrigatórios:
- **Terminal**: Prints minimalistas, apenas status principal
- **Arquivos**: Tudo que pode ser útil para análise futura  
- **Estrutura**: Sempre organized, documentada, reutilizável

### 6.2 Análises Obrigatórias (`Analysis/{dataset_id}/`):
1. **daily_predictions/**: Um gráfico PNG por dia com:
   - Curva do modelo vs mercado
   - Preços históricos reais  
   - Predições futuras do modelo
   - Preços futuros do mercado (contratos)

2. **performance/**: 
   - Performance vs benchmarks (CDI, Buy&Hold)
   - Evolução dos parâmetros do modelo
   - Métricas de risco-retorno

3. **by_tenor/**:
   - Uma pasta por tenor
   - Performance específica por tenor
   - Decisões de compra/venda sobre preços
   - Ratio: (preço_modelo - preço_mercado) / preço_mercado

4. **others/**: Outras análises relevantes

---

## 7) IMPLEMENTAÇÃO - FASES DO REFATORAMENTO

### FASE 1: Estrutura Base
1. ✓ Criar backup dos prompts
2. ✓ Atualizar Architecture.md  
3. Atualizar requirements.txt
4. Criar estrutura de diretórios

### FASE 2: Implementação Core  
1. Implementar Code/src/Model.py (baseado em _old/ComputeModelForward.py)
2. Implementar Code/src/TradingStrategy.py (baseado em _old/TradeEngine.py + PrepareTradingInputs.py)
3. Implementar Code/src/DataManipulation.py (extrair lógica comum)
4. Implementar Code/src/Download.py (baseado em _old/DownloadsData.py)

### FASE 3: Scripts Principais
1. Implementar Code/download.py (fluxograma limpo)
2. Implementar Code/backtest.py (baseado em _old/Backtester.py)  
3. Implementar Code/analysis.py (baseado em _old/BacktesterPlots.py)

### FASE 4: Finalização
1. Atualizar README.md (minimalista)
2. Testes de compatibilidade
3. Validação completa

---

## 8) CHECKLIST DE CONFORMIDADE

### ✅ Estruturais:
- [ ] Códigos principais são fluxogramas legíveis
- [ ] Implementação pesada está em src/
- [ ] Configurações no início dos arquivos
- [ ] Execução direta: `python arquivo.py`
- [ ] Dados completos salvos para análise

### ✅ Funcionais:  
- [ ] Pipeline completo funciona end-to-end
- [ ] Análises visuais completas
- [ ] Performance vs benchmarks
- [ ] Compatibilidade com código _old/

---

## 9) NOTAS CRÍTICAS

**NUNCA ESQUECER:**
1. Este arquivo é a BIBLIA - sempre consultar antes de mudanças
2. Teory.txt é o LIVRO SAGRADO das teorias matemáticas  
3. Códigos principais devem ser fluxogramas legíveis
4. Salvar tudo que pode ser útil para análise futura
5. Manter compatibilidade com implementação original _old/

**SEMPRE LEMBRAR:**
- Configurações intuitivas no início dos arquivos
- Prints minimalistas no terminal
- Estrutura organizada e documentada  
- Implementação robusta e eficiente
