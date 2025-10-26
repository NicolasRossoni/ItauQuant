# Architecture.md

> **Objetivo**: este README é o “prompt de implementação” para gerar **toda a estrutura de código em Python** do projeto.  
> Use-o junto com o **documento Teory.txt** (teoria e fórmulas). Aqui tratamos **apenas** de engenharia de software: arquivos, funções, I/O, formatos, gráficos e passos de execução.

---

## 1) Princípios gerais (estilo e qualidade de código)

- **Documentação impecável**: cada função deve ter docstring clara (contexto, parâmetros com tipos e shapes, retorno com tipos e shapes, exceções, exemplos).  
- **Nomes autoexplicativos**: variáveis e funções com nomes óbvios, sem abreviações crípticas.  
- **Modularidade**: cada arquivo expõe **uma função pública principal**; auxiliares ficam internas e **começam com `_`**.  
- **I/O determinístico**: funções recebem tudo que precisam como argumentos e retornam resultados claros; efeitos colaterais (leitura/escrita em disco) apenas quando explicitamente indicado.  
- **Eficiência e clareza**: código enxuto, sem redundâncias, mas com comentários estratégicos explicando decisões e fórmulas implementadas.  
- **Tipos e validações**: use `typing` (e.g., `np.ndarray`, `pd.DataFrame`) e valide shapes/NaNs.  
- **Logs**: `logging` em nível INFO/DEBUG para fases críticas (ajuste do estado/EM/MLE, métricas de risco, sizing, geração de gráficos).

---

## 2) Pipeline — o que a pipeline faz

1. **ComputeModelForward**: calibra o modelo (Kalman MLE/EM), filtra estados e calcula o vetor de forwards teóricos $F_{\text{modelo}}(t^*, T_{1:M})$ via **fórmula fechada** (vide LaTeX).  
2. **PrepareTradingInputs**: a partir de $F_{\text{mkt}}$ e $F_{\text{modelo}}$ (e histórico), computa **mispricing**, **matriz de risco $\Sigma$**, **limites**, **limiares** e **fricções**.  
3. **TradeEngine**: transforma desvio+risco+limites em **ordens** (buy/sell/qty), além de **signals** e **target weights**.
4. **Geração de dados sintéticos** para desenvolvimento: `GenerateFakeData.py` cria um dataset sintético formatado exatamente como os dados reais futuros.
5. **Teste ponta a ponta teórico**: `TestingTheoryPipeline.py` executa todo o pipeline em dados sintéticos e salva **gráficos** e **artefatos**.

---

## 3) Estrutura de diretórios e arquivos

```

ItauQuant/
├─ src/
│  ├─ ComputeModelForward.py       # Função pública: ComputeModelForward(...)
│  ├─ PrepareTradingInputs.py      # Função pública: PrepareTradingInputs(...)
│  ├─ TradeEngine.py               # Função pública: TradeEngine(...)
│  ├─ DownloadsData.py             # (placeholder) baixar dados reais da API (próxima etapa)
│  ├─ GenerateFakeData.py          # Gera dataset sintético compatível (mesmo formato dos reais)
│  ├─ TestingTheoryPipeline.py     # Teste ponta a ponta com dados sintéticos e gráficos
│  └─ Main.py                      # (opcional) orquestração manual/rápida durante dev
├─ config/
│  └─ default.yaml                 # Parâmetros padrão (kalman, risk, sizing, limits, thresh, etc.)
├─ data/
│  ├─ fakedata/
│  │  └─ <dataset_name>/           # Ex.: wti_synth_01
│  │     ├─ F_mkt.csv
│  │     ├─ ttm.csv
│  │     ├─ S.csv                  # opcional
│  │     └─ costs.csv              # opcional (tick/fee por tenor)
│  └─ <outros datasets reais>/
│     └─ ... (mesmo contrato de arquivos)
├─ images/
│  └─ fake_data_analysis/          # Saída de gráficos do teste teórico
└─ requirements.txt

````

**Contrato de dados (CSV):**
- `F_mkt.csv`: índice = `date` (YYYY-MM-DD), colunas = `tenor_1 ... tenor_M` (preços a termo).  
- `ttm.csv`: mesma grade (datas × tenores), valores em **anos** (ACT/365).  
- `S.csv` (opcional): série com `date` e coluna `S`.  
- `costs.csv` (opcional): por tenor (`tenor`, `tick_value`, `fee_per_contract`, ...).

---

## 4) Especificação de cada arquivo/módulo

### 4.1 `src/ComputeModelForward.py`

**Função pública**  
```python
def ComputeModelForward(
    F_mkt: "np.ndarray|pd.DataFrame",     # shape [T, M]
    ttm: "np.ndarray|pd.DataFrame",       # shape [T, M], em anos
    S: "np.ndarray|pd.Series|None",       # shape [T] (opcional)
    cfg: dict,                            # {'method': 'MLE'|'EM', 'kalman': {...}}
    t_idx: int                            # índice temporal alvo (ex.: -1)
) -> dict:
    """
    Retorna:
      {
        'Theta': dict,                    # {'kappa','sigma_X','sigma_Y','rho','mu'}
        'state_t': np.ndarray,            # shape [2] -> (X_hat_t*, Y_hat_t*)
        'F_model_t': np.ndarray,          # shape [M] -> F_model(t*, T_1..T_M)
        'state_path': np.ndarray|None,    # shape [T, 2] (opcional)
        'F_model_path': np.ndarray|None   # shape [T, M] (opcional)
      }
    """
````

**Passos (internos)**

* `_validate_and_cast_inputs(...)`: garantir shapes, converter DataFrame→ndarray, checar NaNs/máscaras.
* `_fit_states_mle_statsmodels(...)` **ou** `_fit_states_em_pykalman(...)`: escolher por `cfg['method']`.
* `_extract_state_at(state_path, t_idx)`: retorna $(\hat X_{t^*},\hat Y_{t^*})$.
* `_compute_forward_closed_form(X_hat, Y_hat, Theta, ttm_row)`: implementa **fórmula fechada** para vetor de tenores $T_{1:M}$.
* (opcional) `_compute_forward_path(...)` se `cfg['kalman']['save_path']=True`.

**Observações**

* **Bibliotecas**: `statsmodels.tsa.statespace` (MLE) e/ou `pykalman` (EM).
* `cfg['kalman']` pode conter: chutes de `Theta`, `R` (medição), tolerâncias, `max_iter`, etc.
* O método deve **logar** convergência, valores finais e métricas básicas.

---

### 4.2 `src/PrepareTradingInputs.py`

**Função pública**

```python
def PrepareTradingInputs(
    F_mkt_t: "np.ndarray",                # shape [M] no tempo t*
    F_model_t: "np.ndarray",              # shape [M] no tempo t*
    ttm_t: "np.ndarray",                  # shape [M]
    cost: "np.ndarray|pd.Series|None",    # shape [M] (opcional)
    cfg: dict,
    F_mkt_hist: "np.ndarray|None" = None, # shape [T, M] (opcional)
    F_model_hist: "np.ndarray|None" = None# shape [T, M] (opcional)
) -> dict:
    """
    Retorna:
      {
        'mispricing': np.ndarray,  # [M] -> DeltaF = F_model_t - F_mkt_t
        'Sigma': np.ndarray,       # [M, M] -> matriz de risco (cov. retornos)
        'limits': np.ndarray,      # [M] -> limites por tenor (posição/notional/VAR)
        'thresh': np.ndarray,      # [M] -> limiares (z_in/out ou absolutos)
        'frictions': dict          # custos efetivos por tenor
      }
    """
```

**Passos (internos)**

* `_compute_mispricing(F_model_t, F_mkt_t)` → $\Delta F$.
* `_estimate_covariance(...)`:

  * se `F_mkt_hist`/`F_model_hist` disponíveis: usar janela `cfg['risk']['lookback']` com retornos (ou resíduos).
  * senão: fallback para $\Sigma$ diagonal com vol estimada (log-retornos simples) + **shrinkage** de Ledoit–Wolf (se disponível).
* `_derive_limits_and_thresholds(cfg, Sigma, cost)` → vetores por tenor.
* `_build_frictions(cost, cfg)` → consolidar `tick`, `fee`, `slippage`.

---

### 4.3 `src/TradeEngine.py`

**Função pública**

```python
def TradeEngine(
    mispricing: "np.ndarray",             # [M]
    Sigma: "np.ndarray",                  # [M, M]
    limits: "np.ndarray",                 # [M]
    thresh: "np.ndarray|dict",            # [M] ou {'z_in','z_out'}
    frictions: dict,
    method: str = "vol_target",           # "vol_target" | "qp"
    topK: "int|None" = None,
    w_prev: "np.ndarray|None" = None,     # [M]
    cfg: dict | None = None
) -> dict:
    """
    Retorna:
      {
        'signals': np.ndarray,   # [M] em {-1,0,+1}
        'target_w': np.ndarray,  # [M] posições-alvo normalizadas
        'orders': list           # [(maturity_idx, 'BUY'|'SELL', qty), ...]
      }
    """
```

**Passos (internos)**

* `_zscore(mispricing, Sigma)` → z por tenor (usar desvio-padrão marginal).
* `_select_topK_by_abs_z(z, topK)` (opcional).
* **Histerese**: `_apply_hysteresis(z, thresh)` → sinais discretos {-1,0,+1}.
* **Dimensionamento**:

  * `_size_positions_vol_target(signals, Sigma, cfg)` → normaliza risco para alvo de volatilidade.
  * ou `_optimize_qp(mispricing, Sigma, limits, frictions, w_prev, cfg)` → QP estilo mean–variance com penalidades $\ell_1$ e turnover (usar `cvxopt` ou `scipy.optimize`).
* `_apply_limits(target_w, limits)` → cortar por tenor.
* `_build_orders(target_w, w_prev, cfg)` → lista executável.

---

### 4.4 `src/DownloadsData.py` (placeholder)

* Responsável por **baixar dados reais** da API, normalizar para o **contrato de dados** acima e salvar em `data/<dataset_name>/`.
* **Agora**: deixe vazio, apenas a casca do arquivo. Será preenchido após validação com sintético.

---

### 4.5 `src/GenerateFakeData.py`

**Função pública**

```python
def GenerateFakeDataset(
    dataset_name: str,                # ex.: "wti_synth_01"
    T: int,                           # nº datas
    M: int,                           # nº tenores
    seed: int = 42,
    out_root: str = "data/fakedata"
) -> dict:
    """
    Gera dados sintéticos estilo Schwartz–Smith:
      - Simula X_t (OU) e Y_t (Browniano com deriva)
      - Calcula F_model(t, T_i) (fórmula fechada; usar ttm)
      - Cria F_mkt = F_model + ruído (lognormal ou aditivo calibrado por tenor/ttm)
      - Gera grade de ttm coerente (anos)

    Salva CSVs em: {out_root}/{dataset_name}/
      F_mkt.csv, ttm.csv, S.csv (opcional), costs.csv (opcional)
    Retorna paths salvos.
    """
```

**Passos (internos)**

* `_simulate_states(T, cfg_or_theta)` → séries $X_t, Y_t$.
* `_build_ttm_grid(T, M)` → ttm decrescente/rolado (ACT/365).
* `_forward_closed_form_path(X, Y, Theta, ttm)` → $F_{\text{model path}}$ [T,M].
* `_inject_market_noise(F_model_path, ttm, scheme='lognormal')` → $F_{\text{mkt}}$.
* `_save_csvs(out_dir, F_mkt, ttm, S, costs)`.

---

### 4.6 `src/TestingTheoryPipeline.py`

**Função pública**

```python
def RunTheoryPipeline(
    dataset_root: str,                    # ex.: "data/fakedata/wti_synth_01"
    t_idx: int = -1,                      # tempo alvo t*
    save_dir: str = "images/fake_data_analysis",
    method: str = "MLE",                  # "MLE" | "EM"
    sizing: str = "vol_target",           # "vol_target" | "qp"
    topK: "int|None" = None,
    cfg_path: str = "config/default.yaml"
) -> dict:
    """
    Executa o pipeline completo em dados sintéticos e salva gráficos.
    Retorna dicionário com artefatos e caminhos salvos (csv/png).
    """
```

**Passos (detalhados e diretos)**

1. **Carregar dados**: `F_mkt.csv`, `ttm.csv`, `S.csv` (se existir).
2. **Executar** `ComputeModelForward(F_mkt, ttm, S, cfg, t_idx)` → obter `Theta`, `state_t`, $F_{\text{model\_t}}$, e (se habilitado) $F_{\text{model\_path}}$, `state_path`.
3. **Executar** `PrepareTradingInputs(F_mkt[t_idx], F_model_t, ttm[t_idx], cost_vec_or_none, cfg, F_mkt, F_model_path)` → `mispricing`, `Sigma`, `limits`, `thresh`, `frictions`.
4. **Executar** `TradeEngine(...)` → `signals`, `target_w`, `orders`.
5. **Salvar artefatos**:

   * `orders_t.csv` (lista de ordens do tempo $t^*$).
   * `signals_t.csv`, `target_w_t.csv`, `mispricing_t.csv`.
6. **Gerar e salvar **10 gráficos** (apenas `savefig`, **não** `show`) em `images/fake_data_analysis/`:

   * `01_forward_model_vs_market_t{t}.png` — Linha por tenor: $F_{\text{modelo}}$ vs $F_{\text{mkt}}$ em $t^*$.
   * `02_mispricing_bar_t{t}.png` — Barras de $\Delta F$ por tenor.
   * `03_zscores_bar_t{t}.png` — Barras de z por tenor (se calculados na TradeEngine).
   * `04_covariance_heatmap.png` — Heatmap de $\Sigma$ (com rótulos de tenores).
   * `05_state_trajectories.png` — Séries $\hat X_t$ e $\hat Y_t$ (se `state_path` disponível).
   * `06_forward_surface_mispricing.png` — Heatmap de $F_{\text{modelo}}-F_{\text{mkt}}$ no tempo×tenor.
   * `07_sample_forward_timeseries_tenor1.png` — Série temporal $F_{\text{modelo}}$ vs $F_{\text{mkt}}$ para um tenor representativo.
   * `08_target_weights_bar_t{t}.png` — Barras de `target_w` por tenor.
   * `09_signals_scatter_t{t}.png` — Dispersão de `signals` ({-1,0,+1}) por tenor.
   * `10_residuals_hist_t{t}.png` — Histograma de $\Delta F$ ou dos resíduos escolhidos para risco.
     *(Todos com títulos objetivos, eixos com unidades, legendas legíveis e grids discretos.)*

---

## 5) Configuração (YAML)

Arquivo: `config/default.yaml` (exemplo mínimo)

```yaml
kalman:
  method: "MLE"            # "MLE" | "EM"
  save_path: true
  max_iter: 200
  tol: 1e-6
  init_params:
    kappa: 1.0
    sigma_X: 0.3
    sigma_Y: 0.2
    rho: 0.3
    mu: 0.00
  R: 0.01                  # ruído de medição (escalar ou diag por obs)

risk:
  source: "returns"        # "returns" | "residuals"
  lookback: 60
  shrinkage: true

sizing:
  method: "vol_target"     # "vol_target" | "qp"
  vol_target: 0.10
  qp:
    gamma: 5.0
    lambda_l1: 0.0
    lambda_turnover: 0.1

limits:
  leverage: 3.0
  per_tenor_cap: 0.3

thresh:
  z_in: 1.5
  z_out: 0.5
  topK: 4

costs:
  default_tick_value: 10.0
  default_fee: 2.0
```

---

## 6) Dependências (Python 3.11+)

`requirements.txt` (sugestão)

```
numpy
pandas
scipy
statsmodels
pykalman
matplotlib
pyyaml
cvxopt     # se usar QP; caso contrário, use scipy.optimize
```

---

## 7) Execução — fluxo com dados sintéticos

1. **Gerar dataset fake**

```bash
python src/GenerateFakeData.py --dataset-name wti_synth_01 --T 1500 --M 8
```

2. **Rodar pipeline teórico**

```bash
python src/TestingTheoryPipeline.py \
  --dataset-root data/fakedata/wti_synth_01 \
  --t-idx -1 \
  --method MLE \
  --sizing vol_target
```

**Resultados esperados:**

* `images/fake_data_analysis/` populada com os 10 gráficos.
* CSVs: `orders_t.csv`, `signals_t.csv`, `target_w_t.csv`, `mispricing_t.csv`.
* Logs com parâmetros finais do Kalman e métricas de risco.

---

## 8) `src/Main.py` (opcional, dev)

* Script simples para chamar as 3 funções públicas em sequência no dataset escolhido (string única do caminho) e imprimir/inspecionar saídas rapidamente.

---

## 9) Conformidade com o Documento LaTeX

* **ComputeModelForward** deve implementar **exatamente** a expressão fechada de $F(t,T)$ e a estrutura do espaço de estados conforme o documento **DEFINITIVO**.
* Unidades (ttm em anos), shapes e ordenação (datas em linhas, tenores em colunas) devem respeitar o contrato descrito aqui.

---

## 10) Checklist de validação rápida

* [ ] `GenerateFakeDataset` salva CSVs com as dimensões certas.
* [ ] `ComputeModelForward` roda (MLE/EM), retorna $F_{\text{model\_t}}$ coerente e (opcional) paths.
* [ ] `PrepareTradingInputs` retorna `Sigma` SPD (checar autovalores > 0 após shrinkage).
* [ ] `TradeEngine` retorna `signals`, `target_w`, `orders` **não vazios** quando $|z|>z_{\text{in}}$.
* [ ] 10 gráficos salvos com títulos/legendas adequados.
* [ ] Sem `plt.show()` — apenas `savefig(...)`.
* [ ] Logs informativos e sem exceções não tratadas.

---

## 11) Notas finais

* **DownloadsData.py** será preenchido na próxima etapa (integração real com API).
* Mantenha o código **idempotente**: rodadas repetidas devem gerar os mesmos artefatos, salvo quando houver aleatoriedade controlada por `seed`.
* Qualquer nova função auxiliar deve iniciar com `_` e respeitar o contrato de tipos/shapes descritos.