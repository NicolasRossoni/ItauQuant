# 📊 Alternativas de Dados Gratuitas para WTI Futures

## 🔍 Análise: webscrapping.ipynb

**O que é:** Scraping da **B3/BMF** (Bolsa Brasileira)
- Extrai dados de ajustes do pregão da B3
- Exemplos: ICF (Ibovespa Futuro), DOL (Dólar Futuro), IND (Índice Futuro)

**❌ Por que NÃO serve para WTI:**
- B3 não negocia WTI (crude oil) americano
- WTI é negociado na **CME/NYMEX** (bolsa americana)
- Webscraping da CME é contra os termos de serviço e tecnicamente difícil

---

## 💡 O que o Yahoo Finance (yfinance) oferece

### ✅ O que FUNCIONA:
```python
import yfinance as yf
df = yf.download('CL=F', start='2024-01-01', end='2024-12-31')
```

**CL=F** = Contrato contínuo "front month" (mês mais próximo)
- ✅ Preços históricos diários
- ✅ OHLC (Open, High, Low, Close)
- ✅ Volume
- ✅ 100% gratuito, sem API key

### ❌ O que FALTA:
1. **Contratos específicos por mês:**
   - CLZ24 (Dezembro 2024)
   - CLF25 (Janeiro 2025)
   - CLG25 (Fevereiro 2025)
   - etc.

2. **Curva forward completa:**
   - Não tem os 8 tenores simultâneos
   - Apenas o front month

3. **Dados de expiração reais:**
   - Não sabe quando cada contrato expira

**Solução atual:** Criamos tenores sintéticos com contango de 0.4%/mês

---

## 🆓 ALTERNATIVA 1: Nasdaq Data Link (ex-Quandl)

### Como funciona:
```python
import nasdaqdatalink
nasdaqdatalink.ApiConfig.api_key = 'SUA_CHAVE_GRATUITA'

# Baixa contratos contínuos CL1, CL2, ..., CL8
df = nasdaqdatalink.get('CHRIS/CME_CL1', start_date='2024-01-01')
```

### ✅ Vantagens:
- Contratos contínuos CL1...CL12 (próximos 12 meses)
- Dados históricos de settlement
- Open Interest
- API key gratuita (limite de 50 calls/dia)

### ❌ Limitações:
- Ainda são contratos "contínuos" (não exatos)
- Precisa cadastro no site
- Limite de requisições

### 📝 Como obter chave:
1. Acesse: https://data.nasdaq.com/sign-up
2. Crie conta gratuita
3. Copie API key do seu perfil
4. Adicione no .env: `NASDAQ_DATA_LINK_API_KEY=sua_chave_aqui`

### 🔧 Já está implementado!
O código antigo tinha suporte para isso:
```bash
python src/DownloadsData_backup.py \
  --source chris \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --num-tenors 8
```

---

## 🆓 ALTERNATIVA 2: EIA (Energy Information Administration)

### API oficial do governo americano:
```python
import requests
url = "https://api.eia.gov/v2/petroleum/pri/fut/data/"
params = {
    "api_key": "SUA_CHAVE_GRATUITA",
    "frequency": "daily",
    "data[0]": "value",
    "facets[product][]": "RCLC1",  # WTI front month
    "sort[0][column]": "period",
    "sort[0][direction]": "desc",
}
```

### ✅ Vantagens:
- Dados oficiais do governo dos EUA
- 100% gratuito
- API key sem limite

### ❌ Limitações:
- Atualização pode ser atrasada
- Formato complexo

---

## 📊 COMPARAÇÃO FINAL

| Fonte | Gratuita? | Curva completa? | Setup | Qualidade |
|-------|-----------|-----------------|-------|-----------|
| **Yahoo Finance (CL=F)** | ✅ Sim | ❌ Não (só front) | Fácil | Básica |
| **Nasdaq Data Link** | ✅ Sim* | ✅ CL1-CL12 | Média | Boa |
| **EIA API** | ✅ Sim | ⚠️ Parcial | Difícil | Oficial |
| **CME DataMine** | ❌ Não | ✅ Completa | Complexa | Excelente |
| **Webscraping B3** | N/A | N/A | N/A | Não serve |

*Limite de 50 calls/dia na versão gratuita

---

## 🎯 RECOMENDAÇÃO

### Para seu caso (estudo/desenvolvimento):

**Opção 1 (Atual):** Yahoo Finance + Contango Sintético
- ✅ Já funciona
- ✅ Zero configuração
- ✅ Adequado para testar modelo Schwartz-Smith
- ❌ Curva forward não é 100% real

**Opção 2 (Melhor gratuita):** Nasdaq Data Link (CHRIS)
- ✅ Dados melhores que Yahoo
- ✅ Contratos contínuos reais
- ⚠️ Precisa cadastro + API key
- ⚠️ Limite de 50 requisições/dia

**Opção 3 (Ideal, mas paga):** CME DataMine
- ✅ Dados exatos da fonte oficial
- ✅ Contratos individuais por mês
- ❌ Custa ~$100-500/mês
- ❌ Setup complexo

---

## 🚀 Próximos Passos

Se quiser usar Nasdaq Data Link:

1. **Criar conta gratuita:**
   ```bash
   # Acesse: https://data.nasdaq.com/sign-up
   ```

2. **Instalar biblioteca:**
   ```bash
   pip install nasdaq-data-link
   ```

3. **Adicionar chave no .env:**
   ```bash
   echo "NASDAQ_DATA_LINK_API_KEY=sua_chave" >> .env
   ```

4. **Usar o script backup:**
   ```bash
   python src/DownloadsData_backup.py \
     --source chris \
     --start-date 2024-01-01 \
     --end-date 2024-12-31 \
     --num-tenors 8
   ```

---

## ❓ FAQ

**P: Por que não fazer webscraping da CME?**
R: É contra os termos de serviço, legalmente arriscado, e tecnicamente difícil (JavaScript dinâmico, captchas, rate limiting).

**P: Nasdaq Data Link é confiável?**
R: Sim! Era o Quandl, foi comprado pela Nasdaq. Usado por milhares de quantitative traders.

**P: A curva sintética do Yahoo invalida meu modelo?**
R: Não! O modelo Schwartz-Smith calibra os parâmetros com base nos dados. A curva sintética serve para testar a lógica do código. Para produção, use dados pagos.
