# 🎯 TUTORIAL COMPLETO: Nasdaq Data Link (Dados Gratuitos de Commodities)

## ❌ Problema: Yahoo Finance não funciona

Testamos 18 commodities no Yahoo Finance e **TODAS falharam**. O Yahoo removeu acesso a dados de futuros em 2024.

## ✅ Solução: Nasdaq Data Link (ex-Quandl)

**Vantagens:**
- ✅ 100% gratuito
- ✅ Contratos contínuos CL1, CL2, ..., CL12
- ✅ Dados históricos completos
- ✅ API oficial e confiável
- ✅ 50 requisições/dia (suficiente para nosso uso)

---

## 📋 PASSO 1: Criar Conta (5 minutos)

### 1.1. Acesse o site
```
https://data.nasdaq.com/sign-up
```

### 1.2. Preencha o formulário
- **Email:** seu_email@gmail.com
- **First Name:** Seu nome
- **Last Name:** Seu sobrenome
- **Password:** Crie uma senha forte
- ✅ Marque: "I agree to the Terms of Use"
- Clique em **"SIGN UP"**

### 1.3. Confirme o email
- Abra seu email
- Procure mensagem de "Nasdaq Data Link"
- Clique no link de confirmação

---

## 📋 PASSO 2: Obter API Key (2 minutos)

### 2.1. Faça login
```
https://data.nasdaq.com/sign-in
```

### 2.2. Vá para Account Settings
- Clique no seu nome (canto superior direito)
- Clique em **"Account Settings"**
- OU acesse diretamente: https://data.nasdaq.com/account/profile

### 2.3. Copie a API Key
- Na página de perfil, procure por **"API KEY"**
- Você verá algo como: `xYz123AbC456DeF789...`
- Clique em **"COPY"** ou selecione e copie manualmente
- ⚠️ **IMPORTANTE:** Guarde essa chave em local seguro!

---

## 📋 PASSO 3: Configurar no Projeto (3 minutos)

### 3.1. Criar arquivo .env
No terminal, na pasta do projeto:

```bash
cd ~/Documents/ItauQuant
nano .env
```

### 3.2. Adicionar a chave
Cole esta linha (substituindo pela sua chave):

```bash
NASDAQ_DATA_LINK_API_KEY=xYz123AbC456DeF789
```

**Salvar e sair:**
- Pressione `Ctrl + O` (salvar)
- Pressione `Enter` (confirmar)
- Pressione `Ctrl + X` (sair)

### 3.3. Instalar biblioteca Python
```bash
source venv/bin/activate.fish
pip install nasdaq-data-link
```

---

## 📋 PASSO 4: Testar se Funciona

### 4.1. Criar script de teste
```python
# test_nasdaq.py
import nasdaqdatalink
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("NASDAQ_DATA_LINK_API_KEY")

if not api_key:
    print("❌ API KEY não encontrada no .env")
    exit(1)

nasdaqdatalink.ApiConfig.api_key = api_key

print("🔍 Testando conexão...")
try:
    df = nasdaqdatalink.get('CHRIS/CME_CL1', start_date='2024-01-01', end_date='2024-01-31')
    print(f"✅ SUCESSO! Baixados {len(df)} dias de dados")
    print(f"📊 Colunas: {list(df.columns)}")
    print(f"💰 Preço médio: ${df['Settle'].mean():.2f}")
except Exception as e:
    print(f"❌ ERRO: {e}")
```

### 4.2. Rodar teste
```bash
python test_nasdaq.py
```

**Resultado esperado:**
```
🔍 Testando conexão...
✅ SUCESSO! Baixados 22 dias de dados
📊 Colunas: ['Open', 'High', 'Low', 'Last', 'Change', 'Settle', 'Volume', 'Previous Day Open Interest']
💰 Preço médio: $73.45
```

---

## 📋 PASSO 5: Usar no Pipeline

### 5.1. Baixar dados de WTI (8 tenores)
```bash
python src/DownloadsData_backup.py \
  --source chris \
  --start-date 2020-01-01 \
  --end-date 2024-12-31 \
  --num-tenors 8
```

### 5.2. Rodar backtest
```bash
python src/Backtester.py \
  --dataset-root data/free_data/CL_2020_2024_free \
  --train-days 150 --test-days 2 \
  --method MLE --sizing vol_target
```

---

## 🎯 Commodities Disponíveis no Nasdaq

### Energéticas (CHRIS/CME_*)
```python
'CHRIS/CME_CL1'  # WTI Crude Oil (front month)
'CHRIS/CME_CL2'  # WTI (2nd month)
# ... até CL12

'CHRIS/CME_NG1'  # Natural Gas
'CHRIS/CME_RB1'  # RBOB Gasoline
'CHRIS/CME_HO1'  # Heating Oil
```

### Metais (CHRIS/CME_*)
```python
'CHRIS/CME_GC1'  # Gold
'CHRIS/CME_SI1'  # Silver
'CHRIS/CME_HG1'  # Copper
```

### Agrícolas (CHRIS/CME_*)
```python
'CHRIS/CME_C1'   # Corn
'CHRIS/CME_S1'   # Soybeans
'CHRIS/CME_W1'   # Wheat
'CHRIS/ICE_KC1'  # Coffee
'CHRIS/ICE_SB1'  # Sugar
```

---

## ⚠️ Limites da Versão Gratuita

- **50 requisições/dia**
- Para 8 tenores = 8 requisições por execução
- Pode rodar ~6 vezes por dia
- **Solução:** Cache os dados baixados

---

## 🐛 Troubleshooting

### Erro: "API key not found"
```bash
# Verifique se .env está na pasta certa
cat .env

# Deve mostrar:
NASDAQ_DATA_LINK_API_KEY=sua_chave_aqui
```

### Erro: "Rate limit exceeded"
- Você passou de 50 requisições hoje
- Espere até meia-noite (horário UTC)
- OU use dados já baixados

### Erro: "Dataset not found"
- Verifique se o código está correto: `CHRIS/CME_CL1`
- Liste disponíveis em: https://data.nasdaq.com/data/CHRIS

---

## 📊 Estrutura de Dados

### O que você recebe:
```
Date        | Open   | High   | Low    | Last   | Settle | Volume  | Open Interest
2024-01-02  | 71.65  | 72.35  | 71.45  | 72.11  | 72.11  | 145320  | 234567
2024-01-03  | 72.20  | 73.10  | 71.90  | 72.95  | 72.95  | 156789  | 235890
...
```

### O que o pipeline usa:
- **Settle:** Preço de fechamento oficial
- **Date:** Index do DataFrame
- **8 tenores:** CL1, CL2, ..., CL8 (8 meses seguintes)

---

## ✅ Checklist Final

Antes de rodar o pipeline, certifique-se:

- [ ] Conta criada no Nasdaq Data Link
- [ ] Email confirmado
- [ ] API Key copiada
- [ ] Arquivo `.env` criado com a chave
- [ ] Biblioteca `nasdaq-data-link` instalada
- [ ] Teste executado com sucesso
- [ ] Backup do `DownloadsData_backup.py` disponível

---

## 🚀 Próximos Passos

1. **Configure conforme o tutorial acima**
2. **Teste com comando:**
   ```bash
   python test_nasdaq.py
   ```
3. **Baixe dados históricos:**
   ```bash
   python src/DownloadsData_backup.py --source chris --start-date 2020-01-01 --end-date 2024-12-31 --num-tenors 8
   ```
4. **Rode backtest rápido:**
   ```bash
   python src/Backtester.py --dataset-root data/free_data/CL_2020_2024_free --train-days 150 --test-days 2 --method MLE --sizing vol_target
   ```

---

## 📞 Suporte

**Documentação oficial:** https://data.nasdaq.com/tools/api
**Datasets disponíveis:** https://data.nasdaq.com/data/CHRIS

**Problemas?** Leia a seção Troubleshooting acima!
