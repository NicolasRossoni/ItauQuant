# 🔑 Como Configurar a API CME Group

Este guia mostra como obter credenciais da CME Group DataMine API e configurar o projeto.

---

## 📝 Passo 1: Criar Conta na CME Group

1. **Acesse**: https://www.cmegroup.com/market-data/datamine-historical-data.html

2. **Clique em "Request DataMine Access"** ou "Sign Up"

3. **Preencha o formulário**:
   - Nome completo
   - Email corporativo/acadêmico (preferencial)
   - Empresa/Instituição
   - Razão do uso (pesquisa, trading, análise)

4. **Aguarde aprovação** (geralmente 1-2 dias úteis)
   - Você receberá um email de confirmação
   - Algumas contas requerem aprovação manual da CME

---

## 🔐 Passo 2: Obter Credenciais API

### Opção A: API DataMine (Recomendada)

1. **Login**: https://datamine.cmegroup.com/
2. **Navegue para**: `Account Settings` → `API Keys`
3. **Clique em**: `Generate New API Key`
4. **Copie e salve**:
   - `API Key` (público)
   - `API Secret` (privado, mostrado apenas uma vez!)

⚠️ **IMPORTANTE**: Guarde o API Secret em local seguro. Ele não pode ser recuperado depois.

### Opção B: FTP (Alternativa)

Se não tiver acesso à API, você pode usar FTP:
- Host: `ftp.cmegroup.com`
- User: fornecido no email de aprovação
- Password: definida no cadastro

---

## ⚙️ Passo 3: Configurar o Projeto

### 3.1 Copiar Template

```bash
cd /path/to/ItauQuant
cp .env.example .env
```

### 3.2 Editar .env

Abra o arquivo `.env` e cole suas credenciais:

```bash
CME_API_KEY=abc123def456ghi789  # Sua API Key
CME_API_SECRET=xyz789uvw456rst123  # Seu API Secret
```

### 3.3 Verificar Permissões

```bash
# Garantir que .env não seja commitado
cat .gitignore | grep .env
# Deve mostrar: .env
```

---

## 🧪 Passo 4: Testar Conexão

### 4.1 Instalar Dependências

```bash
source venv/bin/activate
pip install python-dotenv requests
```

### 4.2 Testar Download

```bash
python src/DownloadsData.py \
  --product CL \
  --start-date 2023-01-01 \
  --end-date 2023-12-31 \
  --num-tenors 8
```

**Saída esperada (sucesso)**:
```
=== Baixando dados da CME Group API ===
Produto: CL
Período: 2023-01-01 a 2023-12-31
Consultando API: https://datamine.cmegroup.com/api/v1/...
API retornou 2520 registros
✓ Dados salvos em: data/real_data/CL_2023_2023/
```

**Saída esperada (fallback)**:
```
AVISO: API CME não acessível ou credenciais inválidas
Gerando dados SINTÉTICOS para teste do pipeline
✓ Dados sintéticos salvos em: data/real_data/CL_2023_2023/
```

---

## 📊 Produtos Disponíveis

| Código | Nome | Descrição |
|--------|------|-----------|
| **CL** | WTI Crude Oil | Petróleo bruto americano (recomendado) |
| **NG** | Natural Gas | Gás natural Henry Hub |
| **BZ** | Brent Crude Oil | Petróleo bruto europeu |
| **HO** | Heating Oil | Óleo combustível |
| **RB** | RBOB Gasoline | Gasolina |

---

## ❓ Troubleshooting

### Erro 401: Unauthorized
```
✗ Credenciais inválidas
```
**Solução**: Verifique se copiou corretamente API Key e Secret no `.env`

### Erro 403: Forbidden
```
✗ Sua conta não tem permissão para este produto
```
**Solução**: 
- Verifique seu plano CME DataMine
- Alguns produtos requerem assinatura paga
- Use produto de teste (ex: CL)

### Erro: No module named 'dotenv'
```
✗ ModuleNotFoundError: No module named 'dotenv'
```
**Solução**:
```bash
source venv/bin/activate
pip install python-dotenv requests
```

### Dados Sintéticos em Vez de Reais
```
⚠️ AVISO: Gerando dados SINTÉTICOS
```
**Causa**: API não respondeu (credenciais inválidas ou rede)
**Solução**:
1. Verifique credenciais no `.env`
2. Teste conexão: `curl https://datamine.cmegroup.com/`
3. Verifique proxy/firewall

---

## 🔒 Segurança

### ✅ Boas Práticas

- ✅ `.env` está no `.gitignore`
- ✅ Nunca commite credenciais no git
- ✅ Use `.env.example` como template (sem valores reais)
- ✅ Rotacione API keys periodicamente
- ✅ Use variáveis de ambiente em produção

### ❌ Evite

- ❌ Hardcoded credentials no código
- ❌ Compartilhar `.env` via email/chat
- ❌ Commitar `.env` no repositório
- ❌ Usar mesmas credenciais em múltiplos projetos

---

## 📚 Referências

- **CME DataMine Docs**: https://www.cmegroup.com/confluence/display/EPICSANDBOX/CME+DataMine+API
- **API Reference**: https://datamine.cmegroup.com/docs/
- **Product Codes**: https://www.cmegroup.com/markets/products.html
- **Pricing Plans**: https://www.cmegroup.com/market-data/datamine-historical-data/pricing.html

---

## 💡 Dicas

1. **Uso Educacional**: Mencione uso acadêmico/pesquisa para acelerar aprovação
2. **Free Tier**: CME oferece trial gratuito com dados limitados
3. **Alternativas**: Se CME não aprovar, considere Quandl, Alpha Vantage, ou Yahoo Finance
4. **Cache Local**: Baixe dados uma vez e reutilize para economizar requisições

---

## ✅ Checklist de Setup

- [ ] Conta CME criada e aprovada
- [ ] API Key e Secret obtidos
- [ ] Arquivo `.env` criado com credenciais
- [ ] Dependências instaladas (`python-dotenv`, `requests`)
- [ ] Teste executado com sucesso
- [ ] Dados salvos em `data/real_data/`

Pronto! Agora você pode baixar dados reais da CME e usar no pipeline. 🚀
