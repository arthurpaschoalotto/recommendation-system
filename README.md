# 🎬 Recomendador de Filmes com Streamlit

Sistema de recomendação de filmes baseado em regressão supervisionada (`XGBoost`), afinidade por gêneros, similaridade vetorial e penalização de popularidade. A interface é feita com `Streamlit`.

---

## 🔧 1. Criação do ambiente Conda

Clone o repositório e acesse a pasta do projeto:

```bash
git clone https://github.com/seu-usuario/recomendador-filmes.git
cd recomendador-filmes
```

Crie um novo ambiente Conda com as dependências definidas no `environment.yml`:

```bash
conda env create -f environment.yml
```

Ative o ambiente:

```bash
conda activate sis-recomendacao
```

## ▶️ 2. Executar a aplicação Streamlit

Após ativar o ambiente e instalar as dependências, execute:

```bash
streamlit run app.py
```

A aplicação abrirá automaticamente no navegador (ex: `http://localhost:8501`).

> ✅ **Pré-requisitos esperados no diretório do projeto**:
> - `model_xgboost.pkl`: modelo de regressão treinado com `XGBRegressor`
> - `dataset.parquet`: conjunto de dados com os filmes e colunas de engenharia
> - `features.pkl`: lista com as colunas usadas na inferência (`feature_cols`)

---

## 💡 3. Como usar o sistema

### Passo a passo:

1. **Selecione de 5 a 10 filmes** que você gosta na lista apresentada.
   - Isso serve como base para entender seu gosto por gêneros e época.

2. Clique em **"🔍 Gerar Recomendações"**.

3. O sistema irá analisar seus filmes favoritos e recomendar títulos com base em:
   - Similaridade de gênero e subgênero
   - Ano de lançamento aproximado
   - Predição de nota via modelo supervisionado
   - Penalização por popularidade (sugestão de obras menos mainstream)

4. As recomendações aparecerão com:
   - Título
   - Plataformas onde o filme está disponível
   - Uma tag informando os gêneros com base no seu gosto

5. Você pode marcar filmes como **"✅ Já assisti"** para evitá-los em próximas recomendações.

---

## 📁 Estrutura esperada do projeto

```
recomendador-filmes/
├── app.py
├── dataset.parquet
├── features.pkl
├── model_xgboost.pkl
├── environment.yml
└── README.md
```

---

## ✨ Exemplo de mensagem na interface

> 🎯 Aqui estão suas recomendações personalizadas:  
> 📌 **Para você que gosta de filmes de:** *Action, Adventure, Thriller*

---

## 🧪 Tecnologias utilizadas

- `Python 3.10+`
- `pandas`, `scikit-learn`, `xgboost`, `lightgbm`, `catboost`
- `streamlit`
- `joblib`, `parquet` para persistência de dados e modelos

---