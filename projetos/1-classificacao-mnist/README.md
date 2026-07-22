# Projeto 1 — Classificação MNIST

## 💻 O Desafio Técnico

Desenvolva um **modelo de Visão Computacional** capaz de **classificar dígitos manuscritos (0-9)**, e posteriormente **otimize-o para execução em dispositivos Edge**.

O foco não é apenas obter alta acurácia, mas **compreender o fluxo completo**:

**treinamento → validação → salvamento → conversão → otimização**

## 🎯 Conjunto de Dados

Dataset **MNIST**, disponível diretamente via `tf.keras.datasets.mnist` (não é necessário download manual).

## ✅ Requisitos Obrigatórios

### Etapa 1 — Treinamento do Modelo (`train_model.py`)

Implemente:

- Carregamento do dataset MNIST via TensorFlow
- **Split explícito treino/validação** (ex: `validation_split` ou um split manual)
- Construção de uma CNN com:
  - **3 a 4 blocos convolucionais** (`Conv2D` + `BatchNormalization` + `MaxPooling2D`)
  - Camada de `Dropout` antes da saída, para regularização
- Treinamento com **early stopping** baseado na perda de validação (`EarlyStopping`)
- Exibição da **acurácia de validação final** no terminal
- Salvamento do modelo treinado em formato Keras (`model.h5`)

### Etapa 2 — Otimização do Modelo (`optimize_model.py`)

Implemente:

- Carregamento do `model.h5` treinado
- Conversão para **TensorFlow Lite** (`model.tflite`)
- Aplicação de uma técnica de otimização (ex: **Dynamic Range Quantization**)

### Etapa 3 — Inferência com o Modelo Otimizado (`run_inference.py`)

Implemente:

- Carregamento especificamente do **`model.tflite`** (o artefato de edge — não
  o `model.h5`) usando `tf.lite.Interpreter`
- Execução de inferência em pelo menos **5 amostras** do conjunto de teste
- Exibição no terminal, para cada amostra, da classe **predita** vs. a classe **real**

> 💡 Essa etapa existe porque uma métrica agregada (accuracy) pode esconder
> problemas que só aparecem olhando exemplos individuais. Também é o teste mais
> próximo do uso real em produção: carregar o artefato de edge e classificar
> uma entrada por vez.

**Objetivo:** reduzir o tamanho do modelo, mantendo desempenho adequado para aplicações de Edge AI.

## 📂 Estrutura da Pasta

⚠️ Não altere os nomes dos arquivos.

```
projetos/1-classificacao-mnist/
├── train_model.py         # ✏️ Treinamento do modelo
├── optimize_model.py      # ✏️ Conversão e otimização
├── run_inference.py       # ✏️ Inferência de exemplo com o modelo otimizado
├── requirements.txt       # 📄 Dependências do projeto
├── model.h5               # 🤖 Gerado por você — deve ser commitado
├── model.tflite           # ⚡ Gerado por você — deve ser commitado
└── README.md               # 📝 Este arquivo (também usado como relatório)
```

## ⚠️ Restrições e Considerações de Engenharia

- Entrada do modelo: imagens 28x28, 1 canal (grayscale), normalizadas em [0, 1]
- CNN simples — evite arquiteturas muito profundas
- Não utilize modelos pré-treinados
- Número de épocas limitado (ex: até 15, com early stopping)
- Treinamento apenas em CPU

## ⚖️ Critérios de Avaliação

- **Funcionalidade** — execução correta dos scripts e geração dos arquivos `.h5` e `.tflite`
- **Qualidade do modelo** — acurácia de validação consistente com o esperado para o dataset
- **Edge AI** — conversão correta para `.tflite` com técnica de otimização aplicada
- **Documentação** — preenchimento adequado do relatório abaixo

---

## 📝 Relatório do Candidato

👤 **Nome Completo: Thiago Roberto de Lima Ribeiro**

### 1️⃣ Resumo da Arquitetura do Modelo

Foi desenvolvida uma Rede Neural Convolucional (CNN) enxuta composta por **3 blocos convolucionais**:
- **Blocos 1 e 2:** Cada um contém uma camada `Conv2D` (com 32 e 64 filtros de 3x3, respectivamente), seguida por `BatchNormalization` para estabilização da distribuição de entrada das camadas e `MaxPooling2D` (2x2) para redução espacial.
- **Bloco 3:** Contém uma camada `Conv2D` (64 filtros, 3x3) e `BatchNormalization`, omitindo o *pooling* para preservar a resolução espacial residual antes da etapa densa.

A etapa de classificação é composta por uma camada `Flatten`, uma camada totalmente conectada (`Dense`) de 64 neurônios com ativação ReLU, e uma camada de `Dropout` com taxa de 50% (0.5) para forte regularização contra *overfitting*. A saída utiliza uma camada `Dense` de 10 neurônios com ativação Softmax para classificação multiclasse dos dígitos (0-9).

**Estratégia de Validação e Treino:**
O treinamento utilizou um *split* explícito de validação de 20% (`validation_split=0.2`). Foi aplicado o callback `EarlyStopping` monitorando a perda de validação (`val_loss`) com paciência de 3 épocas e restauração dos melhores pesos (`restore_best_weights=True`), garantindo eficiência no treinamento executado exclusivamente em CPU.

### 2️⃣ Bibliotecas Utilizadas

- **TensorFlow (v2.12):** Construção da CNN, callbacks de treinamento, conversão e runtime do interpretador TFLite.
- **NumPy (v2.4.6):** Manipulação, normalização e adequação dimensional dos tensores das imagens.
- **OS / System (Nativo):** Manipulação de caminhos de arquivos e ambiente de execução.

### 3️⃣ Técnica de Otimização do Modelo

Em `optimize_model.py`, foi aplicada a **Quantização de Faixa Dinâmica** (*Dynamic Range Quantization*) fornecida nativamente pelo TensorFlow Lite (`tf.lite.Optimize.DEFAULT`). 

Essa técnica quantiza os pesos do modelo de ponto flutuante de 32 bits (`float32`) para inteiros de 8 bits (`int8`) durante a conversão. Em tempo de execução (inferência), as ativações são quantizadas dinamicamente. Isso proporciona uma redução substancial no tamanho do arquivo final e menor consumo de memória RAM/cache em dispositivos de borda (*Edge AI*), mantendo a perda de acurácia em níveis insignificantes.

### 4️⃣ Resultados Obtidos

- **Acurácia de Validação Final (Teste):** ~98.94%
- **Tamanho do arquivo `model.h5` (Keras original):** ~1.2 MB
- **Tamanho do arquivo `model.tflite` (Otimizado):** ~105 KB *(Redução de aproximadamente 91% no tamanho do artefato)*

### 5️⃣ Comentários Adicionais (Opcional)

- **Decisões de Engenharia:** A inclusão do `BatchNormalization` acelerou significativamente a convergência do treino na CPU, exigindo poucas épocas para atingir alta acurácia.
- **Ambiente e Reprodutibilidade:** O desenvolvimento foi padronizado utilizando **Dev Containers (Docker)** no macOS. Isso garantiu um ambiente isolado em Python 3.10 com todas as dependências resolvidas sem conflitos no sistema operacional hospedeiro, alinhado com as exigências de execução automatizada do GitHub Actions.

### 6️⃣ Exemplo de Inferência

```text
Amostra 1: predito=7 | real=7
Amostra 2: predito=2 | real=2
Amostra 3: predito=1 | real=1
Amostra 4: predito=0 | real=0
Amostra 5: predito=4 | real=4
Amostra 6: predito=1 | real=1
Amostra 7: predito=4 | real=4
Amostra 8: predito=9 | real=9

O artefato otimizado .tflite obteve 100% de acerto no lote de amostras avaliado. A quantização não comprometeu a capacidade de generalização do modelo em imagens individuais de teste, demonstrando prontidão para deploy em hardware embarcado.
