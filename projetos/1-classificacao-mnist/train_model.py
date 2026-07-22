import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

# ---------------------------------------------------------------------------
# Projeto 1 — Classificação MNIST
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o dataset MNIST via tf.keras.datasets.mnist
#   2. Normalizar as imagens para [0, 1] e ajustar o shape para (28, 28, 1)
#   3. Separar um conjunto de validação (ex: validation_split ou split manual)
#   4. Construir uma CNN com 3-4 blocos Conv2D + BatchNormalization + MaxPooling2D,
#      seguida de Dropout antes da camada de saída (10 classes, softmax)
#   5. Treinar com EarlyStopping monitorando a perda de validação
#   6. Exibir a acurácia de validação final no terminal
#   7. Salvar o modelo treinado como "model.h5"
# ---------------------------------------------------------------------------

# Carregamento do dataset MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Pré-processamento e adequação de tensores
# Redimensiona para (28, 28, 1) para adequação ao formato esperado por camadas Conv2D (H, W, C)
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# Normalização de intensidade de pixel para a escala [0.0, 1.0] (estabiliza o gradiente)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Definição da Arquitetura CNN
modelo = models.Sequential()

# Bloco 1: Extração inicial de mapas de características
modelo.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
modelo.add(layers.BatchNormalization())
modelo.add(layers.MaxPooling2D((2, 2)))

# Bloco 2: Padrões intermediários e redução espacial
modelo.add(layers.Conv2D(64, (3, 3), activation='relu'))
modelo.add(layers.BatchNormalization())
modelo.add(layers.MaxPooling2D((2, 2)))

# Bloco 3: Extração profunda (sem pooling para preservar resolução espacial residual)
modelo.add(layers.Conv2D(64, (3, 3), activation='relu'))
modelo.add(layers.BatchNormalization())

# Preparação para a saída
modelo.add(layers.Flatten())
modelo.add(layers.Dense(64, activation='relu'))

# Regularização por Dropout (50%) para prevenir overfitting antes da camada densa final
modelo.add(layers.Dropout(0.5))

# Camada de Saída Softmax para distribuição de probabilidades entre 10 classes
modelo.add(layers.Dense(10, activation='softmax'))

# Compilação do modelo
modelo.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Configuração de Callbacks e Parada Antecipada
# Interrompe o treino se a perda de validação estagnar, restaurando os melhores pesos
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3, 
    restore_best_weights=True
)

# Treinamento do modelo com split de validação automático de 20%
print("Iniciando o treinamento...")
history = modelo.fit(
    x_train, y_train,
    epochs=15, 
    validation_split=0.2,
    callbacks=[early_stopping],
    batch_size=64
)

# Avaliação e Exportação do Artefato Keras
loss, accuracy = modelo.evaluate(x_test, y_test, verbose=0)
print(f"\n✅ Acurácia de Validação Final (Teste): {accuracy * 100:.2f}%")

# Salvando o modelo base (.h5) para posterior conversão/otimização
modelo.save('model.h5')
print("✅ Modelo salvo como model.h5")