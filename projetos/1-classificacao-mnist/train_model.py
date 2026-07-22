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

#carregando o dataset MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Adicionando o canal de cor (1 canal para grayscale)
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# Normalizando os pixels para o intervalo [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Inicializando o modelo sequencial
modelo = models.Sequential()

# Bloco 1
modelo.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
modelo.add(layers.BatchNormalization())
modelo.add(layers.MaxPooling2D((2, 2)))

# Bloco 2
modelo.add(layers.Conv2D(64, (3, 3), activation='relu'))
modelo.add(layers.BatchNormalization())
modelo.add(layers.MaxPooling2D((2, 2)))

# Bloco 3
modelo.add(layers.Conv2D(64, (3, 3), activation='relu'))
modelo.add(layers.BatchNormalization())
# Não usamos MaxPooling aqui porque a imagem já está muito pequena

# Preparação para a saída
modelo.add(layers.Flatten())
modelo.add(layers.Dense(64, activation='relu'))

# Dropout antes da saída
modelo.add(layers.Dropout(0.5))

# Saída (10 classes: dígitos de 0 a 9)
modelo.add(layers.Dense(10, activation='softmax'))

# Compilando o modelo
modelo.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Configurando o Early Stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3, # Espera 3 épocas sem melhora para parar
    restore_best_weights=True
)

# Treina o modelo com split de validação automático de 20%
print("Iniciando o treinamento...")
history = modelo.fit(
    x_train, y_train,
    epochs=15, # Limite máximo
    validation_split=0.2, # Regra obrigatória do split
    callbacks=[early_stopping],
    batch_size=64
)

# Avalia o conjunto de teste para pegar a acurácia final
loss, accuracy = modelo.evaluate(x_test, y_test, verbose=0)
print(f"\n✅ Acurácia de Validação Final (Teste): {accuracy * 100:.2f}%")

# Salva o modelo como "model.h5"
modelo.save('model.h5')
print("✅ Modelo salvo como model.h5")