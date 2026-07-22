import tensorflow as tf
import os

# ---------------------------------------------------------------------------
# Projeto 1 — Otimização do Modelo (MNIST)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo treinado em "model.h5"
#   2. Converter para TensorFlow Lite usando tf.lite.TFLiteConverter
#   3. Aplicar uma técnica de otimização (ex: Dynamic Range Quantization,
#      via converter.optimizations = [tf.lite.Optimize.DEFAULT])
#   4. Salvar o resultado como "model.tflite"
# ---------------------------------------------------------------------------

# Carregamento do modelo treinado
modelo = tf.keras.models.load_model('model.h5')

# Inicialização do conversor TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(modelo)

# Aplicação de Otimização Edge
# Quantiza os pesos de float32 para int8 em tempo de execução, reduzindo o tamanho
# do arquivo com perda negligenciável de acurácia.
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Execução da conversão para TFLite
tflite_model = converter.convert()

# Salvamento do artefato otimizado como "model.tflite"
caminho_tflite = 'model.tflite'
with open(caminho_tflite, 'wb') as f:
    f.write(tflite_model)