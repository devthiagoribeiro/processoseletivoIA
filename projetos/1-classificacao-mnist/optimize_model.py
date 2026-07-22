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

# carregando o modelo treinado
modelo = tf.keras.models.load_model('projetos/1-classificacao-mnist/model.h5')

#convertendo para TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(modelo)

# Aplicando a Otimização exigida
converter.optimizations = [tf.lite.Optimize.DEFAULT]

#convertendo o modelo para TFLite
tflite_model = converter.convert()

# Salvando o modelo otimizado
caminho_tflite = 'projetos/1-classificacao-mnist/model.tflite'
with open(caminho_tflite, 'wb') as f:
    f.write(tflite_model)