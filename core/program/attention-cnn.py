import tensorflow as tf
from tensorflow.keras import layers, models

def spatial_attention_block(input_feature):
    # Spatial Attention Block
    avg_pool = tf.reduce_mean(input_feature, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(input_feature, axis=-1, keepdims=True)
    concat = layers.concatenate([avg_pool, max_pool], axis=-1)
    attention = layers.Conv2D(1, (7, 7), padding='same', activation='sigmoid')(concat)
    return layers.multiply([input_feature, attention])

def build_attention_cnn(input_shape=(224, 224, 3)):
    # Backbone ResNet50 pre-trained on ImageNet
    base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    base_model.trainable = True
    
    # Connect backbone to Spatial Attention
    x = base_model.output
    x = spatial_attention_block(x)
    
    # Classification layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs=base_model.input, outputs=output)
    return model
