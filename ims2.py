from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import DenseNet121
from keras.layers import GlobalAveragePooling2D
from keras.regularizers import l1

# Inicializar el modelo
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

for layer in base_model.layers:
    layer.trainable = False

classifier = Sequential()
classifier.add(base_model)
classifier.add(GlobalAveragePooling2D())
classifier.add(Dense(256, activation='relu'))
classifier.add(Dropout(0.5))  # Agregar dropout más agresivo
classifier.add(Dense(1, activation='sigmoid'))

# Compilar el modelo con optimizador Adam y tasa de aprendizaje ajustada
optimizer = Adam(lr=0.0001)
classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Generador de datos de entrenamiento y prueba con aumento de datos avanzado
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=30,  # Aumentar el rango de rotación
    width_shift_range=0.2,  # Cambio aleatorio en el ancho
    height_shift_range=0.2,  # Cambio aleatorio en la altura
    brightness_range=[0.5, 1.5],  # Cambio aleatorio de brillo
)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('/Users/cristhianrecalde/Downloads/dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('/Users/cristhianrecalde/Downloads/dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

# Entrenar el modelo con más épocas y monitorizar la precisión en el conjunto de validación
classifier.fit_generator(training_set,
                         steps_per_epoch=8000//32,
                         epochs=25,  # Aumentar el número de épocas
                         validation_data=test_set,
                         validation_steps=2000,
                         )

# Evaluar el modelo en el conjunto de prueba
test_loss, test_acc = classifier.evaluate(test_set, steps=len(test_set))

# Imprimir la eficiencia del modelo
print("Eficiencia del modelo en el conjunto de prueba:")
print("Pérdida (Loss): {:.4f}".format(test_loss))
print("Precisión (Accuracy): {:.2f}%".format(test_acc * 100))

# Predicción de una imagen de ejemplo
import numpy as np
from tensorflow.keras.preprocessing import image

test_image = image.load_img('/Users/cristhianrecalde/Downloads/WhatsApp Image 2024-04-25 at 17.32.28 (1).jpeg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

result = classifier.predict(test_image)

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print("Prediction:", prediction)
