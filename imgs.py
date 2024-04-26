#redes neuronales de convuluciÃ³n 

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras import regularizers


# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
#cambiar 64
classifier.add(Conv2D(256, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 256, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   rotation_range=30,  # Aumentar el rango de rotación
                                   width_shift_range=0.2,  # Cambio aleatorio en el ancho
                                   height_shift_range=0.2,  # Cambio aleatorio en la altura
                                   brightness_range=[0.5, 1.5],  # Cambio aleatorio de brillo
                                   )

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('/Users/cristhianrecalde/Downloads/dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('/Users/cristhianrecalde/Downloads/dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000//32,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000,
                         )


# Evaluación del modelo en el conjunto de prueba
test_loss, test_acc = classifier.evaluate(test_set, steps=len(test_set))

# Impresión de la eficiencia del modelo
print("Eficiencia del modelo en el conjunto de prueba:")
print("Pérdida (Loss): {:.4f}".format(test_loss))
print("Precisión (Accuracy): {:.2f}%".format(test_acc * 100))



#part3 
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
