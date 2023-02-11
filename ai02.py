import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
import os
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from PIL import Image
import numpy as np
import keras

def on_closing():
    exit()

if __name__ == '__main__':
    strup = tk.Tk()
    strup.title("Entrainement de l'IA...")
    strup.geometry("720x480")
    label = tk.Label(strup, text="Entrainement de l'IA à chaque lancement... NE PAS FERMER L'APPLICATION !!! (Temps moyen: 2-3 minutes)")
    label.pack()

    classify_button = tk.Button(text="Lancement", command=strup.quit)
    classify_button.pack()

    strup.protocol("WM_DELETE_WINDOW", on_closing)
    strup.mainloop()

# Paramétrage de l'apprentissage
batch_size = 32
num_classes = 2
epochs = 1

# Répertoire contenant le modèle
model_dir = "C:/Users/cleme/OneDrive - student.helmo.be/AI"
model_path = os.path.join(model_dir, "chat_vs_chien_model.h5")

# Préparation des données d'entraînement
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    directory=r"C:\Users\cleme\OneDrive - student.helmo.be\AI\train",
    target_size=(224,224),
    batch_size=batch_size,
    class_mode="categorical")

val_generator = val_datagen.flow_from_directory(
    directory=r"C:\Users\cleme\OneDrive - student.helmo.be\AI\validation",
    target_size=(224,224),
    batch_size=batch_size,
    class_mode="categorical")

def train_model():
    # Entraînement du modèle
    model.fit(
        train_generator, # générateur d'images pour l'entraînement
        steps_per_epoch=len(train_generator), # nombre d'itérations par époque (un époque = une passe complète sur l'ensemble de données)
        epochs=epochs, # nombre d'époques configurable ci-dessus...
        validation_data=val_generator, # générateur d'images pour la validation
        validation_steps=len(val_generator)) # nombre d'itérations pour la validation
    status = "Coeur créé, vous pouvez clore la fenêtre et relancer le programme afin de mettre à jour les informations."
    print(status)
    pass

# Vérification de la présence d'un modèle existant
if os.path.exists(model_path):
    status = "Le coeur de l'IA a été trouvé, inititalisation du programme... Programe chargé !"
    print(status)
    model = keras.models.load_model(model_path)
    train_model()
else:
    # Définition du modèle
    status = "Le coeur de l'IA n'a pas été trouvé, création d'un nouveau coeur... NE PAS FERMER LA PAGE !!! (Temps moyen: 2-3 minutes)"
    print(status)
    model = Sequential([
        Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(224,224,3)),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(64, (3,3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    # Compilation du modèle
    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
    train_model()
    model.save(os.path.join(model_dir, 'chat_vs_chien_model.h5'))

# Préparation des données de test
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    directory=r"C:\Users\cleme\OneDrive - student.helmo.be\AI\test",
    target_size=(224,224),
    batch_size=batch_size,
    class_mode="binary")


def test_model(image_path):
    model = keras.models.load_model(model_path)
    test_image = Image.open(image_path)
    test_image = test_image.resize((224, 224))
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)

    if result[0][0] == 1:
        prediction = 'C\'est un chat'
    else:
        prediction = 'C\'est un chien'
    return prediction


def select_image():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    return file_path


def classify_image():
    image_path = select_image()
    prediction = test_model(image_path)
    print(prediction)
    label = tk.Label(app, text=prediction)
    label.pack()


if __name__ == '__main__':
    app = tk.Tk()
    app.title("Classification d'images")
    app.geometry("720x50")
    nope_button = tk.Button(text="Classifier une image", command=classify_image)
    nope_button.pack()

    label = tk.Label(app, text=status)
    label.pack()
    app.mainloop()