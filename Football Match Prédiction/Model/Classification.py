import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Input, Conv2D, Flatten, Dense, concatenate, Conv2DTranspose, Dropout
import os
from zipfile import ZipFile
import pandas as pd
import numpy as np
from keras import optimizers, regularizers
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# # nous supposons que la base d'image se trouve dans le répertoire "IA-Vision" de votre Drive
# zip_file_path = "C:/Users/LD/Desktop/Projet P5/Matrices_Team-20240117T084539Z-001.zip"
# extracted_folder_path = "C:/Users/LD/Desktop/Projet P5/Matrices_Team"
#
# with ZipFile(zip_file_path, 'r') as zip:
#     zip.extractall(extracted_folder_path)
#     print('Extraction terminée')
#
# folder_path = extracted_folder_path
#
# print("Contenu du répertoire extrait :", os.listdir(extracted_folder_path))

def load_data():
    csv_path = "C:/Users/LD/Desktop/Projet P5/Team_game_result.csv"
    return pd.read_csv(csv_path)


def shape_matrice():
    # Dossier contenant les matrices d'équipes
    folder_path = "C:/Users/LD/Desktop/Projet P5/Matrices_Team/Matrices_Team"

    # Liste des fichiers dans le dossier
    team_files = os.listdir(folder_path)

    # Parcourir chaque fichier et afficher la forme de la matrice
    for team_file in team_files:
        if team_file.endswith(".npy"):
            team_matrix = np.load(os.path.join(folder_path, team_file))
            matrix_shape = team_matrix.shape
    print(f"Matrice {team_file} - Shape : {matrix_shape}")


def preprocess_data():
    folder_path = "C:/Users/LD/Desktop/Projet P5/Matrices_Team/Matrices_Team"
    X_combined_matrices = []
    Y_results = []

    for index, row in load_data().iterrows():
        home_team_name = row['Home_Team']
        away_team_name = row['Away_Team']
        result = row['Result']

        home_team_matrix = np.load(os.path.join(folder_path, f"{home_team_name}.npy"))
        away_team_matrix = np.load(os.path.join(folder_path, f"{away_team_name}.npy"))
        home_team_matrix = home_team_matrix / 20
        away_team_matrix = away_team_matrix / 20

        combined_matrix = np.concatenate([home_team_matrix, away_team_matrix], axis=0)

        X_combined_matrices.append(combined_matrix)
        Y_results.append(result)

    # Convertir les listes en tableaux NumPy
    X_combined = np.array(X_combined_matrices)
    Y = np.array(Y_results)

    print("X_combined shape:", X_combined.shape)
    print("Y shape:", Y.shape)
    return X_combined, Y


def model_test():
    # Taille des matrices d'équipes
    input_shape = (6, 4, 20)

    # Modèle CNN
    model = Sequential([
        Conv2D(32, (2, 2), strides=(2, 2), activation='relu', padding='same', input_shape=input_shape),
        Conv2D(64, (2, 2), strides=(1, 1), activation='relu', padding='same'),
        Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same'),
        Flatten(),
        Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        Dense(3, activation='softmax')
    ])

    # Compiler le modèle
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'],
                  weighted_metrics=[])

    # Afficher le résumé du modèle
    model.summary()
    return model


def model_conv2dTranspose():
    # Taille des matrices d'équipes
    input_shape = (6, 4, 20)

    # Modèle CNN
    model2 = Sequential([
        Conv2DTranspose(32, (2, 2), strides=(2, 3), activation='relu', padding='same', input_shape=input_shape),
        Conv2DTranspose(32, (3, 3), strides=(1, 1), activation='relu', input_shape=input_shape),
        Conv2DTranspose(32, (2, 2), strides=(2, 2), activation='relu', padding='same', input_shape=input_shape),
        Conv2DTranspose(32, (2, 2), strides=(2, 2), activation='relu', padding='same', input_shape=input_shape),
        Conv2DTranspose(32, (2, 2), strides=(2, 2), activation='relu', padding='same', input_shape=input_shape),
        Conv2DTranspose(3, (2, 2), strides=(2, 2), activation='relu', padding='same', input_shape=input_shape),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(3, activation='softmax')
    ])

    # Compiler le modèle
    model2.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

    # Afficher le résumé du modèle
    model2.summary()
    return model2


def train_model(X_combined, Y, model, epochs=25, batch_size=32, test_size=0.2):
    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, Y_train, Y_test = train_test_split(X_combined, Y, test_size=0.2, random_state=42)

    # Calculer les poids des classes
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(Y_train), y=Y_train)
    class_weight_dict = dict(enumerate(class_weights))
    print("class_weight_dict", class_weight_dict)
    # Entraîner le modèle
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,
                        validation_split=0.2)

    # Évaluation du modèle sur l'ensemble de test
    loss, accuracy = model.evaluate(X_test, Y_test)
    print(f"Loss: {loss}, Accuracy: {accuracy}")

    return model, history


def plot_confusion_matrix(model, X_test, Y_test):
    # Prédiction sur l'ensemble de test
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    print(y_pred[:15])

    # Calculer la matrice de confusion
    conf_matrix = confusion_matrix(Y_test, y_pred_classes)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['AwayWin', 'Draw', 'HomeWin'],
                yticklabels=['AwayWin', 'Draw', 'HomeWin'])
    plt.xlabel('Prédiction')
    plt.ylabel('Vraie classe')
    plt.title('Matrice de Confusion')
    plt.show()


def plot_training_history(history):
    # Créer un DataFrame à partir de l'historique
    history_df = pd.DataFrame(history.history)

    # Tracer les courbes d'entraînement
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history_df['accuracy'], label='Train Accuracy')
    plt.plot(history_df['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_df['loss'], label='Train Loss')
    plt.plot(history_df['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


def load_data_Forme():
    csv_path = "C:/Users/LD/Desktop/Projet P5/Team_game_result10-1_forme.csv"
    return pd.read_csv(csv_path)


def preprocess_data_Forme():
    folder_path = "C:/Users/LD/Desktop/Projet P5/Matrices_Team/Matrices_Team"
    X_combined_matrices = []
    Y_results = []

    for index, row in load_data_Forme().iterrows():
        home_team_name = row['Home_Team']
        away_team_name = row['Away_Team']
        home_coef = row['Forme_dom']
        away_coef = row['Forme_exte']
        result = row['Result']

        home_team_matrix = np.load(os.path.join(folder_path, f"{home_team_name}.npy"))
        away_team_matrix = np.load(os.path.join(folder_path, f"{away_team_name}.npy"))
        home_team_matrix = (home_team_matrix * (1 + home_coef)) / 20
        away_team_matrix = (away_team_matrix * (1 + away_coef)) / 20

        combined_matrix = np.concatenate([home_team_matrix, away_team_matrix], axis=0)

        X_combined_matrices.append(combined_matrix)
        Y_results.append(result + 1)

    # Convertir les listes en tableaux NumPy
    X_combined = np.array(X_combined_matrices)
    Y = np.array(Y_results)

    print("X_combined shape:", X_combined.shape)
    print("Y shape:", Y.shape)
    return X_combined, Y
