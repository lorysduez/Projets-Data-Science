import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Input, Conv2D, Flatten, Dense, concatenate, Conv2DTranspose, Dropout, MaxPooling2D
import os
from zipfile import ZipFile
import pandas as pd
import numpy as np
from keras import optimizers, regularizers
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from Classification import load_data


def load_data_regression():
    csv_path = "C:/Users/LD/Desktop/Projet P5/Team_game_result_regression.csv"
    return pd.read_csv(csv_path)


def preprocess_data_regression():
    folder_path = "C:/Users/LD/Desktop/Projet P5/Matrices_Team/Matrices_Team"
    X_combined_matrices = []
    Y_results = []

    for index, row in load_data_regression().iterrows():
        home_team_name = row['Home_Team']
        away_team_name = row['Away_Team']
        home_clubs_goals = row['Home_Clubs_Goals']
        away_clubs_goals = row['Away_Clubs_Goals']

        home_team_matrix = np.load(os.path.join(folder_path, f"{home_team_name}.npy"))
        away_team_matrix = np.load(os.path.join(folder_path, f"{away_team_name}.npy"))
        home_team_matrix = home_team_matrix / 20
        away_team_matrix = away_team_matrix / 20

        combined_matrix = np.concatenate([home_team_matrix, away_team_matrix], axis=0)

        X_combined_matrices.append(combined_matrix)
        Y_results.append((home_clubs_goals, away_clubs_goals))

    # Convertir les listes en tableaux NumPy
    X_combined = np.array(X_combined_matrices)
    Y = np.array(Y_results)

    print("X_combined shape:", X_combined.shape)
    print("Y shape:", Y.shape)
    return X_combined, Y


def model1_regression():
    # Taille des matrices d'équipes
    input_shape = (6, 4, 20)

    # Modèle CNN
    model = Sequential([
        Conv2D(32, (2, 2), strides=(2, 2), activation='relu', padding='same', input_shape=input_shape),
        # Dropout(0.5),
        Conv2D(64, (2, 2), strides=(1, 1), activation='relu', padding='same'),
        # Dropout(0.3),
        Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same'),
        # MaxPooling2D(),
        Flatten(),
        Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        Dense(2, activation='linear')
    ])

    # Compiler le modèle
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss='mean_squared_error',
                  metrics=['mse'],
                  weighted_metrics=[])

    # Afficher le résumé du modèle
    model.summary()
    return model


def model_conv2dTranspose_Regression():
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
        Dense(256, activation='linear'),
        Dense(2, activation='softmax')
    ])

    # Compiler le modèle
    model2.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mean_squared_error',
                   metrics=['mse'])

    # Afficher le résumé du modèle
    model2.summary()
    return model2


def train_model(X_combined, Y, model, epochs=25, batch_size=32, test_size=0.2):
    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, Y_train, Y_test = train_test_split(X_combined, Y, test_size=0.2, random_state=42)

    # Calculer les poids des classes
    # class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(Y_train), y=Y_train)
    # class_weight_dict = dict(enumerate(class_weights))
    # print("class_weight_dict", class_weight_dict)
    # Entraîner le modèle
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,
                        validation_split=0.2)

    # Évaluation du modèle sur l'ensemble de test
    loss, mse = model.evaluate(X_test, Y_test)
    print(f"Loss: {loss}, MSE: {mse}")

    return model, history


def plot_training_history_regression(history):
    history_df = pd.DataFrame(history.history)

    # Tracer les courbes d'entraînement
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 1, 1)
    plt.plot(history_df['mse'], label='Train mse')
    plt.plot(history_df['val_mse'], label='Validation mse')
    plt.title('Training and Validation mse')
    plt.xlabel('Epoch')
    plt.ylabel('mse')
    plt.legend()

    plt.show()


def plot_ypredVSy_test(y_pred, y_test, sample_size):
    y_pred_arrondie = np.round(y_pred)
    plt.figure(figsize=(18, 6))
    x = np.linspace(0, 5, 15)
    y = x
    sample_indices = np.random.choice(len(y_test), size=sample_size)

    "Vraies Valeurs et Prédictions"
    plt.subplot(1, 3, 1)
    plt.scatter(y_test[sample_indices, 0], y_test[sample_indices, 1], c='blue', label='Vraies valeurs', marker='o',
                s=100)
    plt.scatter(y_pred[sample_indices, 0], y_pred[sample_indices, 1], c='red', label='Prédictions', marker='x', s=100)
    for index in sample_indices:
        plt.plot([y_test[index, 0], y_pred[index, 0]], [y_test[index, 1], y_pred[index, 1]], color='gray',
                 linestyle='--')

    plt.title('Comparaison entre y_test et y_pred')
    plt.xlabel('Home_Clubs_Goals')
    plt.ylabel('Away_Clubs_Goals')
    plt.legend(loc='upper right')

    "Prédictions et Prédictions Corrigées"
    plt.subplot(1, 3, 2)
    plt.scatter(y_pred[sample_indices, 0], y_pred[sample_indices, 1], c='red', label='Prédictions', marker='x', s=100)
    plt.scatter(y_pred_arrondie[sample_indices, 0], y_pred_arrondie[sample_indices, 1], c='green',
                label='Prédictions corrigées', marker='o', edgecolors='black', linewidths=3, s=100)
    for index in sample_indices:
        plt.plot([y_pred[index, 0], y_pred_arrondie[index, 0]], [y_pred[index, 1], y_pred_arrondie[index, 1]],
                 color='green', linestyle='--')
    plt.title('Comparaison entre y_pred et y_pred_arrondie')
    plt.xlabel('Home_Clubs_Goals')
    plt.ylabel('Away_Clubs_Goals')
    plt.legend(loc='upper right')

    "Vraie Valeurs et Prédictions Corrigées"
    plt.subplot(1, 3, 3)
    plt.scatter(y_test[sample_indices, 0], y_test[sample_indices, 1], c='blue', label='Vraies valeurs', marker='o',
                s=100)
    plt.scatter(y_pred_arrondie[sample_indices, 0], y_pred_arrondie[sample_indices, 1], c='green',
                label='Prédictions corrigées', marker='o', edgecolors='black', linewidths=3, s=100)
    plt.plot(x, y)

    for index in sample_indices:
        plt.plot([y_test[index, 0], y_pred_arrondie[index, 0]], [y_test[index, 1], y_pred_arrondie[index, 1]],
                 color='green', linestyle='--')

    plt.title('Comparaison entre y_test et y_pred_arrondie')
    plt.xlabel('Home_Clubs_Goals')
    plt.ylabel('Away_Clubs_Goals')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


def preprocess_data_regression_vecteurs():
    folder_path = "C:/Users/LD/Desktop/Projet P5/Vecteurs_Team"
    X_combined_matrices = []
    Y_results = []

    for index, row in load_data_regression().iterrows():
        home_team_name = row['Home_Team']
        away_team_name = row['Away_Team']
        home_clubs_goals = row['Home_Clubs_Goals']
        away_clubs_goals = row['Away_Clubs_Goals']

        home_team_matrix = np.load(os.path.join(folder_path, f"{home_team_name}.npy"))
        away_team_matrix = np.load(os.path.join(folder_path, f"{away_team_name}.npy"))
        home_team_matrix = home_team_matrix / 20
        away_team_matrix = away_team_matrix / 20

        combined_matrix = np.concatenate([home_team_matrix, away_team_matrix], axis=0)

        X_combined_matrices.append(combined_matrix)
        Y_results.append((home_clubs_goals, away_clubs_goals))

    # Convertir les listes en tableaux NumPy
    X_combined = np.array(X_combined_matrices)
    Y = np.array(Y_results)

    print("X_combined shape:", X_combined.shape)
    print("Y shape:", Y.shape)
    return X_combined, Y


def model1_regression_vecteur():
    # Taille des matrices d'équipes
    input_shape = (440, 1)
    model = Sequential([
        Flatten(input_shape=(440, 1)),
        Dense(1048, activation='relu'),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(2, activation='linear', kernel_regularizer=regularizers.l2(0.01))
    ])

    # Compiler le modèle
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss='mean_squared_error',
                  metrics=['mse'],
                  weighted_metrics=[])

    # Afficher le résumé du modèle
    model.summary()
    return model


def count_team_occurrences(team_name, dataframe, result_value=None):
    # Filtre le DataFrame pour les occurrences du nom de l'équipe dans la colonne Home_Team
    team_results = dataframe[(dataframe['Home_Team'] == team_name)]

    # Si une valeur de résultat est spécifiée, filtre également par cette valeur
    if result_value is not None:
        team_results = team_results[team_results['Result'] == result_value]

    # Compte le nombre d'occurrences
    count = len(team_results)

    return count


def count_team_occurrences_away(team_name, dataframe, result_value=None):
    # Filtre le DataFrame pour les occurrences du nom de l'équipe dans la colonne Home_Team
    team_results = dataframe[(dataframe['Away_Team'] == team_name)]

    # Si une valeur de résultat est spécifiée, filtre également par cette valeur
    if result_value is not None:
        team_results = team_results[team_results['Result'] == result_value]

    # Compte le nombre d'occurrences
    count = len(team_results)

    return count


def calculate_average_home_goals(team_name, dataframe):
    # Filtre le DataFrame pour les occurrences du nom de l'équipe dans la colonne Home_Team
    team_results = dataframe[dataframe['Home_Team'] == team_name]

    # Calculer la moyenne des buts marqués à domicile
    average_goals = team_results['Home_Clubs_Goals'].mean()

    return average_goals


def calculate_average_away_goals(team_name, dataframe):
    # Filtre le DataFrame pour les occurrences du nom de l'équipe dans la colonne Away_Team
    team_results = dataframe[dataframe['Away_Team'] == team_name]

    # Calculer la moyenne des buts marqués à l'extérieur
    average_goals = team_results['Away_Clubs_Goals'].mean()

    return average_goals


def plot_goals_averages(home_team_name, away_team_name, dataframe):
    # Calculer la moyenne des buts à domicile
    home_average_goals = calculate_average_home_goals(home_team_name, dataframe)

    # Calculer la moyenne des buts à l'extérieur
    away_average_goals = calculate_average_away_goals(away_team_name, dataframe)

    # Créer un graphique à barres
    teams = [home_team_name, away_team_name]
    averages = [home_average_goals, away_average_goals]

    positions = np.array([0, 0.2])
    width = 0.1  # Largeur de chaque barre

    plt.bar(positions, averages, width, color=['blue', 'green'])
    plt.xlabel('Équipes')
    plt.ylabel('Moyenne des buts')
    plt.title('Moyenne des buts à domicile et à l\'extérieur')
    plt.xticks(positions, teams)  # Positionner les étiquettes des équipes sur l'axe des x
    plt.savefig('static/bar_graph_goals.png',
                bbox_inches='tight')  # Vous pouvez ajuster le chemin selon votre structure de projet
    plt.close()

# # Exemple d'utilisation
# data= load_data()  # Assurez-vous que load_data_season() renvoie le DataFrame
# team_name_to_check = "Paris Saint-Germain Football Club_2022"
# zero_results_count = count_team_occurrences(team_name_to_check, data, result_value=0.0)
# pourcentage_win_value = pourcentage_win(team_name_to_check, data)
# print(f"{zero_results_count} victoire")
# print(f"{round(pourcentage_win_value*100,1)}% de matchs gagnés")
#
# zero_results_count = count_team_occurrences(team_name_to_check, data, result_value=0.0)
# one_results_count = count_team_occurrences(team_name_to_check, data, result_value=1.0)
# two_results_count = count_team_occurrences(team_name_to_check, data, result_value=2.0)
# total_matches_count = count_team_occurrences(team_name_to_check, data)  # Nouvelle ligne
#
# # Créer un camembert
# labels = ['Victoire', 'Défaite', 'Match nul']
# sizes = [zero_results_count, one_results_count, two_results_count]
#
# # Ajouter le sous-titre et les valeurs en pourcentage avec le nombre total de matchs
# plt.pie(sizes, labels=labels, autopct=lambda p: f'{p:.1f}% ({int(p * total_matches_count / 100)})', startangle=90, colors=['green', 'red', 'gray'])
# plt.title(f"Répartition des résultats pour {team_name_to_check}")
# plt.suptitle(f"Nombre total de matchs à domicile : {total_matches_count}")  # Ajout du sous-titre
# plt.show()
