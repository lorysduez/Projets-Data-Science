from Classification import load_data, model_test
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
import os
from Regression import count_team_occurrences, load_data_regression, count_team_occurrences_away, plot_goals_averages
import matplotlib.pyplot as plt

app = Flask(__name__)


def load_team_matrix(team_name):
    matrix_path = f"C:/Users/LD/Desktop/Projet P5/Matrices_Team/Matrices_Team/{team_name}.npy"
    return np.load(matrix_path)


def preprocess_data_appli(home_team_name, away_team_name):
    folder_path = "C:/Users/LD/Desktop/Projet P5/Matrices_Team/Matrices_Team"
    X_combined_matrices = []

    home_team_matrix = np.load(os.path.join(folder_path, f"{home_team_name}.npy"))
    away_team_matrix = np.load(os.path.join(folder_path, f"{away_team_name}.npy"))
    home_team_matrix = home_team_matrix / 20
    away_team_matrix = away_team_matrix / 20

    combined_matrix = np.concatenate([home_team_matrix, away_team_matrix], axis=0)

    X_combined_matrices.append(combined_matrix)
    X_combined = np.array(X_combined_matrices)

    return X_combined


def predict_goals(home_team_matrix, away_team_matrix, model):
    X_combined = preprocess_data_appli(home_team_matrix, away_team_matrix)
    y_pred = model.predict(X_combined)
    rounded_result = np.round(y_pred).astype(int)
    return rounded_result


def predict_cotes(home_team_matrix, away_team_matrix, model):
    X_combined = preprocess_data_appli(home_team_matrix, away_team_matrix)
    y_pred = model.predict(X_combined)
    inverse_values = [[1 / x for x in row] for row in y_pred]
    rounded_cotes = [round(cote, 2) for cote in inverse_values[0]]

    results = ["Victoire Équipe Domicile", "Victoire Équipe Extérieure", "Match Nul"]
    resultats = [f"{label}: {cote}" for label, cote in zip(results, rounded_cotes)]

    return resultats


def generate_pie_chart(team_name, away_team, dataframe):
    zero_results_count = count_team_occurrences(team_name, dataframe, result_value=0.0)
    one_results_count = count_team_occurrences(team_name, dataframe, result_value=1.0)
    two_results_count = count_team_occurrences(team_name, dataframe, result_value=2.0)
    total_matches_count = count_team_occurrences(team_name, dataframe)

    zero_results_count_away = count_team_occurrences_away(away_team, dataframe, result_value=0.0)
    one_results_count_away = count_team_occurrences_away(away_team, dataframe, result_value=1.0)
    two_results_count_away = count_team_occurrences_away(away_team, dataframe, result_value=2.0)
    total_matches_count_away = count_team_occurrences_away(away_team, dataframe)

    labels = ['Victoire', 'Défaite', 'Match nul']
    labels_away = ['Défaite', 'Victoire', 'Match nul']
    sizes = [zero_results_count, one_results_count, two_results_count]
    sizes_away = [zero_results_count_away, one_results_count_away, two_results_count_away]

    plt.pie(sizes, labels=labels, autopct=lambda p: f'{p:.1f}% ({int(p * total_matches_count / 100)})', startangle=90,
            colors=['green', 'red', 'gray'])
    plt.title(f"{team_name}")
    plt.suptitle(f"Nombre total de matchs à domicile : {total_matches_count}")
    plt.savefig('static/pie_chart_home.png',
                bbox_inches='tight')  # Vous pouvez ajuster le chemin selon votre structure de projet
    plt.close()

    plt.pie(sizes_away, labels=labels_away, autopct=lambda p: f'{p:.1f}% ({int(p * total_matches_count_away / 100)})',
            startangle=90,
            colors=['red', 'green', 'gray'])
    plt.title(f"{away_team}")
    plt.suptitle(f"Nombre total de matchs à extérieur : {total_matches_count_away}")

    # Enregistrez le diagramme dans un fichier image ou dans un objet BytesIO pour l'intégrer dans votre modèle HTML
    plt.savefig('static/pie_chart_away.png',
                bbox_inches='tight')  # Vous pouvez ajuster le chemin selon votre structure de projet
    plt.close()


@app.route('/', methods=['GET', 'POST'])
def index():
    dataframe = load_data()

    liste_equipes = dataframe['Home_Team'].unique().tolist()
    liste_equipes_ext = dataframe['Away_Team'].unique().tolist()

    model = tf.keras.models.load_model("C:/Users/LD/Desktop/Projet P5/ModelClassification")
    model_regression = tf.keras.models.load_model("C:/Users/LD/Desktop/Projet P5/Model1")

    if request.method == 'POST':
        equipe_dom_selectionnee = request.form['equipe_dom_dropdown']
        equipe_ext_selectionnee = request.form['equipe_ext_dropdown']

        cotes = predict_cotes(equipe_dom_selectionnee, equipe_ext_selectionnee, model)
        goals_predictions = predict_goals(equipe_dom_selectionnee, equipe_ext_selectionnee, model_regression)
        generate_pie_chart(equipe_dom_selectionnee, equipe_ext_selectionnee, dataframe)
        plot_goals_averages(equipe_dom_selectionnee, equipe_ext_selectionnee, load_data_regression())
    else:
        equipe_dom_selectionnee = None
        equipe_ext_selectionnee = None
        cotes = []
        goals_predictions = []

    return render_template('index.html', liste_equipes=liste_equipes, liste_equipes_ext=liste_equipes_ext,
                           equipe_dom_selectionnee=equipe_dom_selectionnee,
                           equipe_ext_selectionnee=equipe_ext_selectionnee, cotes=cotes,
                           goals_predictions=goals_predictions)


if __name__ == '__main__':
    app.run(debug=True)
