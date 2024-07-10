#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 15:51:00 2024

@author: mathieu
"""

import numpy as np
import pandas as pd
import os 


Season = 2018
League = "IT1"


Game_played = pd.read_csv("Base de données/Lien Derly/games.csv")
I = ["game_id","competition_id","season","date","home_club_name","away_club_name","home_club_goals","away_club_goals"]
Game_played=Game_played[I]

Game_played = Game_played[Game_played["competition_id"] == League]
Game_played = Game_played[Game_played["season"] == Season]
G = Game_played.to_numpy()

Team = np.unique(G[:,4:5])

lien_arsenal = "Base de données/2020/GB1/StatsWholeTeamAfterScore/Arsenal.csv"
Arsenal = pd.read_csv(lien_arsenal)
Arsenal = Arsenal.to_numpy()



def Create_vecteur_one_team(path):
    Vecteur = np.zeros([220,1])
    Team_from_csv = pd.read_csv(path)
    Team_from_csv = Team_from_csv.to_numpy()[:,2:]
    for i in range(20):
        for j in range(11):
            Vecteur[i*11+j,0]=Team_from_csv[j,i]
    return Vecteur
vec = Create_vecteur_one_team(lien_arsenal)
path = "Base de données/" + str(Season) + "/" + League


def create_team_league_season(path):
    List_team = os.listdir(path + "/StatsWholeTeamAfterScore")
    
    # Récupérer la fin du chemin et la stocker dans league
    base_path, league = os.path.split(path)
    # Récupérer la fin de la base et la stocker dans année
    base_path , season = os.path.split(base_path)
    season=int(season)
    Game_played = pd.read_csv("Base de données/Lien Derly/games.csv")
    I = ["competition_id","season","home_club_name"]
    Game_played = Game_played[I]
    Game_played = Game_played[Game_played["competition_id"] == league]
    Game_played = Game_played[Game_played["season"] == season]
    Game_played = Game_played.to_numpy()
    Name_team = np.unique(Game_played[:,2:3]).tolist()
    for x in Name_team:
        for y in List_team:
            if y[:-4] in x :
                Vecteur = Create_vecteur_one_team(path+"/StatsWholeTeamAfterScore/"+y)
                List_team.remove(y)
                #Name_team.remove(x)
                with open("Base de données/Vecteurs_Team/"+x+"_"+f"{season}"+".npy", 'wb') as f:
                    np.save(f, Vecteur)
    print(List_team)
    return Name_team
    
n = create_team_league_season(path)
for x in range(2018,2023):
    create_team_league_season("Base de données/" + str(x) + "/" + "L1")
    create_team_league_season("Base de données/" + str(x) + "/" + "ES1")
    create_team_league_season("Base de données/" + str(x) + "/" + "FR1")
    create_team_league_season("Base de données/" + str(x) + "/" + "GB1")
    create_team_league_season("Base de données/" + str(x) + "/" + "IT1")
    
'''
M = Create_matrice_one_team("Base de données/2020/FR1/StatsWholeTeamAfterScore/Nîmes.csv")
with open("Base de données/Matrices_Team/"+"Nîmes Olympique"+"_"+f"{2020}"+".npy", 'wb') as f:
    np.save(f, M)

'''

with open("/Users/mathieu/Documents/IMT M2/Projet P5 - Foot Predictor/Base de données/Vecteurs à vérifier/Associazione Calcio Fiorentina_2022.npy", 'rb') as f:
    Vec = np.load(f)