#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 18:28:44 2023

@author: mathieu
"""

import numpy as np
import pandas as pd
import os 


Season = 2022
League = "IT1"


Game_played = pd.read_csv("Base de données/Lien Derly/games.csv")
I = ["game_id","competition_id","season","date","home_club_name","away_club_name","home_club_goals","away_club_goals"]
Game_played=Game_played[I]

Game_played = Game_played[Game_played["competition_id"] == League]
Game_played = Game_played[Game_played["season"] == Season]
G = Game_played.to_numpy()

Team = np.unique(G[:,4:5])
'''
lien_arsenal = "Base de données/Saison 2020-2021/Premier League/StatsWholeTeamAfterScore_2020-2021/Arsenal.csv"
Arsenal = pd.read_csv(lien_arsenal)
Arsenal = Arsenal.to_numpy()
'''
lien = "Base de données/Saison 2020-2021/Premier League/StatsWholeTeamAfterScore/Arsenal.csv"
#def Create_Team_season_league(path):
    
def Create_matrice_one_team(path):
    Matrice = np.zeros([3,4,20])
    Team_from_csv = pd.read_csv(path)
    Team_from_csv = Team_from_csv.to_numpy()[:,2:]
    for i in range(20):
        Matrice[2,0,i]=Team_from_csv[1,i]
        Matrice[2,1,i]=Team_from_csv[2,i]
        Matrice[2,2,i]=Team_from_csv[3,i]
        Matrice[2,3,i]=Team_from_csv[4,i]
        Matrice[1,0,i]=Team_from_csv[5,i]
        Matrice[1,1,i]=Team_from_csv[6,i]
        Matrice[1,2,i]=Team_from_csv[7,i]
        Matrice[1,3,i]=Team_from_csv[8,i]
        Matrice[0,1,i]=Team_from_csv[9,i]
        Matrice[0,2,i]=Team_from_csv[10,i]
    return Matrice

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
                Matrice = Create_matrice_one_team(path+"/StatsWholeTeamAfterScore/"+y)
                List_team.remove(y)
                #Name_team.remove(x)
                with open("Base de données/Matrices à vérifier/"+x+"_"+f"{season}"+".npy", 'wb') as f:
                    np.save(f, Matrice)
    print(List_team)
    return Name_team
    
n = create_team_league_season(path)
    
'''
M = Create_matrice_one_team("Base de données/2020/FR1/StatsWholeTeamAfterScore/Nîmes.csv")
with open("Base de données/Matrices_Team/"+"Nîmes Olympique"+"_"+f"{2020}"+".npy", 'wb') as f:
    np.save(f, M)
'''
    
    