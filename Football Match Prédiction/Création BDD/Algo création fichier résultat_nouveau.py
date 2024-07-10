#!/usr/bin/env python3
# -*- codin -*-
"""
Created on Mon Jan  8 20:25:06 2024

@author: mathieu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 18:28:44 2023

@author: mathieu
"""

import numpy as np
import pandas as pd
import os 


Game_played = pd.read_csv("Base de données/games.csv")
I = ["game_id","competition_id","season","date","home_club_name","away_club_name","home_club_goals","away_club_goals"]
Game_played=Game_played[I]

Game_played = Game_played[(Game_played["competition_id"] == "IT1") | (Game_played["competition_id"] == "FR1") | (Game_played["competition_id"] == "GB1") | (Game_played["competition_id"] == "ES1") | (Game_played["competition_id"] == "L1")]
#Game_played = Game_played[(Game_played["competition_id"] == "GB1") & (Game_played["season"] == 2022)]
Game_played = Game_played[(Game_played["season"] == 2018) | (Game_played["season"] == 2019) | (Game_played["season"] == 2020) | (Game_played["season"] == 2021) | (Game_played["season"] == 2022)]
Game_played['date'] = pd.to_datetime(Game_played['date'])
Game_played.sort_values(by='date', inplace = True)
#Game_index = Game_played[(Game_played["competition_id"] == "GB1") & (Game_played["season"] == 2022)].index
#Game_played = Game_played.drop(Game_index)
Game_played=Game_played[["season","home_club_name","away_club_name","home_club_goals","away_club_goals"]]
G = Game_played.to_numpy()

Result = np.zeros([G.shape[0],1])

for i in range(G.shape[0]):
    G[i,1]= G[i,1] + "_"+str(G[i,0])
    G[i,2]= G[i,2] + "_"+str(G[i,0])
    if G[i,3]>G[i,4]:
        Result[i,0]=int(1)
    if G[i,3]<G[i,4]:
        Result[i,0]=int(-1)
    if G[i,3]==G[i,4]:
        Result[i,0]=int(0)
'''
R = []
for i in range(Result.shape[0]):
    R.append(Result[i,0])
print("Victoire domicile dans le dataset : ",format(R.count(1)/Result.shape[0]*100, '.2f'),"%")
print("Victoire extérieur dans le dataset : ",format(R.count(-1)/Result.shape[0]*100, '.2f'),"%")
print("Match nul dans le dataset : ",format(R.count(0)/Result.shape[0]*100, '.2f'),"%")
'''

Team_game = G[:,1:3]

Team_game_result  = np.hstack((Team_game,Result))
Forme_team = np.zeros([G.shape[0],2])
Team_game_result =  np.hstack((Team_game_result,Forme_team))

Team = np.unique(Team_game_result[:,0])
T = []
for i in range(Team.shape[0]):
    T.append(Team[i])
Forme=Team.reshape((-1,1))
H = np.zeros((len(T),1+5))
Forme = np.hstack((Forme,H))

for i in range(Team_game_result.shape[0]):
    indice_domicile = T.index(Team_game_result[i,0])
    indice_exte = T.index(Team_game_result[i,1])
    d = 0
    e = 0
    for j in range(5):
        d += Forme[indice_domicile,2+j]
        e += Forme[indice_exte,2+j]
    Team_game_result[i,3] = d/5
    Team_game_result[i,4] = e/5
    if Forme[indice_domicile,1] == 4 :
        Forme[indice_domicile,int(2 + Forme[indice_domicile,1])] = Team_game_result[i,2]
        Forme[indice_domicile,1] = 0    
    else:
        Forme[indice_domicile,int(2 + Forme[indice_domicile,1])] = Team_game_result[i,2]
        Forme[indice_domicile,1] = Forme[indice_domicile,1] + 1
    if Forme[indice_exte,1] == 4 :
        Forme[indice_exte,int(2 + Forme[indice_exte,1])] = -Team_game_result[i,2]
        Forme[indice_exte,1] = 0
    else :
        Forme[indice_exte,int(2 + Forme[indice_exte,1])] = -Team_game_result[i,2]
        Forme[indice_exte,1] = Forme[indice_exte,1] + 1
    
for i in range(Team_game_result.shape[0]):
    if Team_game_result[i,2]==-1:
        Team_game_result[i,2]=2
  
DF = pd.DataFrame(np.hstack((Team_game_result[:,:2],Team_game_result[:,3:])),columns=["Home_Team","Away_Team","Forme_dom","Forme_exte"])

# save the dataframe as a csv file
#DF.to_csv("Team_game_result102_forme.csv")
       
tab = pd.DataFrame({"Résultat" : Team_game_result[:,2],
                    "Forme domicile" : Team_game_result[:,3],
                    "Forme extérieur" : Team_game_result[:,4]})
u = tab.corr()
Buts = np.zeros((len(T),6))#Nombre de match à dom et exterieur, Moyenne de buts mis et pris à dom et Moyenne de buts mis/pris à extérieur
Buts = np.hstack((Team.reshape((-1,1)),Buts))

Buts_antérieur_match = np.zeros((Team_game_result.shape[0],4)) 

for i in range(Team_game_result.shape[0]):
    indice_domicile = T.index(Team_game_result[i,0])
    indice_exte = T.index(Team_game_result[i,1])
    Buts_antérieur_match[i,0],Buts_antérieur_match[i,1],Buts_antérieur_match[i,2],Buts_antérieur_match[i,3] = Buts[indice_domicile,3],Buts[indice_domicile,4],Buts[indice_exte,5],Buts[indice_exte,6]
    
    Buts[indice_domicile,1] = Buts[indice_domicile,1] + 1
    n_matchs_joues = Buts[indice_domicile,1]
    Buts[indice_domicile,3] = Buts[indice_domicile,3]*((n_matchs_joues-1)/n_matchs_joues) +  G[i,3]/n_matchs_joues
    Buts[indice_domicile,4] = Buts[indice_domicile,4]*((n_matchs_joues-1)/n_matchs_joues) +  G[i,4]/n_matchs_joues
    
    Buts[indice_exte,2] = Buts[indice_exte,2] + 1
    n_matchs_joues = Buts[indice_exte,2]
    Buts[indice_exte,5] = Buts[indice_exte,5]*((n_matchs_joues-1)/n_matchs_joues) +  G[i,4]/n_matchs_joues
    Buts[indice_exte,6] = Buts[indice_exte,6]*((n_matchs_joues-1)/n_matchs_joues) +  G[i,3]/n_matchs_joues
    
LOLOLOL= np.hstack((G[:,1:],Buts_antérieur_match))
LOLOLOLO = pd.DataFrame(LOLOLOL,columns=["Home_Team","Away_Team","Buts_dom","Buts_exté","Moy_but_mis_dom","Moy_but_pris_dom","Moy_but_mis_exte","Moy_but_pris_exte"])
LOLOLOLO.to_csv("LOLOLO.csv")
tableau = pd.DataFrame({"Résultat" : Team_game_result[:,2],
                    "Moy_but_mis_dom" : Buts_antérieur_match[:,0],
                    "Moy_but_pris_dom" : Buts_antérieur_match[:,1],
                    "Moy_but_mis_exte" : Buts_antérieur_match[:,2],
                    "Moy_but_pris_exte" : Buts_antérieur_match[:,3]})
o = tableau.corr()
