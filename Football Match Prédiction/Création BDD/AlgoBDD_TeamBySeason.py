import numpy as np
import pandas as pd
import os

Game_played = pd.read_csv("C:/Users/LD/Desktop/Projet P5/CSV Kaggle/games.csv")
I = ["game_id", "competition_id", "season", "date", "home_club_name", "away_club_name", "home_club_goals",
     "away_club_goals"]
Game_played = Game_played[I]

Game_played = Game_played[(Game_played["competition_id"] == "IT1") | (Game_played["competition_id"] == "FR1") | (
            Game_played["competition_id"] == "GB1") | (Game_played["competition_id"] == "ES1") | (
                                      Game_played["competition_id"] == "L1")]
Game_played = Game_played[
    (Game_played["season"] == 2018) | (Game_played["season"] == 2019) | (Game_played["season"] == 2020) | (
                Game_played["season"] == 2021) | (Game_played["season"] == 2022)]
Game_played = Game_played[["season", "home_club_name", "away_club_name", "home_club_goals", "away_club_goals"]]
G = Game_played.to_numpy()

for i in range(G.shape[0]):
    G[i,1]= G[i,1] + "_"+str(G[i,0])
    G[i,2]= G[i,2] + "_"+str(G[i,0])

Team_game = G[:, 0:3]


DF = pd.DataFrame(Team_game, columns=["Season", "Home_Team", "Away_Team"])

# save the dataframe as a csv file
DF.to_csv("C:/Users/LD/Desktop/Projet P5/CSV Kaggle/team_game_by_season.csv")