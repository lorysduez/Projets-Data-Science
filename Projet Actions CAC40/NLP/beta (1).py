import os 
import os.path

import google
import lesechos
import investir_lesechos
import latribune
import manager

import pandas as pd
from datetime import datetime

"""
Immobilier : Unibail-Rodamco-Westfield, Klépierre
Services publics : Suez, Engie
Chimie et matériaux : Arkema, Solvay
Aéronautique et défense : Airbus Group, Thales
Luxe : LVMH, Kering
Services aux entreprises : Dassault Systemes, Accor
Biotechnologie : Sanofi, Ipsen
Alimentation : Danone, Nestlé
Médias : Vivendi, Lagardère, Publicis Groupe
Énergie et matières premières : Total, Engie, Valeo
Technologies et télécommunications : Orange, Atos, Capgemini
Consumer goods et distribution : Carrefour, Danone, L'Oréal
Santé : Sanofi, EssilorLuxottica, Ipsen
Industrie : Air Liquide, Michelin, Peugeot
Services financiers : Axa, BNP Paribas, Crédit Agricole
Médias et divertissements : Vivendi, Lagardère, Publicis Groupe
"""


company_name = "airbus"

website = "lesechos.fr"
google.main(company_name, website)
lesechos.main(company_name)
investir_lesechos.main(company_name)

website = "latribune.fr"
google.main(company_name, website)
latribune.main(company_name)

manager.main(company_name)

"""
lengths = []
company_files = os.listdir("datasets")

def concated_files():
    df = pd.read_csv(f"datasets/{company_files[0]}", index_col=0)
    lengths.append(len(df))
    for company_file in company_files[1:]:
        tmp = pd.read_csv(f"datasets/{company_file}", index_col=0)
        lengths.append(len(tmp))
        df = pd.concat([df, tmp])
    df.reset_index(drop = True, inplace = True)
    print(lengths, sum(lengths), len(df))
    return df

def sorted_files(df):
    for i in range (len(df["dates"])):
        df["dates"][i] = datetime.strptime(df["dates"][i], "%d/%m/%Y").date()    
    df.sort_values(by = "dates", ascending = False, inplace = True)
    df.reset_index(drop = True, inplace = True)
    return df

def date_format(df):
    for i in range (len(df['dates'])):
        df['dates'][i] = df['dates'][i].strftime("%d/%m/%Y")
    return df

def save(df):
    df.to_csv(f"datasets/dataset.csv")


df = concated_files()
df = sorted_files(df)
df = date_format(df)
save(df)
"""