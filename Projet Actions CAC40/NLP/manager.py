import glob
import pandas as pd
from datetime import datetime

##################################################################################

company_files = []
lengths = []

##################################################################################

def files(company_name): 
    for file in glob.glob(f"csv/{company_name}_*.csv"):
        company_files.append(file[4:]) 

def concated_files(company_name):
    df = pd.read_csv(f"csv/{company_files[0]}", index_col=0)
    lengths.append(len(df))
    for company_file in company_files[1:]:
        tmp = pd.read_csv(f"csv/{company_file}", index_col=0)
        lengths.append(len(tmp))
        df = pd.concat([df, tmp])
    df.reset_index(drop = True, inplace = True)
    df["company_name"] = company_name
    #print(lengths, sum(lengths), len(df))
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

def save(company_name, df):
    df.to_csv(f"datasets/{company_name}.csv")

def main(company_name):
    files(company_name)
    df = concated_files(company_name)
    df = sorted_files(df)
    df = date_format(df)
    save(company_name, df)