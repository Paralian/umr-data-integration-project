
#sys.path.append('./../ext_modules')
import numpy as np
import pandas as pd
import sys
import re
import jellyfish

# Load DF
DF_yugioh   = pd.read_csv('./../source_data/card_data.csv')  # Dataset 1: Yu-Gi-Oh!
DF_skyrim   = pd.read_csv('./../source_data/Skyrim_Named_Characters.csv')  # Dataset 2: Skyrim
DF_dd5      = pd.read_csv('./../source_data/Dd5e_monsters.csv')  # Dataset 3: d&d5


# Get column names
def column_names(df):
    return df.columns.values.tolist()

attribute_yugioh    = column_names(DF_yugioh)
attribute_skyrim    = column_names(DF_skyrim)
attribute_dd5       = column_names(DF_dd5)


## Perform integration rules and normalization for each dataset in place

# For Yu-Gi-Oh!:
DF_yugioh['vitality'] = DF_yugioh[['ATK', 'DEF']].max(axis=1)  # Rule: vitality <- max(ATK, DEF)
DF_yugioh['vitality'] = DF_yugioh['vitality']/DF_yugioh['vitality'].max(axis=0)  # Normalization

DF_yugioh['harmful'] = [True if atk>0 else False for atk in DF_yugioh["ATK"]]  # Rule: if ATK>0 is harmful

DF_yugioh['development_stage'] = DF_yugioh['Level']/DF_yugioh['Level'].max(axis=0)  # dev_stage <- norm(Level)
DF_yugioh['attack'] = DF_yugioh['ATK']/DF_yugioh['ATK'].max(axis=0)

DF_yugioh.rename(columns={"Name": "name", "Type": "type", "Race": "kind"}, inplace=True)  # Rename columns to concat with final entity

# For Skyrim:

# >>>>>>>>>>>>Cleaning
def split_space(string):
    return string.split()[0]

def replace(string):
    string = string.encode("ascii", "ignore")
    string = string.decode()
    return string.replace('PC', '81').replace('x', ' ').replace('×', ' ').replace('(', ' ').replace('-', ' ').replace('+', ' ').replace('Radiant', '').replace('Leveled', '')  # PC=81 because it's max level in Skyrim before expansion

DF_skyrim['Level'] = DF_skyrim['Level'].apply(lambda x: replace(x) if isinstance(x, str) else np.nan)
DF_skyrim['Level_parsed'] = DF_skyrim.Level.apply(split_space).astype(float).astype(int)

DF_skyrim['Health'] = DF_skyrim['Health'].apply(lambda x: replace(x) if isinstance(x, str) else np.nan)
DF_skyrim['Health_parsed'] = DF_skyrim.Health.apply(lambda x: int(split_space(x)) if isinstance(x, str) else np.nan)

DF_skyrim['Stamina'] = DF_skyrim['Stamina'].apply(lambda x: replace(x) if isinstance(x, str) else np.nan)
DF_skyrim['Stamina_parsed'] = DF_skyrim.Stamina.apply(lambda x: int(split_space(x)) if isinstance(x, str) else np.nan)

DF_skyrim['Magicka'] = DF_skyrim['Magicka'].apply(lambda x: replace(x) if isinstance(x, str) else np.nan)
DF_skyrim['Magicka_parsed'] = DF_skyrim.Magicka.apply(lambda x: int(split_space(x)) if isinstance(x, str) else np.nan)
# <<<<<<<<<<Cleaning

DF_skyrim['vitality'] = DF_skyrim['Health_parsed']/DF_skyrim['Health_parsed'].max(axis=0)  # Vitality <- norm(Health)

DF_skyrim['development_stage'] = DF_skyrim['Level_parsed']/DF_skyrim['Level_parsed'].max(axis=0)  # dev_stage <- norm(Level)

DF_skyrim['attack'] = (DF_skyrim['Stamina_parsed']/DF_skyrim['Stamina_parsed'].max(axis=0) + DF_skyrim['Magicka_parsed']/DF_skyrim['Magicka_parsed'].max(axis=0))/2  # Rule: attack <- (norm(Stamina)+norm(Magicka))/2

skyrim_aggresion_levels = pd.unique(DF_skyrim['Aggression'])  # First is unaggressive
DF_skyrim['harmful'] = [False if aggro == skyrim_aggresion_levels[0] else True for aggro in DF_skyrim['Aggression']]  # Rule: False if unaggressive, else True

DF_skyrim.rename(columns={"Name": "name", "Class Details": "type", "Race": "kind"}, inplace=True)  # Rename columns to concat with final entity

# For D&D5:
def split_coma_first(string):
    return string.split(', ')[0]

def split_coma_second(string):
    return string.split(', ')[1]

# Rule: kind, type <- Race + Alignment
DF_dd5['kind'] = DF_dd5['Race + alignment'].apply(split_coma_first)
DF_dd5['type'] = DF_dd5['Race + alignment'].apply(split_coma_second)

DF_dd5['HP_parsed'] = DF_dd5.HP.apply(split_space).astype(int)
DF_dd5['Armor_parsed'] = DF_dd5.Armor.apply(split_space).astype(int)
DF_dd5['vitality'] = (DF_dd5['HP_parsed']/DF_dd5['HP_parsed'].max(axis=0) + DF_dd5['Armor_parsed']/DF_dd5['Armor_parsed'].max(axis=0))/2  # Rule: vitality <- (norm(HP)+norm(Armor))/2

DF_dd5['Speed_parsed'] = DF_dd5.Speed.apply(split_space).apply(lambda x: int(x) if x != 'Swim' else np.nan)  # Swim is a word in the column with unknown origin
DF_dd5['attack'] = (DF_dd5['Speed_parsed']/DF_dd5['Speed_parsed'].max(axis=0) + DF_dd5['Armor_parsed']/DF_dd5['Armor_parsed'].max(axis=0))/2  # Rule: attack <- (norm(Speed)+norm(Armor))/2

DF_dd5['harmful'] = True  # Rule: all are harmful...

# '''
# Creature size may be used as development stage, despite the lack of accuracy, since creatures are big or small depending
# on its species/kind and not its development stage.
# The dev_stage will be the space occupied by the creature, normalized with respect to the maximum.
# Information about creatures size can be found at https://www.dungeonsolvers.com/2019/11/25/creature-size-in-dd-5e-size-matters/
# '''
dd5_sizes = pd.unique(DF_dd5['Size'])  # ['Large', 'Medium', 'Huge', 'Gargantuan', 'Small', 'Tiny']
dd5_size_dic = {'Tiny': 0.5/16, 'Small': 1/16, 'Medium': 1/16, 'Large': 4/16, 'Huge': 9/16, 'Gargantuan': 1}  # Sizes in squares
DF_dd5['development_stage'] = DF_dd5['Size'].map(dd5_size_dic)

DF_dd5.rename(columns={"Name": "name"}, inplace=True)  # Rename columns to concat with final entity


#### Compile all data together

DF_yugioh['universe'] = 'yugioh'
DF_skyrim['universe'] = 'skyrim'
DF_dd5['universe'] = 'dd5'

DF = pd.DataFrame(columns=['name', 'type', 'kind', 'development_stage', 'vitality', 'attack', 'harmful', 'universe'])  # Define global entity as pandas DF

DF = pd.concat([DF, DF_yugioh[['name', 'type', 'kind', 'development_stage', 'vitality', 'attack', 'harmful', 'universe']]], ignore_index=True)
DF = pd.concat([DF, DF_skyrim[['name', 'type', 'kind', 'development_stage', 'vitality', 'attack', 'harmful', 'universe']]], ignore_index=True)
DF = pd.concat([DF, DF_dd5[['name', 'type', 'kind', 'development_stage', 'vitality', 'attack', 'harmful', 'universe']]], ignore_index=True)

# Save DataFrame
DF.to_csv('./../final_data/integrated_entity.csv')

