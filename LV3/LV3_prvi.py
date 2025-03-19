import pandas as pd
import numpy as np

mtcars = pd.read_csv('mtcars.csv')

top5_mpg = mtcars.sort_values(by='mpg', ascending=True).head(5)
print("Top 5 automobila s najvecom potrosnjom:")
print(top5_mpg[['car', 'mpg']])


v8_low_mpg = mtcars[mtcars['cyl'] == 8].sort_values(
    by='mpg', ascending=True).tail(3)
print("Top 3 najmanja v8 potrosaca:")
print(v8_low_mpg[['car', 'mpg']])

six_avg = mtcars[mtcars['cyl'] == 6]['mpg'].mean()
print(f"\nSrednja potrosnja svih sestaka je: {six_avg:.2f} mpg")

four_pot_avg = mtcars[(mtcars['cyl'] == 4) & (
    mtcars['wt'] <= 2.2) & (mtcars['wt'] >= 2.0)]['mpg'].mean()
print(
    f"Srednja potrosnja automobila s 4 cilindra a da su izmedu 2000 i 2200 poundsa: {four_pot_avg:.2f}")

broj_manual = mtcars[mtcars['am'] == 1].shape[0]
broj_automat = mtcars[mtcars['am'] == 0].shape[0]
print(f"Broj manualnih automobila: {broj_manual}")
print(f"Broj automobila s automatskim mjenjačem: {broj_automat}")

automatik_preko100 = mtcars[(mtcars['am'] == 0) &
                            (mtcars['hp'] > 100)].shape[0]
print(f"Broj automatika preko 100 konja: {automatik_preko100}")

mtcars['wt_kg'] = mtcars['wt'] * 453.592
print(
    f"Masa svakog automobila u kilogramima: {mtcars[['car', 'wt', 'wt_kg']]}")
