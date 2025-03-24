import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

mtcars = pd.read_csv('mtcars.csv')

# 1. Barplot potrošnje automobila s 4, 6 i 8 cilindara
avg_mpg_by_cyl = mtcars.groupby('cyl')['mpg'].mean()
plt.bar(avg_mpg_by_cyl.index, avg_mpg_by_cyl.values, color='green')
plt.title('Prosječna potrošnja automobila s 4, 6 i 8 cilindara')
plt.xlabel('Broj cilindara')
plt.ylabel('Potrošnja (mpg)')
plt.show()

# 2. Boxplot distribucije težine automobila s 4, 6 i 8 cilindara
cylinders = [4, 6, 8]
data_to_plot = [mtcars[mtcars['cyl'] == cyl]['wt'] for cyl in cylinders]
plt.boxplot(data_to_plot, labels=cylinders)
plt.title('Distribucija težine automobila s 4, 6 i 8 cilindara')
plt.xlabel('Broj cilindara')
plt.ylabel('Težina (1000 lbs)')
plt.show()

# 3. Usporedba potrošnje između ručnih i automatskih mjenjača
manual_mpg = mtcars[mtcars['am'] == 1]['mpg']
auto_mpg = mtcars[mtcars['am'] == 0]['mpg']
plt.boxplot([auto_mpg, manual_mpg], labels=['Automatski', 'Ručni'])
plt.title('Usporedba potrošnje - Ručni vs. Automatski mjenjač')
plt.xlabel('Mjenjač')
plt.ylabel('Potrošnja (mpg)')
plt.show()

# 4. Odnos ubrzanja i snage automobila za ručni i automatski mjenjač
manual = mtcars[mtcars['am'] == 1]
auto = mtcars[mtcars['am'] == 0]
plt.scatter(manual['hp'], manual['qsec'], color='red', label='Ručni')
plt.scatter(auto['hp'], auto['qsec'], color='yellow', label='Automatski')
plt.title('Odnos ubrzanja i snage za ručne i automatske mjenjače')
plt.xlabel('Snaga (hp)')
plt.ylabel('Ubrzanje (qsec)')
plt.legend()
plt.show()
