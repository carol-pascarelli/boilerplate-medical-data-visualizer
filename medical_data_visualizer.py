import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
file_path= ('medical_examination.csv')
df = pd.read_csv(file_path)

# 2
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2)).apply(lambda x: 1 if x > 25 else 0)

# 3
df.loc[df['cholesterol'] == 1, ['cholesterol']] = 0

df.loc[df['cholesterol'] > 1, ['cholesterol']] =1

df.loc[df['gluc'] == 1, ['gluc']] = 0

df.loc[df['gluc'] > 1, ['gluc']] = 0


# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])


    # 6
    df_cat = ['total'] = 1
    

    # 7
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'], as_index=False).count()



    # 8
    fig = sns.catplot(data=df_cat, x = 'variable', y = 'total', hue = 'value', kind = 'bar', col='cardio', height =4, aspect = 2)
    plt.show()


    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12
    corr = df_heat.corr(method='pearson')

    # 13
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True



    # 14
    fig, ax = plt.subplots(figsize=(12, 8))

    # 15
    sns.heatmap(corr, linewidths=1, annot=True, square= True, mask =mask, fmt ='.1f', center=0.08, cbar_kws={'shrink':0.5})


    # 16
    fig.savefig('heatmap.png')
    return fig
