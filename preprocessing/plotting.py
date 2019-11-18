# %%

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


# %%

def draw_word_num_pic():
    with open('descriptions_results_num_params.sav', 'rb') as f:
        tmp = pickle.load(f)
    des_num = tmp[0]
    resu_num = tmp[1]
    des_num_reta = tmp[2]
    des_num_manu = tmp[3]
    des_num_other = tmp[4]
    resu_num_reta = tmp[5]
    resu_num_manu = tmp[6]
    resu_num_other = tmp[7]

    sns.kdeplot(des_num, shade=True)  # bins=60,histtype="stepfilled", alpha=.8
    plt.title('Words number distribution of website desriptions')
    plt.xlabel('Number of words ')
    plt.ylabel('Density')
    plt.show()
    plt.clf()

    sns.kdeplot(resu_num, shade=True)
    plt.title('Words number distribution of search results')
    plt.xlabel('Number of words ')
    plt.ylabel('Density')
    plt.show()
    #     print(len(des_num))
    #     print(pd.DataFrame({"description":des_num, "result":resu_num}).describe())

    sns.kdeplot(des_num_reta, shade=True)  # bins=60,histtype="stepfilled", alpha=.8
    plt.title('Words number distribution of retailer website desriptions')
    plt.xlabel('Number of words ')
    plt.ylabel('Density')
    plt.show()
    plt.clf()

    sns.kdeplot(des_num_manu, shade=True)  # bins=60,histtype="stepfilled", alpha=.8
    plt.title('Words number distribution of manufacturer website desriptions')
    plt.xlabel('Number of words ')
    plt.ylabel('Density')
    plt.show()
    plt.clf()

    sns.kdeplot(des_num_other, shade=True)  # bins=60,histtype="stepfilled", alpha=.8
    plt.title('Words number distribution of other website desriptions')
    plt.xlabel('Number of words ')
    plt.ylabel('Density')
    plt.show()
    plt.clf()

    sns.kdeplot(resu_num_reta, shade=True)  # bins=60,histtype="stepfilled", alpha=.8
    plt.title('Words number distribution of retailer search results')
    plt.xlabel('Number of words ')
    plt.ylabel('Density')
    plt.show()
    plt.clf()

    sns.kdeplot(resu_num_manu, shade=True)  # bins=60,histtype="stepfilled", alpha=.8
    plt.title('Words number distribution of manufacturer search results')
    plt.xlabel('Number of words ')
    plt.ylabel('Density')
    plt.show()
    plt.clf()

    sns.kdeplot(resu_num_other, shade=True)  # bins=60,histtype="stepfilled", alpha=.8
    plt.title('Words number distribution of other search results')
    plt.xlabel('Number of words ')
    plt.ylabel('Density')
    plt.show()
    plt.clf()

    df_all = pd.DataFrame({'descriptions': des_num, 'results': resu_num})
    df_reta = pd.DataFrame({'retailer descriptions': des_num_reta, 'retailer results': resu_num_reta})
    df_manu = pd.DataFrame({'manufacturer descriptions': des_num_manu, 'manufacturer results': resu_num_manu})
    df_other = pd.DataFrame({'other descriptions': des_num_other, 'other results': resu_num_other})

    dfs = [df_all, df_reta, df_manu, df_other]

    for df in dfs:
        print(df.describe())


# %%

draw_word_num_pic()

# %%


# %%


