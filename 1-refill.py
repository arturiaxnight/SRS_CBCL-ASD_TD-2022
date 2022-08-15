""" 回填 """
import pandas as pd
from tqdm import tqdm
import numpy as np
from catboost import CatBoostRegressor, Pool

raw_df = pd.read_excel('ASDTD_SRSCBCL_220804.xlsx').set_index('famid')
print(raw_df.head())

# 紀錄完整的資料
refill_df = raw_df.copy()
indexWithMissing = raw_df.columns[raw_df.isna().any()].tolist()
indexWithoutMissing = raw_df.columns[~raw_df.isna().any()].drop("group").tolist()
print(indexWithoutMissing)

# 回填會用到的function
def eval_predict_score(pool: Pool): # 評估回填誤差
    model = CatBoostRegressor()
    model.fit(pool, verbose=0)
    return model.get_best_score()['learn']['RMSE']


def screen_idx(whole_df): # 尋找回填目標
    performance = []
    idx_missing = whole_df.columns[whole_df.isna().any()].tolist()
    idx_complete = whole_df.columns[~whole_df.isna().any()].tolist()
    for i in tqdm(idx_missing, desc="screening", leave=False):
        next_df = whole_df[idx_complete + [i]]
        train_df = next_df[~next_df[i].isna()]
        train_pool = Pool(train_df[idx_complete], train_df[i])
        rmse = eval_predict_score(train_pool)
        performance.append([i, rmse])
    performance.sort(key=lambda x: x[1])
    return performance[0][0]

def iter_refilling(whole_df): # 循環回填
    pbar = tqdm(desc='Imputation...', total=whole_df.isna().any().sum())
    i = 0
    while whole_df.isna().any().sum() > 0:
        idx_complete = whole_df.columns[~whole_df.isna().any()].tolist()
        target_idx = screen_idx(whole_df)
        next_df = whole_df[idx_complete + [target_idx]]
        train_df = next_df[~next_df[target_idx].isna()]
        predict_df = next_df[next_df[target_idx].isna()]
        train_pool = Pool(train_df[idx_complete], train_df[target_idx])
        model = CatBoostRegressor()
        model.fit(train_pool, verbose=0)
        predict_value = model.predict(predict_df)
        target_mark = whole_df[target_idx].index[whole_df[target_idx].apply(np.isnan)]
        if (train_df[target_idx] % 1 == 0).all():  # INT features
            for x in range(len(predict_value)):
                whole_df[target_idx].at[target_mark[x]] = np.around(predict_value[x], 0)
        else:  # FLOAT features
            for x in range(len(predict_value)):
                whole_df[target_idx].at[target_mark[x]] = predict_value[x]
        i+=1
        pbar.update(1)
    return whole_df

# 執行回填
case_df = refill_df.query("group == 1").drop("group", axis=1)
print(case_df.head())
control_df = refill_df.query("group == 0").drop("group", axis=1)
print(control_df.head())

caseRefilledDataframe = iter_refilling(case_df)
controlRefilledDataframe = iter_refilling(control_df)
caseRefilledDataframe.to_csv("refilled-case.csv")
controlRefilledDataframe.to_csv("refilled-control.csv")
print(caseRefilledDataframe.isna().sum().sum(), print(controlRefilledDataframe.isna().sum().sum()))
