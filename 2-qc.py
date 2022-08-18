import sweetviz as sv
import pandas as pd
from tqdm import tqdm
from scipy.stats import mannwhitneyu


def svBeforeAfter(before, after, fileName):
    edaReport = sv.compare([before, 'without refill'], [after, 'with refill'])
    edaReport.show_html(filepath=f'EDA-{fileName}.html', open_browser=False)


def mwUtest(before, after, fileName):
    indices = after.columns
    nonPtest = pd.DataFrame()
    for i in tqdm(indices):
        statistic, pValue = mannwhitneyu(before[i], after[i], nan_policy='omit')
        local_result = pd.DataFrame({'Feature': [i], 'Mann-Whitney U statistic': [statistic], 'p-value': pValue})
        nonPtest = pd.concat([nonPtest, local_result], ignore_index=True)
    nonPtest.set_index('Feature').to_csv(f'QC-{fileName}.csv')


raw_df = pd.read_excel('ASDTD_SRSCBCL_220804.xlsx').set_index('famid')
raw_asd = raw_df.query('group == 1').drop(columns='group')
print(raw_asd.head())
raw_tdc = raw_df.query('group == 0').drop(columns='group')
print(raw_tdc.head())

refill_asd = pd.read_csv('refilled-case.csv').set_index('famid')
print(refill_asd.head())
refill_tdc = pd.read_csv('refilled-control.csv').set_index('famid')
print(refill_tdc.head())

svBeforeAfter(raw_asd, refill_asd, 'ASD')
svBeforeAfter(raw_tdc, refill_tdc, 'TDC')
mwUtest(raw_asd.dropna(), refill_asd, 'ASD')
mwUtest(raw_tdc.dropna(), refill_tdc, 'TDC')

