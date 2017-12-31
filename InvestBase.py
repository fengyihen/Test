# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 21:43:06 2016
一些公共函数，例如收益率/回撤计算函数等

@author: Yizhen
"""

import pandas as pd

def tradestat_portfolio(portfolio):
    ###对收益曲线的统计
    tradestat = pd.DataFrame({
        'startdate': [min(portfolio['date'])],
        'enddate': [max(portfolio['date'])]
    })
    tradestat['ratio'] = portfolio['ratio'].iloc[portfolio.shape[0] - 1]

    tradestat['meandayratio'] = portfolio['dayratio'].mean()

    mdd = 0
    mdddate = 0
    portfolio['year'] = 0
    for i in portfolio.index:
        portfolio.loc[i, 'year'] = int(str(portfolio.loc[i, 'date'])[0:4])
        mdd1 = portfolio.loc[i, 'ratio'] - min(portfolio.loc[i:, 'ratio'])
        if mdd1 > mdd:
            mdd = mdd1
            mdddate = portfolio.loc[i, 'date']

    for year in range(2008, 2018):
        temp = portfolio[portfolio['year'] == year]
        if temp.shape[0] == 0:
            continue
        temp.index = range(0, temp.shape[0])
        tradestat[str(year) + 'ratio'] = sum(temp['dayratio'])

    tradestat['yearratio'] = tradestat['ratio'] / portfolio.shape[0] * 252
    tradestat['mdd'] = mdd
    tradestat['mdddate'] = mdddate
    tradestat['RRR'] = tradestat['yearratio'] / tradestat['mdd']

    tradestat['sharpratio'] = portfolio['dayratio'].mean() / portfolio[
        'dayratio'].std() * 252**0.5

    print(tradestat)

    return tradestat

def tradestatlist(self, hsmatradeday):  #n=200,100,50

    tradestatlist = pd.DataFrame()

    tradestat = self.tradestat(hsmatradeday)
    tradestat['Hedge'] = 'NoHedge'
    tradestatlist = pd.concat(
        [tradestatlist, tradestat], ignore_index=True)

    temp = hsmatradeday[['date', 'hedge300dayratio', 'hedge300ratio']]
    temp = temp.rename(columns={
        'hedge300dayratio': 'dayratio',
        'hedge300ratio': 'cumratio'
    })
    tradestat = self.tradestat(temp)
    tradestat['Hedge'] = 'Hedge300'
    tradestatlist = pd.concat(
        [tradestatlist, tradestat], ignore_index=True)

    temp = hsmatradeday[['date', 'hedge500dayratio', 'hedge500ratio']]
    temp = temp.rename(columns={
        'hedge500dayratio': 'dayratio',
        'hedge500ratio': 'cumratio'
    })
    tradestat = self.tradestat(temp)
    tradestat['Hedge'] = 'Hedge500'
    tradestatlist = pd.concat(
        [tradestatlist, tradestat], ignore_index=True)

    if any(hsmatradeday.columns == 'hedgecta1ratio'):
        temp = hsmatradeday[[
            'date', 'hedgecta1dayratio', 'hedgecta1ratio'
        ]]
        temp = temp.rename(columns={
            'hedgecta1dayratio': 'dayratio',
            'hedgecta1ratio': 'cumratio'
        })
        tradestat = self.tradestat(temp)
        tradestat['Hedge'] = 'HedgeCTA1'
        tradestatlist = pd.concat(
            [tradestatlist, tradestat], ignore_index=True)

    print(tradestatlist)
    tradestatlist.to_csv(
        "Test\\testresult\\" + self.label + ".csv", index=False)
