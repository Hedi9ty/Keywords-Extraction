# author: Hedi
# date: 2010/11/27
# coding: utf-8

# # Import Dependencies
# import package
import os
import numpy as np
import pandas as pd
import datetime as dt
# import DTT Function
from Functions.jieba_split_beta import zh_split
from Functions.text_mining_beta import text_mining

# # Load Data
data = pd.read_excel('NEWS_v1_20191126.xls')
# # 輸入銀行及對應主題關鍵字
銀行_mapping = {'永豐':1, '國泰':2, '玉山':3, '富邦':4, '台新':5, '中信':6}
銀行_list = ['永豐','國泰','玉山','富邦','台新','中信']
主題_list = ['數位轉型','數位帳戶','數位組織','開發運作A','開發運作B','AI']

# # 斷詞
def jieba_split(Series):
    
    cwd = os.getcwd()
    # Add Dictionary
    analyzer = zh_split(cwd + '/Dictionary/dict.txt.big')
    analyzer.add_dictionary(cwd +'/Dictionary/company_dict.txt')
    analyzer.add_dictionary(cwd +'/Dictionary/edu_dict.txt')
    analyzer.add_dictionary(cwd +'/Dictionary/fin_dict.txt')
    analyzer.add_dictionary(cwd +'/Dictionary/geo_dict.txt')    
    analyzer.add_dictionary(cwd +'/Dictionary/law_dict.txt')
    analyzer.add_dictionary(cwd +'/Dictionary/reg_dict.txt')
    # word split
    analyzer.split(Series)
    analyzer.get_dictionary()
    # word filter
    analyzer.word_filter(w_len = 0, path_word = cwd + '/Dictionary/stop_words.txt')
    # using Ngram to find key word
    analyzer.find_keyword(n=2)
    
    return analyzer
 
# # 關鍵字分析
def key_word_analysis(bank, topic):

    # # 斷詞
    df = data[data[topic] == float(銀行_mapping[bank])]
    analyzer = jieba_split(df['Content'])

    # # Text Vectorization
    # create word disctionary
    tm = text_mining(analyzer.split_list)
    tm.get_dictionary()
    # create TF, TF-IDF and textrank vector for each text
    tm.CounterVector()
    tm.TfidfVector()
    tm.textrank()

    # # Key Word Analysis
    n_key = 20
    times = 2
    # sort TFIDF vector
    tm.TFIDF_Vector = [sorted(row, key = lambda x:x[1], reverse = True)for row in tm.TFIDF_Vector]
    # get top n key word fot each text
    TR_result =  tm.Word_Cloud(tm.TEXTRANK_Vector, n_key = n_key*times, dictionary=tm.dic).sort_values('Value', ascending = False)
    try:
        TFIDF_result = tm.Word_Cloud(tm.TFIDF_Vector, n_key = n_key*times, dictionary=tm.dic).sort_values('Value', ascending = False)
    except ValueError:
        TFIDF_result = tm.Word_Cloud(tm.TF_Vector, n_key = n_key*times, dictionary=tm.dic).sort_values('Value', ascending = False)
    TR_result =  tm.Word_Cloud(tm.TEXTRANK_Vector, n_key = n_key*times, dictionary=tm.dic).sort_values('Value', ascending = False)
    # get key phrase
    TR_index = TR_result.關鍵字.unique()
    TFIDF_index = TFIDF_result.關鍵字.unique()
    # add key word
    addword_TR = [w for _, _, w in analyzer.add_word[:1000] if w[0] in TR_index and w[1] in TR_index]
    addword_TFIDF = [w for _, _, w in analyzer.add_word[:1000] if w[0] in TFIDF_index and w[1] in TFIDF_index]
    addword = list(set(addword_TR + addword_TFIDF))
    addword
    # remove key word and add key phrase
    TR_result = tm.key_phrase(addword, TR_result)
    TFIDF_result = tm.key_phrase(addword, TFIDF_result)
    # filter key word which len = 1
    TR_result = TR_result[TR_result.關鍵字.apply(lambda x:len(x) > 1)]
    TFIDF_result = TFIDF_result[TFIDF_result.關鍵字.apply(lambda x:len(x) > 1)]
    # group by key word
    TR_result_group = TR_result.groupby('關鍵字').sum().sort_values('Value', ascending = False).reset_index().drop('index', axis = 1)
    TFIDF_result_group = TFIDF_result.groupby('關鍵字').sum().sort_values('Value', ascending = False).reset_index().drop('index', axis = 1)

    # # get key word format
    result = TR_result_group.merge(TFIDF_result_group, on = '關鍵字', how = 'outer')
    result = result.fillna(-1)
    result['value'] = result[['Value_x', 'Value_y']].sum(axis=1)
    result = result.sort_values('value', ascending = False)
    
    return result

# # Output
if __name__ == '__main__':
    
    date = dt.datetime.strftime(dt.datetime.now(), '%Y%m%d')
    for i in 銀行_list:
        bank = i
        for j in 主題_list:
            topic = j
    result = key_word_analysis(bank, topic)
    output = result[['關鍵字', 'value']]
    output.to_excel(bank+'_'+topic+'_銀行_關鍵字_'+date+'.xlsx', encoding = 'utf8', index = False)
    print('-'*30+bank+'_'+topic+'_關鍵字分析完成'+'-'*30)
