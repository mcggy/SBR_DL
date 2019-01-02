"""
Description: Call mongodb connection class and use that
"""

# Application imports
from apis import MongoConn
import pandas as pd
from pandas import Series
from nltk.stem.snowball import SnowballStemmer
from bs4 import BeautifulSoup
import re
import itertools
import nltk
import random
import os
import json




#11个文件，每个文件1000对（一对是三个）,要求有11000个master bug report，以及11000个duplicate bug report（duplicate bug report的个数肯定大于11000）
def get_data():

    db_obj=MongoConn(connection_obj={'DATABASE':'bug_report','HOST':'localhost','PORT':27017},collection='mozilla')


    allBR_count=db_obj.count()
    print('所有数据的个数：'+str(allBR_count))

    #控制台中的写法： db.getCollection('eall').find().sort({'creation_ts':-1}).limit(20000)    #limit设置应该是16000，这里为了测试方便，把它改成1600
    sbr_list=db_obj.list_document()
    print('the number of security of bug reports is ：'+str(len(sbr_list)))

    #转化成pandas数据
    sbr_df=pd.DataFrame(sbr_list)
    # duplicate_data.set_index('dup_id') #设置了之后，没有起作用

    sbr_df=sbr_df.rename(columns={'bug_id':'issue_id'})
    sbr_df['Security'] = 1

    sbr_bugID=[]
    for bug_report in sbr_list:
        #这里存储的dup_id用的是list
        sbr_bugID.append(bug_report['bug_id'])
    nsbr_df=pd.read_csv('../resource/mozilla/mozilla_50000.csv')

    nsbr_df=nsbr_df.rename(columns={'bug_id':'issue_id','short_desc':'summary','reporter_text':'reported_by','fixer_text':'assigned_to'})
    nsbr_df['Security']=0


    # filter nsbr that has existed in sbr
    nsbr_df=nsbr_df[nsbr_df.issue_id.map(lambda x: x not in sbr_bugID)]

    columns=['issue_id','summary','description','reported_by','assigned_to','product','component','version','priority','Security']
    merge_df=pd.concat([sbr_df.reindex(columns=columns),nsbr_df.reindex(columns=columns)],ignore_index=True)
    merge_df.issue_id=merge_df.issue_id.map(lambda x:int(x))

    merge_df.sort_values(by='issue_id',inplace=True)
    merge_df.to_csv('../resource/mozilla/mozilla_merge.csv')



def main():
    get_data()


if __name__=='__main__':
    main()