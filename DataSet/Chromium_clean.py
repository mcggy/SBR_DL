import pandas as pd
from parse import preprocess_br,remove_code
import re
import os

def clean_str(string):
    print(string)
    string = re.sub("\\s+", " ", string)

    return string.strip()

def dealNan(x):
    if type(x) == float:
        return False
    return True

def clean_data():
    path='../resource/chromium/chromium.csv'
    df=pd.read_csv(path,encoding='iso-8859-1')
    df=df[df.apply(lambda x:dealNan(x['bug_title']),axis=1)]
    df['bug_description'] = df['bug_description'].map(lambda x: ' ' if type(x)==float else x)

    df['bug_title']=df['bug_title'].map(lambda x: clean_str(x))
    df['bug_description'] = df['bug_description'].map(lambda x: clean_str(x))


    file, _ = os.path.splitext(path)
    pandas_data_file = file + '_update.csv'
    df.to_csv(pandas_data_file)

def proposed_data():
    path = '../resource/chromium/chromium_update.csv'
    df = pd.read_csv(path, encoding='iso-8859-1')
    df['bug_title_pro']=df['bug_title'].map(lambda x:' '.join(preprocess_br(x)))
    df['bug_description_pro'] = df['bug_description'].map(lambda x: ' '.join(preprocess_br(x)))
    file, _ = os.path.splitext(path)
    pandas_data_file = file + '_process.csv'
    df.to_csv(pandas_data_file)

def judge_label(label):
    try:
        if int(label)==0:
            return True
        elif int(label)==1:
            return True
        else:
            return False
    except Exception:
        return False

def update_columns():
    path = '../resource/chromium/chromium_update_process.csv'
    df = pd.read_csv(path, encoding='iso-8859-1')
    df = df.rename(columns={'bug_label': 'Security','bug_title':'summary','bug_description':'description','bug_id':'issue_id'})
    columns=['issue_id','summary','description','bug_title_pro','bug_description_pro','Security']

    df=df.reindex(columns=columns)
    df = df[df.apply(lambda x: dealNan(x['bug_title_pro']), axis=1)]
    df['bug_description_pro'] = df['bug_description_pro'].map(lambda x:' ' if type(x)==float else ' '+x)

    df = df[df.apply(lambda x: judge_label(x['Security']), axis=1)]
    df['issue_id'] = df['issue_id'].map(lambda x: int(x))
    df.sort_values(by='issue_id', inplace=True)

    file, _ = os.path.splitext(path)
    pandas_data_file = file + '2.csv'
    df.to_csv(pandas_data_file)
def remove_code_t():
    string='https://www.baidu.com/s?ie=utf-8&f=8&rsv_bp=1&ch=7&tn=9'
    str = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '',
                          string)
    print(str)

if __name__ == '__main__':
    # clean_data()
    # proposed_data()
    update_columns()
    # remove_code_t()