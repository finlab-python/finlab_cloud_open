import zipfile
import re
import os
import numpy as np
import pandas as pd
from io import BytesIO

convert = {i: int(i) for i in '1234567890'}
convert = {**convert, **{i:j for i,j in zip('０１ㄧ零○一二三四五六七八九１２３４５６７８９', '01100123456789123456789')}}
convert = {**convert, **{'十':'_', '百': '', '千':'', '年': '-', '月': '-', '日': '', '-': '-', '-': '-', '-': '-'}}

def convert_to_date(s):
  ret = ''
  for c in s:
    ret += str(convert[c])

  ret = ret.split('-')
  if '_' in ret[-2]:
    ret[-2] = ret[-2].replace('_', '1')

  if '_' in ret[-1]:

    # 10
    if len(ret[-1]) == 1:
      ret[-1] = '10'

    # 11~19
    elif len(ret[-1]) == 2 and ret[-1][0] == '_':
      ret[-1] = '1' + ret[-1][-1]

    # 21-29
    elif len(ret[-1]) == 3 and ret[-1][1] == '_':
      ret[-1] = ret[-1][0] + ret[-1][2]

    # 20, 30
    elif len(ret[2]) == 2:
      ret[-1] = ret[-1][0] + '0'

    else:
      print('cannot convert', s, ' with _ ', ret[2])
      return None

  return '-'.join(ret[-2:])



def add_to_financial_statements(df, dname='./tw_financial_statements'):

  fpath = os.path.join(dname, 'financial_statement.feather')

  table = pd.read_feather(fpath)
  table.set_index(['stock_id', 'date'], inplace=True)

  for c in df.columns:
    if c in table.columns:
      table[c].fillna(df[c], inplace=True)
    else:
      table[c] = df[c]

  table.reset_index().to_feather(fpath)


def extract_audit_dates(year, season, path='./tw_financial_statements/'):
  if year >= 2019:
    ret = extract_audit_dates2019(year, season, path)
  else:
    ret = extract_audit_dates2013(year, season, path)

  add_to_financial_statements(ret, path)

def extract_audit_dates2019(htmls, year, season):

  def str_to_date(s):
    if s:
      if season == 4:
        ret = str(year+1) + '-' + s
      else:
        ret = str(year) + '-' + s

      try:
        ret = pd.to_datetime(ret)
        return ret.to_pydatetime().toordinal()
      except Exception  as e:
        print('**ERROR: ' + str(e))
        print('**INPUT: ' + str(s))
    return np.nan

  date1, date2 = None, None

  for html in htmls:

    is_found = html.iloc[:, 0].astype(str).str.find('通過財報之日期及程序')
    if (is_found != -1).any():
      text = html[is_found != -1].iloc[0].iloc[1].replace('\n', '').replace(' ', '')
      matches = re.findall("[０零○１ㄧ一二三四五六七八九十百千\d]+年[零○０１ㄧ一二三四五六七八九十百千\d]+月[零○０ㄧ一二三四五六七八九十百千\d]+日", text)
      if matches:
        date1 = convert_to_date(matches[-1])

    is_found = html.iloc[:, 0].astype(str).str.find('核閱或查核日期')
    if (is_found != -1).any():
      date2 = (convert_to_date(html[is_found != -1].iloc[0].iloc[1]))



  return str_to_date(date1), str_to_date(date2)

def extract_audit_dates2013(year, season, path='./tw_financial_statements/'):

  def str_to_date(s):
    if s:
      if season == 4:
        ret = str(year+1) + '-' + s
      else:
        ret = str(year) + '-' + s

      try:
        ret = pd.to_datetime(ret)
        return ret.to_pydatetime().toordinal()
      except Exception  as e:
        print('**ERROR: ' + str(e))
        print('**INPUT: ' + str(s))
    return np.nan
  zip_path = os.path.join(path, f'{year}{season}.zip')
  print(zip_path)
  izip = zipfile.ZipFile(zip_path)


  date1s = {}
  date2s = {}

  for name in sorted(izip.namelist()):

      date1, date2 = None, None

      if name[-4:] != '.xml':
          continue

      sid = name.split('-')[5]
      print(sid)
      text = izip.read(name).decode('utf-8')

      s = text.find('ReviewAuditDate')
      e = text.rfind('ReviewAuditDate')
      matches = re.findall('\d+-\d+-\d+', text[s:e])

      if matches:
        date1 = convert_to_date(matches[-1])

      s = text.find('notes:DateAndProceduresOfAuthorisationForIssueOfFinancialStatements')
      e = text.rfind('notes:DateAndProceduresOfAuthorisationForIssueOfFinancialStatements')
      matches = re.findall("[０零○１ㄧ一二三四五六七八九十百千\d]+年[零○０１ㄧ一二三四五六七八九十百千\d]+月[零○０ㄧ一二三四五六七八九十百千\d]+日", text[s:e])

      if matches:
        date2 = convert_to_date(matches[-1])

      print(year, season, sid, date1, date2)
      date1s[sid] = date1
      date2s[sid] = date2

  df_date = pd.DataFrame({
    'board_approval_date': pd.Series(date1s).map(str_to_date),
    'audit_date':pd.Series(date2s).map(str_to_date),
  }).reset_index()

  print(df_date)

  df_date.rename(columns={'index': 'stock_id'}, inplace=True)

  print(df_date)

  df_date['date'] = str(year) + '-Q' + str(season)

  print(df_date)

  df_date.set_index(['stock_id', 'date'], inplace=True)
  return df_date
