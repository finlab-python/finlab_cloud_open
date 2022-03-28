from fdata.financial_statements.crawler import crawl_financial_statement
from fdata.financial_statements.parser2019 import parse_statements2019
from fdata.financial_statements.fundamental_features import create_features
from fdata.file_interface import CloudDataFrameInterface
import os
import datetime
import pandas as pd
import numpy as np
import logging

# Get an instance of a logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def combine_financial_statements(path, from_year=2019, to_year=2030):

    years = [str(y) for y in list(range(from_year, to_year+1))]

    pickle_files = sorted([f for f in os.listdir(
        path) if f[-6:] == 'pickle' and f[:4] in years])

    dfs = [pd.read_pickle(os.path.join(path, p)) for p in pickle_files]

    df = pd.concat(dfs)

    def change_date(date):
        qstr = {(3, 31): '-Q1',
                (6, 30): '-Q2',
                (9, 30): '-Q3',
                (12, 31): '-Q4'}[(date.month, date.day)]

        return str(date.year) + qstr

    df.reset_index(inplace=True)
    df['date'] = df['date'].map(change_date)
    df.set_index(['stock_id', 'date'], inplace=True)
    return df


"""
refactor functions
"""


def id_f(df, codes, func):
    ret = func(df[codes[0]])
    for c in codes[1:]:
        ret.fillna(func(df[c]), inplace=True)
    return ret


def fillna_list(df_list):
    try:
        ret = df_list[0].sparse.to_dense().copy()
    except:
        ret = df_list[0].copy()

    for df in df_list[1:]:
        try:
            df = df.sparse.to_dense()
        except:
            pass
        ret.fillna(df, inplace=True)
    return ret


def split_range_season(series):

    def single_season(df):
        ret = df.diff()
        ret.iloc[0] = df.iloc[0]
        return ret

    group = series.index.get_level_values(
        0) + ' ' + series.index.get_level_values(1).str[:4]
    try:
        return series.sparse.to_dense().groupby(group).apply(single_season)
    except:
        return series.groupby(group).apply(single_season)


def combine_columns(df, items, mode='cumulative_first'):
    cumulative_item = fillna_list([df['C ' + it] for it in items])
    cumulative_item = split_range_season(cumulative_item)
    q_item = fillna_list([df['Q ' + it] for it in items])
    if mode == 'cumulative_first':
        ret = fillna_list([cumulative_item, q_item])
    else:
        ret = fillna_list([q_item, cumulative_item])
    return ret


def regular_fillna(df, flist):
    return fillna_list([df[k] for k in flist])


def 營業費用(df):
    ret = combine_columns(df, ['6000', '58500', '58000', '5230'])
    newcol = split_range_season(
        df['C 531000'] + df['C 532000'] + df['C 533000'])
    return ret.fillna(newcol)


def 研究發展費(df):
    ret = combine_columns(df, ['5233', '6300'])
    ret.fillna(0, inplace=True)
    return ret


def 預期信用減損(df):
    ret = combine_columns(df, ['6450', '7380', '5285'])
    ret.fillna(0, inplace=True)
    return ret


def 其他收益及費損淨額(df):
    ret = combine_columns(df, ['6500'])
    ret.fillna(0, inplace=True)
    return ret


def 營業利益(df):
    return combine_columns(df, ['6900', '5XXXXX', '61000', 'A00010'])


def 利息收入(df):
    return combine_columns(df, ['7100', 'A21200'])


def 其他收入(df):
    return combine_columns(df, ['7010'])


def 其他利益及損失(df):
    return combine_columns(df, ['7020'])


def 財務成本(df):
    ret = combine_columns(df, ['7050', '51700', '521200'])
    ret.fillna(0, inplace=True)
    return ret


def 採權益法之關聯企業及合資損益之份額(df):
    ret = combine_columns(df, ['7060', '4222', '601000'])
    ret.fillna(0, inplace=True)
    return ret


def 預期信用減損_損失_利益(df):
    return combine_columns(df, ['4255', 'A20300'])


def 營業外收入及支出(df):
    ret = combine_columns(df, ['7000', '600000'])
    eps_correct = df['C 9750'].notna()
    ret[~eps_correct] = np.nan
    return ret


def 稅前淨利(df):
    return combine_columns(df, ['7900', 'A00010', '902001', '61000', '61001', '62000'])


def 所得稅費用(df):
    return combine_columns(df, ['7950', '6200', '61003', '61003', '63000', '701000'])


def 繼續營業單位損益(df):
    return combine_columns(df, ['8000', '6300', '61000', '902002'])


def 停業單位損益(df):
    ret = combine_columns(df, ['8100', '62500', '6400'])
    ret.fillna(0, inplace=True)
    return ret


def 合併前非屬共同控制股權損益(df):
    ret = combine_columns(df, ['8160', '65700'])
    ret.fillna(0, inplace=True)
    return ret


def 合併總損益(df):
    return combine_columns(df, ['8200', '66000', '69000', '902005', '6500'])


def 本期綜合損益總額(df):
    return combine_columns(df, ['8500', '85000', '69700', '66000', '6700', '902006'])


def 歸屬母公司淨利損(df):
    ret = combine_columns(
        df, ['8610', '6810', '67101', '86100', '69901', '913100'])
    ret.fillna(合併總損益(df), inplace=True)
    return ret


def 歸屬非控制權益淨利損(df):
    ret = combine_columns(
        df, ['8620', '6820', '69903', '86200', '67111', '913200'])
    ret.fillna(0, inplace=True)
    return ret


def 歸屬共同控制下前手權益淨利損(df):
    ret = combine_columns(df, ['8615', '67105', '86150'])
    ret.fillna(0, inplace=True)
    return ret


def 綜合損益歸屬母公司(df):
    ret = combine_columns(
        df, ['8710', '6910', '914100', '87100', '67301', '69951'])
    ret.fillna(本期綜合損益總額(df), inplace=True)
    return ret


def 綜合損益歸屬非控制權益(df):
    ret = combine_columns(
        df, ['8720', '8400', '69953', '914200', '67311', '87200', '6920'])
    ret.fillna(0, inplace=True)
    return ret


def 綜合損益歸屬共同控制下前手權益(df):
    ret = combine_columns(df, ['8615', '67105', '86150', '69902'])
    ret.fillna(0, inplace=True)
    return ret

def to_dense(df):
    try:
        return df.sparse.to_dense()
    except:
        return df


def 每股盈餘(df):
    ret = df['Q 9750'].sparse.to_dense()
    for i in ['Q 975000', 'Q 70000', 'Q 97500', 'Q 67500', 'Q 7000']:
        ret.fillna(to_dense(df[i]), inplace=True)

    c_ret = df['C 9750'].sparse.to_dense()
    for i in ['C 975000', 'C 70000', 'C 97500', 'C 67500', 'C 7000']:
        c_ret.fillna(to_dense(df[i]), inplace=True)

#     c_ret = c_ret.groupby(c_ret.index.get_level_values(
#         1)).apply(lambda s: s - s.shift(fill_value=0))
#     concat_data['c_ret_shift']=concat_data['c_ret'].shift().fillna(0)
#     concat_data['year']=concat_data.index.get_level_values('date').year
#     concat_data['year']=(concat_data['year']-concat_data['year'].shift()).fillna(0)
#     concat_data['re_ret']=[a-b if c>0 else a for a,b,c in zip(concat_data['c_ret'],concat_data['c_ret_shift'],concat_data['year'])]
#     concat_data['ret']=concat_data['ret'].fillna(concat_data['re_ret'])
    return ret.fillna(split_range_season(c_ret))


def 加權平均股數(df):
    return ((df['3110 A1'] + df['3110 Z1']) / 2) / 10


def 每股盈餘完全稀釋(df):
    ret = combine_columns(df, ['9850'])
    ret.fillna(每股盈餘(df), inplace=True)
    return ret


def 稅前息前淨利(df):
    # return combine_columns(df, ['A00010'])
    return combine_columns(df, ['A10000']) + 財務成本(df)


def 稅前息前折舊前淨利(df):
    return combine_columns(df, ['A10000']) + 財務成本(df) + combine_columns(df, ['A20100', '59001']) + combine_columns(df, [
        'A20200', '59003'])


def 常續性稅後淨利(df):
    return combine_columns(df, ['A10000'])


tw_financial_items = {
    # 資產負債表
    '  現金及約當現金': lambda df: regular_fillna(df, ['QE 1100', 'QE 11000']),
    '  透過損益按公允價值衡量之金融資產－流動': lambda df: regular_fillna(df, ['QE 1110', 'QE 112000']),
    '  透過其他綜合損益按公允價值衡量之金融資產－流動': lambda df: regular_fillna(df, ['QE 1120', 'QE 113200', 'QE 1125', 'QE 14190']),
    '  按攤銷後成本衡量之金融資產－流動': lambda df: regular_fillna(df, ['QE 1136', 'QE 1137', 'QE 113300', 'QE 1145']),
    '  避險之金融資產－流動': lambda df: df['QE 1139'],
    '  合約資產－流動': lambda df: regular_fillna(df, ['QE 1140', 'QE 1141']),
    '  應收帳款及票據': lambda df: regular_fillna(df, ['QE 1170', 'QE 1172', 'QE 114130', 'QE 13007', 'QE 13525']).fillna(0)+regular_fillna(df, ['QE 1150', 'QE 1151', 'QE 114110', 'QE 12110', 'QE 13001']).fillna(0),
    '  其他應收款': lambda df: df['QE 1200']+regular_fillna(df, ['QE 1210', 'QE 114170']),  #(df['QE 1210'].fillna(0)).sparse.to_dense().fillna(df['QE 114170']),
    '  存貨': lambda df: regular_fillna(df, ['QE 130X', 'QE 1270']),
    '  待出售非流動資產': lambda df: regular_fillna(df, ['QE 114710', 'QE 1460', 'QE 1461']), # df['QE 114710'].sparse.to_dense().fillna(df['QE 1460']).fillna(df['QE 1461']),
    '  當期所得稅資產－流動': lambda df: regular_fillna(df, ['QE 1220', 'QE 114600', 'QE 114600', 'QE 13200', 'QE 12600', 'QE 1260']),# df['QE 1220'].sparse.to_dense().fillna(df['QE 114600']).fillna(df['QE 114600']).fillna(df['QE 13200']).fillna(df['QE 12600']).fillna(df['QE 1260']),
    '其他流動資產': lambda df: regular_fillna(df, ['QE 1470', 'QE 119000']),
    '流動資產': lambda df: regular_fillna(df, ['QE 11XX', 'QE 110000']),
    '  透過損益按公允價值衡量之金融資產－非流動': lambda df: regular_fillna(df, ['QE 1510', 'QE 122000']),
    '  透過其他綜合損益按公允價值衡量之金融資產－非流動': lambda df: regular_fillna(df, ['QE 1517', 'QE 122000']),
    '  按攤銷後成本衡量之金融資產－非流動': lambda df: regular_fillna(df, ['QE 1535', 'QE 1536', 'QE 123300', 'QE 1435']),
    '  避險之金融資產－非流動': lambda df: df['QE 1538'],
    '  合約資產－非流動': lambda df: regular_fillna(df, ['QE 1560', 'QE 1561']),
    '  採權益法之長期股權投資': lambda df: df['QE 1550'],
    '  預付投資款': lambda df: regular_fillna(df, ['QE 1960', 'QE 1422']),
    '  不動產廠房及設備': lambda df: regular_fillna(df, ['QE 1600', 'QE 16000', 'QE 125000', 'QE 18500']),
    '  商譽及無形資產合計': lambda df: (df['QE 1805'].fillna(0)+df['QE 19007'].fillna(0)+df['QE 1780'].fillna(0)+df['QE 127000'].fillna(0)+df['QE 17000'].fillna(0)+df['QE 19000'].fillna(0)),
    '    遞延所得稅資產': lambda df: regular_fillna(df, ['QE 1840', 'QE 1800', 'QE 128000', 'QE 19300', 'QE 17800', 'QE 19301', 'QE 17810']),
    '  遞延資產合計': lambda df: (df['QE 19669'].fillna(0)+df['QE 1840'].fillna(0)+df['QE 1800'].fillna(0)+df['QE 128000'].fillna(0)+df['QE 19300'].fillna(0)+df['QE 17800'].fillna(0)+df['QE 19301'].fillna(0)+df['QE 17810'].fillna(0)),
    '    使用權資產': lambda df: regular_fillna(df, ['QE 1755', 'QE 125800', 'QE 16700', 'QE 16701', 'QE 18600', 'QE 18601', 'QE 1595']),
    '    投資性不動產淨額': lambda df: df['QE 1760'],
    '  其他非流動資產': lambda df: regular_fillna(df, ['QE 1900', 'QE 129000']),
    '非流動資產': lambda df: regular_fillna(df, ['QE 15XX', 'QE 14XX', 'QE 120000']),
    '資產總額': lambda df: regular_fillna(df, ['QE 1XXX', 'QE 906001', 'QE 19999', 'QE 10000', 'QE 1XXXX']),
    '  短期借款': lambda df: regular_fillna(df, ['QE 2100', 'QE 2110', 'QE 211100', 'QE 21021', 'QE 25511', 'QE 24401']),
    '  應付商業本票∕承兌匯票': lambda df: (df['QE 23023'].fillna(0)+df['QE 211200'].fillna(0)+df['QE 25513'].fillna(0)),
    '  透過損益按公允價值衡量之金融負債－流動': lambda df: regular_fillna(df, ['QE 2120', 'QE 212000', 'QE 2123', 'QE 2140']),
    '  避險之金融負債－流動': lambda df: df['QE 2126'],
    '  按攤銷後成本衡量之金融負債－流動': lambda df: df['QE 2128'],
    '  合約負債－流動': lambda df: regular_fillna(df, ['QE 2130', 'QE 214145', 'QE 2165']),
    '  應付帳款及票據': lambda df: (df['QE 2170'].fillna(0)+df['QE 2180'].fillna(0)+df['QE 2171'].fillna(0)+df['QE 2181'].fillna(0)+df['QE 214130'].fillna(0)+df['QE 214140'].fillna(0)+df['QE 23007'].fillna(0)
                             + df['QE 2150'].fillna(0)+df['QE 2151'].fillna(0)+df['QE 2160'].fillna(0)+df['QE 2161'].fillna(0)+df['QE 214110'].fillna(0)+df['QE 21100'].fillna(0)+df['QE 23001'].fillna(0)),
    '  其他應付款': lambda df: (df['QE 2200'].fillna(0)+df['QE 2220'].fillna(0)),
    '  當期所得稅負債': lambda df: regular_fillna(df, ['QE 2230', 'QE 214600', 'QE 23200', 'QE 21700']),
    '  負債準備－流動': lambda df: regular_fillna(df, ['QE 2250', 'QE 215100']),
    '  與待出售非流動資產直接相關之負債': lambda df: df['QE 214710'],
    '  與待出售非流動資產直接相關之負債': lambda df: df['QE 214710'],
    '  租賃負債─流動': lambda df: regular_fillna(df, ['QE 2280', 'QE 216000']),
    '  一年內到期長期負債': lambda df: regular_fillna(df, ['QE 2320', 'QE 215200']),
    '  特別股負債－流動': lambda df: df['QE 2325'],
    '流動負債': lambda df: regular_fillna(df, ['QE 21XX', 'QE 210000']),
    '  透過損益按公允價值衡量之金融負債－非流動': lambda df: regular_fillna(df, ['QE 2500', 'QE 2510', 'QE 222000']),
    '  避險之金融負債－非流動': lambda df: df['QE 2511'],
    '  按攤銷後成本衡量之金融負債－非流動': lambda df: df['QE 2520'],
    '  合約負債－非流動': lambda df: regular_fillna(df, ['QE 2527', 'QE 2535']),
    '  特別股負債－非流動': lambda df: df['QE 2635'],
    '  應付公司債－非流動': lambda df: regular_fillna(df, ['QE 2530', 'QE 2531']),
    '  銀行借款－非流動': lambda df: df['QE 2102'],
    '  租賃負債－非流動': lambda df: regular_fillna(df, ['QE 2580', 'QE 2625', 'QE 226000']),
    '  負債準備－非流動': lambda df: regular_fillna(df, ['QE 2550', 'QE 225100']),
    '  遞延貸項': lambda df: df['C A32250'],
    '  應計退休金負債': lambda df: regular_fillna(df, ['QE 2205', 'C 531060']),
    '  遞延所得稅': lambda df: regular_fillna(df, ['QE 2570', 'QE 228000']),
    '非流動負債': lambda df: regular_fillna(df, ['QE 25XX', 'QE 220000']),
    '負債總額': lambda df: regular_fillna(df, ['QE 2XXX', 'QE 906003', 'QE 29999', 'QE 20000', 'QE 2XXXX']),
    '    普通股股本': lambda df: regular_fillna(df, ['QE 3110', 'QE 301010', 'QE 31101']),
    '    特別股股本': lambda df: regular_fillna(df, ['QE 3120', 'QE 31103']),
    '    預收股款': lambda df: regular_fillna(df, ['QE 3998', 'QE 399998', 'QE 39997', 'QE 38112', 'QE 39900']),
    '    待分配股票股利': lambda df: regular_fillna(df, ['QE 3150', 'QE 301070', 'QE 31107', 'QE 31400']),
    '    換股權利證書': lambda df: df['QE 3130'],
    '  股本': lambda df: regular_fillna(df, ['QE 3100', 'QE 301010', 'QE 31100']),
    '  資本公積合計': lambda df: regular_fillna(df, ['QE 3200', 'QE 302000']),
    '    法定盈餘公積': lambda df: regular_fillna(df, ['QE 3310', 'QE 304010', 'QE 32001', 'QE 33100']),
    '  資本公積合計': lambda df: regular_fillna(df, ['QE 3200', 'QE 302000']),
    '    未分配盈餘': lambda df: regular_fillna(df, ['QE 3350', 'QE 304040', 'QE 304040', 'QE 32011', 'QE 3330', 'QE 33300']),
    '  保留盈餘': lambda df: regular_fillna(df, ['QE 3300', 'QE 304000', 'QE 32000', 'QE 33000']),
    '  其他權益': lambda df: regular_fillna(df, ['QE 3400', 'QE 305000', 'QE 34000', 'QE 32500']),
    '  庫藏股票帳面值': lambda df: regular_fillna(df, ['QE 3500', 'QE 305500', 'QE 32600']),
    '母公司股東權益合計': lambda df: regular_fillna(df, ['31XX Z1', '3XXX Z1', 'QE 31000', 'QE 300000']),
    '共同控制下前手權益': lambda df: regular_fillna(df, ['35XX Z1', 'QE 305600', 'Q 69902']),
    '合併前非屬共同控制股權': lambda df: df['355X Z1'],
    '非控制權益': lambda df: regular_fillna(df, ['36XX Z1', 'QE 39500', 'Q 69903', 'QE 306000', 'QE 36000', 'QE 38001']),
    '股東權益總額': lambda df: regular_fillna(df, ['3XXX Z1', 'QE 906004', 'QE 3XXXX']),
    '負債及股東權益總額': lambda df: regular_fillna(df, ['QE 3X2X', 'QE 906002', 'QE 3X2XX']),

    # 損益表
    '營業收入淨額': lambda df: combine_columns(df, ['4000', '41000', '400000']),
    '營業成本': lambda df: combine_columns(df, ['5000']),
    '營業毛利': lambda df: combine_columns(df, ['5900']),
    '營業費用': 營業費用,
    '  研究發展費': 研究發展費,
    '推銷費用': lambda df: combine_columns(df, ['6100', '5231']),
    '管理費用': lambda df: combine_columns(df, ['6200']),
    '  預期信用減損（損失）利益－營業費用': 預期信用減損,
    '其他收益及費損淨額': 其他收益及費損淨額,
    '營業利益': 營業利益,
    # '其他利益及損失':其他利益及損失,
    '財務成本': 財務成本,
    '採權益法之關聯企業及合資損益之份額': 採權益法之關聯企業及合資損益之份額,
    # '預期信用減損（損失）利益': 預期信用減損_損失_利益,
    '營業外收入及支出': 營業外收入及支出,
    '稅前淨利': 稅前淨利,
    '所得稅費用': 所得稅費用,
    '繼續營業單位損益': 繼續營業單位損益,
    '停業單位損益': 停業單位損益,
    '合併前非屬共同控制股權損益': 合併前非屬共同控制股權損益,
    '合併總損益': 合併總損益,
    '本期綜合損益總額': 本期綜合損益總額,
    '歸屬母公司淨利（損）': 歸屬母公司淨利損,
    '歸屬非控制權益淨利（損）': 歸屬非控制權益淨利損,
    '歸屬共同控制下前手權益淨利（損）': 歸屬共同控制下前手權益淨利損,
    '綜合損益歸屬母公司': 綜合損益歸屬母公司,
    '綜合損益歸屬非控制權益': 綜合損益歸屬非控制權益,
    '綜合損益歸屬共同控制下前手權益': 綜合損益歸屬共同控制下前手權益,
    '每股盈餘': 每股盈餘,
    # '加權平均股數': 加權平均股數,
    # '每股盈餘－完全稀釋': 每股盈餘完全稀釋


    # 現金流量表
    '繼續營業單位稅前淨利（淨損）': lambda df: combine_columns(df, ['A00010', '7900', '902001', '61000', '61001', '62000']),
    '本期稅前淨利（淨損）': lambda df: combine_columns(df, ['A10000', '64001']),
    '折舊費用': lambda df: combine_columns(df, ['A20100', '59001']),
    '攤銷費用': lambda df: combine_columns(df, ['A20200', '59003']),
    '呆帳費用提列（轉列收入）數': lambda df: combine_columns(df, ['A20300']),
    '透過損益按公允價值衡量金融資產及負債之淨損失（利益）': lambda df: combine_columns(df, ['A20400']),
    '利息費用': lambda df: combine_columns(df, ['A20900', '7510', '51000', '5010']),
    '利息收入': lambda df: combine_columns(df, ['A21200', '421200', '41000', '41510', '4240', '4010']),
    '股利收入': lambda df: combine_columns(df, ['A21300', '7130', '421300', '4221', '602491']),
    '採用權益法認列之關聯企業及合資損失（利益）之份額': lambda df: combine_columns(df, ['A22300']),
    '處分及報廢不動產、廠房及設備損失（利益）': lambda df: combine_columns(df, ['A22500']),
    '處分無形資產損失（利益）': lambda df: combine_columns(df, ['A22800']),
    '處分投資損失（利益）': lambda df: combine_columns(df, ['A23100']),
    '非金融資產減損迴轉利益': lambda df: combine_columns(df, ['A23800']),
    '未實現銷貨利益（損失）': lambda df: combine_columns(df, ['A23900']),
    '已實現銷貨損失（利益）': lambda df: combine_columns(df, ['A24000']),
    '未實現外幣兌換損失（利益）': lambda df: combine_columns(df, ['A24100']),
    '收益費損項目合計': lambda df: combine_columns(df, ['A20010']),
    '應收帳款（增加）減少': lambda df: combine_columns(df, ['A31150']),
    '應收帳款－關係人（增加）減少': lambda df: combine_columns(df, ['A31160']),
    '存貨（增加）減少': lambda df: combine_columns(df, ['A31200']),
    '與營業活動相關之資產之淨變動合計': lambda df: combine_columns(df, ['A31000', 'A61000', 'A71000', 'A41000', 'A51000', 'A91000']),
    '應付帳款增加（減少）': lambda df: combine_columns(df, ['A32150', 'A62230']),
    '應付帳款－關係人增加（減少）': lambda df: combine_columns(df, ['A32160']),
    '與營業活動相關之負債之淨變動合計': lambda df: combine_columns(df, ['A32000', 'A62000', 'A72000', 'A42000', 'A52000', 'A92000']),
    '營運產生之現金流入（流出）': lambda df: combine_columns(df, ['A33000']),
    '退還（支付）之所得稅': lambda df: combine_columns(df, ['A33500']),
    '營業活動之淨現金流入（流出）': lambda df: combine_columns(df, ['AAAA']),
    '取得透過其他綜合損益按公允價值衡量之金融資產': lambda df: combine_columns(df, ['B00010']),
    '處分透過其他綜合損益按公允價值衡量之金融資產': lambda df: combine_columns(df, ['B00020']),
    '取得不動產、廠房及設備': lambda df: combine_columns(df, ['B02700']),
    '處分不動產、廠房及設備': lambda df: combine_columns(df, ['B02800']),
    '取得無形資產': lambda df: combine_columns(df, ['B04500']),
    '處分無形資產': lambda df: combine_columns(df, ['B04600']),
    '收取之利息': lambda df: combine_columns(df, ['A33100', 'B07500', 'AC0400']),
    '收取之股利': lambda df: combine_columns(df, ['A33200', 'B07600', 'AC0200']),
    '其他投資活動': lambda df: combine_columns(df, ['B09900']),
    '投資活動之淨現金流入（流出）': lambda df: combine_columns(df, ['BBBB']),
    '短期借款增加': lambda df: combine_columns(df, ['C00100']),
    '短期借款減少': lambda df: combine_columns(df, ['C00200']),
    '應付短期票券增加': lambda df: combine_columns(df, ['C00500']),
    '應付短期票券減少': lambda df: combine_columns(df, ['C00600']),
    '發行公司債': lambda df: combine_columns(df, ['C01200']),
    '償還公司債': lambda df: combine_columns(df, ['C01300']),
    '舉借長期借款': lambda df: combine_columns(df, ['C01600']),
    '償還長期借款': lambda df: combine_columns(df, ['C01700']),
    '存入保證金增加': lambda df: combine_columns(df, ['C03000']),
    '存入保證金減少': lambda df: combine_columns(df, ['C03100']),
    '發放現金股利': lambda df: combine_columns(df, ['C04500']),
    '支付之利息': lambda df: combine_columns(df, ['C05600', 'A33300', 'AC0300']),
    '籌資活動之淨現金流入（流出）': lambda df: combine_columns(df, ['CCCC']),
    '本期現金及約當現金增加（減少）數': lambda df: combine_columns(df, ['EEEE']),
    '期初現金及約當現金餘額': lambda df: combine_columns(df, ['E00100']),
    '期末現金及約當現金餘額': lambda df: combine_columns(df, ['E00200']),
    '資產負債表帳列之現金及約當現金': lambda df: combine_columns(df, ['E00210']),
    'board_approval_date': lambda df: df['board_approval_date'],
    'audit_date': lambda df: df['audit_date'],
}


# move special symbols in financial_items
def update_col_name(col):
    replace_list = ['（', '）', '－', '、']
    for r in replace_list:
        col = col.replace(r, '_')
    while col[-1] == '_':
        col = col[:-1]
    return col


def add_to_database(refactor_df, add_timestamps):
    old_df = pd.read_feather(
        'tw_financial_statements/financial_statement.feather')
    old_df.set_index(['stock_id', 'date'], inplace=True)

    new_df = pd.concat([old_df, refactor_df])
    new_df = new_df[~new_df.index.duplicated(keep='last')]

    new_df = new_df.reset_index()
    new_df = pd.DataFrame(
        {k: v for k, v in zip(new_df.columns, new_df.T.values)})

    # register download dates to new index
    if add_timestamps:
        new_download_dates = pd.Series(pd.NaT, index=new_df.index)
        if 'download_date' in old_df.columns:
            new_download_dates.fillna(old_df.download_date)
        new_index = set(new_df.index) - set(old_df.index)
        new_download_dates.loc[new_index] = datetime.datetime.now().date()
        new_df['download_date'] = new_download_dates

    new_df.to_feather('tw_financial_statements/financial_statement.feather')


def index_str_to_date(df):
    def str_to_date(d):

        month = {
            '1': '-05-15',
            '2': '-08-14',
            '3': '-11-14',
            '4': '-03-31',
        }[d[-1]]

        year = d[:4] if d[-1] != '4' else str(int(d[:4])+1)
        return pd.to_datetime(year + month)

    df = df.copy()
    df['date'] = df.date.map(str_to_date)
    return df


def build2019():
    """
    build tw_financial_statements/refactor_2019_2020.pickle
    """

    df = combine_financial_statements('tw_financial_statements')

    # refactor_df financial_items by func in financial_items
    logger.info('start refactor data.')
    refactor_df = pd.DataFrame()
    for key, func in tw_financial_items.items():
        col_name = key
        if ' ' in key:
            col_name = key.replace(' ', '')
        new_name = update_col_name(col_name)
        refactor_df[new_name] = func(df)
    refactor_df['營業毛利'].fillna(
        refactor_df['繼續營業單位稅前淨利_淨損'] + refactor_df['營業費用'], inplace=True)
    refactor_df.to_pickle('tw_financial_statements/refactor_2019_2020.pickle')


def build2013_2019():
    """
    build tw_financial_statements/financial_statement.feather
    """

    build2019()

    df_post = pd.read_pickle(
        'tw_financial_statements/refactor_2019_2020.pickle')
    df_pre = pd.read_pickle(
        'tw_financial_statements/refactor_2013_2018.pickle')

    df_total = df_pre.append(df_post).sort_index()
    df_total = df_total[~df_total.index.duplicated(keep='last')]
    df_total = df_total.apply(lambda s: s.sparse.to_dense(
    ) if isinstance(s.dtype, pd.SparseDtype) else s)
    df_total.reset_index().to_feather(
        'tw_financial_statements/financial_statement.feather')


def main(year=2020, season=4, add_timestamps=False):
    """tw_financial_statement download,refactor,upload to gcp
    Args:
      year(int):financial_statement year,ex:2021
      season(int):financial_statement season,ex:1
    """

    # download ex:20201.zip to tw_financial_statements dir
    crawl_financial_statement(year, season, 'tw_financial_statements')

    # parse ex:20201.zip statement to 20201.pickle
    parse_statements2019(year, season, 'tw_financial_statements')

    # combine2019~recent-year data
    build2019()

    df = pd.read_pickle(
        'tw_financial_statements/refactor_2019_2020.pickle')

    add_to_database(df, add_timestamps)

    df_total = pd.read_feather(
        'tw_financial_statements/financial_statement.feather')

    logger.info('start create fundamental_features.')
    fundamental_features = create_features(df_total)
    fundamental_features = fundamental_features.reset_index()
    fundamental_features.to_feather(
        'tw_financial_statements/fundamental_features.feather')
    logger.info('Finish refactor and save data at local.')

    #  upload data to gcp storage,note: set key env,os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
    logger.info('Start upload financial_statement data to gcp storage.')
    CloudDataFrameInterface('finlab_tw_stock').write_df(
        index_str_to_date(df_total), 'financial_statement')
    logger.info('Start upload fundamental_features data to gcp storage.')
    CloudDataFrameInterface('finlab_tw_stock').write_df(
        index_str_to_date(fundamental_features), 'fundamental_features')
    logger.info('Finish upload data to gcp storage.')


def update_financial_statement(add_timestamps=False):

    print('udpate financial statements')
    month = datetime.datetime.now().month
    year = datetime.datetime.now().year
    day = datetime.datetime.now().day

    if month >= 10:
        season = 3
    elif month >= 7:
        season = 2
    elif month >= 4:
        season = 1
    else:
        season = 4

    if season == 4:
        year -= 1

    prev_season = (season + 2) % 4 + 1
    prev_year = year if season != 1 else year - 1

    print('update ', prev_year, prev_season)
    print('update ', year, season)

    main(prev_year, prev_season, add_timestamps)
    main(year, season, add_timestamps)
