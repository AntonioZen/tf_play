import logging
import os
import pandas as pd
import time
import report_mail
import brain
# import  bitfinex_test
from utils import init_logger
import zipfile


import requests

from datetime import datetime
# from bittrex_wrapper import get_pairs_history

import platform

print(platform.python_version())

start_time = time.time()
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

##################################################################
# bitfinex top market: 28.05.2018
# BTCUSD	7,495.1	1.0%	7,560.0	7,392.4	134,072,145 USD
# EOSUSD	12.158	2.4%	12.650	11.694	114,860,519 USD
# ETHUSD	564.03	0.7%	585.44	549.08	104,033,713 USD
# BCHUSD	998.80	1.7%	1,029.0	958.20	42,099,894 USD
# IOTUSD	1.6199	7.5%	1.6620	1.4730	34,487,864 USD
# XRPUSD	0.60686	2.3%	0.62839	0.58348	20,021,781 USD
# LTCUSD	120.22	-0.5%	123.00	118.45	9,648,060 USD
# XMRUSD	159.05	-2.8%	163.69	151.90	7,519,251 USD
# NEOUSD	52.759	0.8%	53.701	51.586	4,249,418 USD
#
##################################################################

def get_pair_data(pair = 'BTCUSD', data_set_size = 10, current_timestamp = int(round(time.time() * 1000)), batch_size = 120):

    half_hour = 1800000

    full = pd.DataFrame()

    for i in range(data_set_size):
        start_param = current_timestamp - half_hour * batch_size * (i + 1)
        end_param = current_timestamp - half_hour * batch_size * i
        print(datetime.fromtimestamp(start_param / 1e3).strftime('%Y-%m-%d %H:%M:%S.%f+%Z'),
              datetime.fromtimestamp(end_param / 1e3).strftime('%Y-%m-%d %H:%M:%S.%f+%Z'))

        r = requests.get('https://api.bitfinex.com/v2/candles/trade:30m:t' + pair + '/hist',
                         params={'start': start_param, 'end': end_param})
        data = r.json()
        df = pd.DataFrame(data)

        full = full.append(df, ignore_index=True)

        time.sleep(10)

    full.columns = ['MTS', 'OPEN', 'CLOSE', 'HIGH', 'LOW', 'VOLUME']
    full = full.sort_values(by=['MTS'])
    return full


def get_market_data(top_market, data_set_size = 300, current_timestamp = int(round(time.time() * 1000)), batch_size = 120):

    market_data = pd.DataFrame()

    for pair in top_market:
        data = get_pair_data(pair=pair, data_set_size=data_set_size, current_timestamp=current_timestamp,
                             batch_size=batch_size)

        market_data[pair] = data.CLOSE

    return market_data


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))


def create_basis():
    pass


def create_tf_dataset_pd(pair_history_data, test_size=1000):
    """
    :param pair_history_data:
    :param test_size:
    :return:
    """
    train_features = pd.DataFrame(columns=list(range(500)))
    train_labels = pd.DataFrame(columns=list(range(10)))
    test_features = pd.DataFrame(columns=list(range(500)))
    test_labels = pd.DataFrame(columns=list(range(10)))

    n = len(pair_history_data)

    for window in range(n - 51):  # (len(data) - 50):
        wend = n - window
        wstart = wend - 51

        price_window = pair_history_data.iloc[wstart:wend]
        price_window[9] = 1 # np.arange(1, 1 + len(price_window) * 0.002, 0.002)
        price_image = price_window.transpose()

        X1_norm = pd.DataFrame()

        for i in range(51):
            X1_norm['X%d' % i] = price_image[price_image.columns[i]] / price_image[price_image.columns[49]]

        X1_norm.iloc[9] = 1  # 1.002
        feature1 = X1_norm.iloc[:, 0:50].as_matrix().reshape(500)
        label1 = X1_norm.X50.as_matrix().reshape(10)

        if window < test_size:
            test_features.loc[window] = feature1
            test_labels.loc[window] = label1
        else:
            train_features.loc[window] = feature1
            train_labels.loc[window] = label1

        train_features = train_features.fillna(1)
        train_labels = train_labels.fillna(1)
        test_features = test_features.fillna(1)
        test_labels = test_labels.fillna(1)

    return train_features, train_labels, test_features, test_labels


def create_tf_scoringset_pd(pair_history_data):
    features = pd.DataFrame(columns=list(range(500)))

    price_window = pair_history_data.tail(50)
    price_window[9] = 1
    price_image = price_window.transpose()
    X1_norm = pd.DataFrame()
    for i in range(50):
        X1_norm['X%d' % i] = price_image[price_image.columns[i]] / price_image[price_image.columns[49]]

    feature = X1_norm.iloc[:, 0:50].as_matrix().reshape(500)
    features.loc[0] = feature
    features = features.fillna(1)

    return features


if __name__ == '__main__':

    logger = init_logger(BASE_PATH + '/logs/brain.log')

    top_market = [
        'BTCUSD',
        'EOSUSD',
        'ETHUSD',
        'BCHUSD',
        'IOTUSD',
        'XRPUSD',
        'LTCUSD',
        'XMRUSD',
        'NEOUSD'
    ]

    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    data = get_market_data(top_market, data_set_size = 400)
    data.to_csv(BASE_PATH + '/datasets/bitfinex_markets_summaries ' + now + '.csv', index=False)
    ############################################################################################
    # scoring_set = create_tf_scoringset_pd(data)
    ############################################################################################
    X_train, Y_train, X_test, Y_test = create_tf_dataset_pd(data)
    #
    X_train.to_csv(BASE_PATH + '/datasets/X_train ' + now + '.csv', index=False)
    Y_train.to_csv(BASE_PATH + '/datasets/Y_train ' + now + '.csv', index=False)
    X_test.to_csv(BASE_PATH + '/datasets/X_test ' + now + '.csv', index=False)
    Y_test.to_csv(BASE_PATH + '/datasets/Y_test ' + now + '.csv', index=False)
    ############################################################################################
    # X_train = pd.read_csv(BASE_PATH + '/datasets/X_train ' + '2018-06-04T13:54:59' + '.csv')
    # Y_train = pd.read_csv(BASE_PATH + '/datasets/Y_train ' + '2018-06-04T13:54:59' + '.csv')
    # X_test = pd.read_csv(BASE_PATH + '/datasets/X_test ' + '2018-06-04T13:54:59' + '.csv')
    # Y_test = pd.read_csv(BASE_PATH + '/datasets/Y_test ' + '2018-06-04T13:54:59' + '.csv')
    train_dataset = X_train, Y_train, X_test, Y_test
    ############################################################################################
    print('-'*30)

    monkey = brain.MonkeyEngine(name = 'Bitfinex_Bill', version='_1_0', n_input=500, n_classes=10, batch_size=128, learn_rate=0.00001)

    results, results_test, portfolio_vector, current_cost, test_cost = monkey.train(train_dataset=train_dataset, predict_features=None, epochs=1000, save=True, restore=False)

    top_market.append(('USDTUSDT'))

    results_df = pd.DataFrame(results)
    results_df_test = pd.DataFrame(results_test)

    results_df = results_df.sort_index(ascending=False)
    results_df_test = results_df_test.sort_index(ascending=False)
    results_df.to_csv(BASE_PATH + '/datasets/output_' + now + '.csv')
    results_df_test.to_csv(BASE_PATH + '/datasets/output_test_' + now + '.csv')

    # zip model:
    model_arc_path = BASE_PATH + '/model/model_' + now + '.zip'
    zipf = zipfile.ZipFile(model_arc_path, 'w', zipfile.ZIP_DEFLATED)
    zipdir(BASE_PATH + '/trained_models', zipf)
    zipf.close()

    logging.info(results_df)
