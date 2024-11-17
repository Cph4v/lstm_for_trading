import ccxt as cx
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
apikey = os.getenv('APIKEY')
secret = os.getenv('SECRET')


def preprocess(seq_len, rolling_factor, percent=1, limit=48, apikey=apikey, secret=secret):
    
    # df = pd.read_csv(btc_path, delimiter=',', usecols=['time', 'open', 'high', 'low', 'close', 'volume'])
    exchange = cx.bybit({
        'apiKey': apikey,
        'secret': secret,
    })
    # cScsl7AqofQR383VPZ
    # B7VtUSxP3JdSoplcEFKjR83MP9yXrrm64lV7

    response = exchange.fetch_ohlcv('BTC/USDT', '1m', limit=limit)

    new_data = []

    for i in range(limit):

            unsplitted_date = exchange.iso8601(response[i][0]).split('T')

            date = '.'.join(unsplitted_date[0].split('-'))

            time = ':'.join(unsplitted_date[1].split('.')[0].split(':')[:2])

            new_data.append([date, time, response[i][1], response[i][2], response[i][3], response[i][4], response[i][5]])

    df = pd.DataFrame(new_data, columns=['date','time','open','high','low','close','volume'])

    # Replace 0 to avoid dividing by 0 later on
    df['volume'].replace(to_replace=0, method='ffill', inplace=True)
    df.sort_values('date', inplace=True)

    # Apply moving average with a window of 10 days to all columns
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].rolling(rolling_factor).mean()

    # Drop all rows with NaN values
    df.dropna(how='any', axis=0, inplace=True)

    # what percent of data can be used for process.default is 100 percent.
    df = df.tail(int(df.shape[0]*percent))

    before_pct_changes = df.copy()
    '''Calculate percentage change'''
    df['open'] = df['open'].pct_change() 
    df['high'] = df['high'].pct_change()
    df['low'] = df['low'].pct_change()
    df['close'] = df['close'].pct_change()
    df['volume'] = df['volume'].pct_change()

    df.dropna(how='any', axis=0, inplace=True)

    df_for_last = df.copy()
    '''Create indexes to split dataset'''
    times = sorted(df.index.values)
    last_10pct = sorted(df.index.values)[-int(0.1*len(times))] 
    last_20pct = sorted(df.index.values)[-int(0.2*len(times))] 

    '''Normalize price columns'''
    min_return = min(df[(df.index < last_20pct)][['open', 'high', 'low', 'close']].min(axis=0))
    max_return = max(df[(df.index < last_20pct)][['open', 'high', 'low', 'close']].max(axis=0))

    df['open'] = (df['open'] - min_return) / (max_return - min_return)
    df['high'] = (df['high'] - min_return) / (max_return - min_return)
    df['low'] = (df['low'] - min_return) / (max_return - min_return)
    df['close'] = (df['close'] - min_return) / (max_return - min_return)

    '''Normalize volume column'''
    min_volume = df[(df.index < last_20pct)]['volume'].min(axis=0)
    max_volume = df[(df.index < last_20pct)]['volume'].max(axis=0)

    df['volume'] = (df['volume'] - min_volume) / (max_volume - min_volume)


    df.drop(columns=['date', 'time'], inplace=True)
    # df = df.tail(int(df.shape[0]*percent))
    # df = df.head(int(percent))
    last_value = df['high'].values
    df = df.values


    print('df data shape: {}'.format(df.shape))

    data, label = [], []
    for i in range(seq_len, len(df)):
        data.append(df[i-seq_len:i])
        label.append(df[:, 1][i])  # Predicting 'high' price

    data, label = np.array(data), np.array(label)


    return data, label, df, last_value, df_for_last, before_pct_changes, max_return, min_return
