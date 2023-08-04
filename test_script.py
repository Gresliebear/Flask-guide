
import requests
import pandas as pd
import time


def get_ticker_data(ticker, time):

    if time == 'daily':
        data = get_daily_time_series(api_key, symbol, outputsize='compact', datatype='json')
    elif time == 'weekly':

    if time == 'monthly':
        data =


def get_ticker_data_graph_item(ticker):
    """Get ticker data for graph item."""
    # Get ticker data
    ticker_data = get_ticker_data(ticker)
    # Get ticker data for graph item
    ticker_data_graph_item = get_ticker_data_graph_item_from_ticker_data(ticker_data)
    return ticker_data_graph_item


from alpha_vantage.timeseries import TimeSeries 

def get_earnings_data(api_key, list_of_tickers):
    """Fetch earnings data for a list of tickers using Alpha Vantage API.

    Parameters:
        api_key (str): API key from Alpha Vantage.
        list_of_tickers (list): List of ticker symbols.

    Returns:
        pandas.DataFrame: DataFrame containing earnings data for all tickers.
    """
    # Initialize an empty DataFrame to store the data
    earnings_data = pd.DataFrame()

    # Initialize API client
    ts = TimeSeries(key=api_key)

    for idx, ticker in enumerate(list_of_tickers, 1):
        try:
            print(f"Fetching data for {ticker} ({idx}/{len(list_of_tickers)})...")
            # Get quarterly earnings data for the current ticker
            _, data = ts.get_earnings(symbol=ticker)
            if data:
                # Extract the 'quarterlyEarnings' list from the response
                quarterly_earnings = data['quarterlyEarnings']
                if isinstance(quarterly_earnings, list) and len(quarterly_earnings) > 0:
                    # Convert the list to a DataFrame and append it to the earnings_data DataFrame
                    df = pd.DataFrame(quarterly_earnings)
                    df['ticker'] = ticker
                    earnings_data = earnings_data.append(df, ignore_index=True)
                else:
                    print(f"No quarterly earnings data found for {ticker}")
            else:
                print(f"No data found for {ticker}")
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")

        # Add a short delay after every 4 requests to comply with API rate limits
        if idx % 4 == 0 and idx < len(list_of_tickers):
            print("Waiting for 60 seconds...")
            time.sleep(60)

    return earnings_data

def get_daily_time_series(api_key, symbol, outputsize='compact', datatype='json'):
    """
    Fetch daily time series data for a specific equity from Alpha Vantage.

    Parameters:
        api_key (str): API key from Alpha Vantage.
        symbol (str): The name of the equity (e.g., 'IBM', 'AAPL', 'MSFT').
        outputsize (str): 'compact' (default) or 'full' to specify the data size.
        datatype (str): 'json' (default) or 'csv' to specify the data format.

    Returns:
        pandas.DataFrame: DataFrame containing the daily time series data.
    """
    base_url = 'https://www.alphavantage.co/query'
    function = 'TIME_SERIES_DAILY'

    # Build the API request URL
    params = {
        'function': function,
        'symbol': symbol,
        'outputsize': outputsize,
        'datatype': datatype,
        'apikey': api_key
    }

    # Make the API call
    response = requests.get(base_url, params=params)

    # Check if the API call was successful
    if response.status_code == 200:
        data = response.json()
        if 'Time Series (Daily)' in data:
            # Extract the daily time series data from the response
            daily_time_series = data['Time Series (Daily)']
            # Convert the data to a DataFrame
            df = pd.DataFrame.from_dict(daily_time_series, orient='index')
            # Convert date strings to datetime objects
            df.index = pd.to_datetime(df.index)
            # Convert data values to numeric types
            df = df.apply(pd.to_numeric)
            return df
        else:
            print("Error: Data not found in the API response.")
    else:
        print(f"Error: API request failed with status code {response.status_code}.")
    return None


def get_weekly_time_series(api_key, list_of_tickers, datatype='json'):
    """
    Fetch weekly time series data for a list of equities from Alpha Vantage.

    Parameters:
        api_key (str): API key from Alpha Vantage.
        list_of_tickers (list): List of ticker symbols.
        datatype (str): 'json' (default) or 'csv' to specify the data format.

    Returns:
        pandas.DataFrame: DataFrame containing the weekly time series data for all tickers.
    """
    base_url = 'https://www.alphavantage.co/query'
    function = 'TIME_SERIES_WEEKLY'

    # Initialize an empty DataFrame to store the combined data
    combined_data = pd.DataFrame()

    for ticker in list_of_tickers:
        # Build the API request URL for the current ticker
        params = {
            'function': function,
            'symbol': ticker,
            'datatype': datatype,
            'apikey': api_key
        }

        # Make the API call
        response = requests.get(base_url, params=params)

        # Check if the API call was successful
        if response.status_code == 200:
            data = response.json()
            if 'Weekly Time Series' in data:
                # Extract the weekly time series data from the response
                weekly_time_series = data['Weekly Time Series']
                # Convert the data to a DataFrame
                df = pd.DataFrame.from_dict(weekly_time_series, orient='index')
                # Convert date strings to datetime objects
                df.index = pd.to_datetime(df.index)
                # Convert data values to numeric types
                df = df.apply(pd.to_numeric)
                # Add a 'ticker' column to identify the data for each ticker
                df['ticker'] = ticker
                # Append the data for the current ticker to the combined DataFrame
                combined_data = combined_data.append(df)
            else:
                print(f"Error: Data not found for {ticker}.")
        else:
            print(f"Error: API request failed for {ticker} with status code {response.status_code}.")

    return combined_data

def get_time_series(api_key, function, list_of_tickers, datatype='json'):
    """
    Fetch time series data for a list of equities from Alpha Vantage.

    Parameters:
        api_key (str): API key from Alpha Vantage.
        function (str): The time series function to use ('TIME_SERIES_WEEKLY_ADJUSTED' or 'TIME_SERIES_MONTHLY').
        list_of_tickers (list): List of ticker symbols.
        datatype (str): 'json' (default) or 'csv' to specify the data format.

    Returns:
        pandas.DataFrame: DataFrame containing the time series data for all tickers.
    """
    base_url = 'https://www.alphavantage.co/query'

    # Initialize an empty DataFrame to store the combined data
    combined_data = pd.DataFrame()

    for idx, ticker in enumerate(list_of_tickers, 1):
        # Build the API request URL for the current ticker
        params = {
            'function': function,
            'symbol': ticker,
            'datatype': datatype,
            'apikey': api_key
        }

        # Make the API call
        response = requests.get(base_url, params=params)

        # Check if the API call was successful
        if response.status_code == 200:
            data = response.json()
            if 'Weekly Adjusted Time Series' in data or 'Monthly Time Series' in data:
                # Extract the time series data from the response
                time_series_data = data['Weekly Adjusted Time Series'] if function == 'TIME_SERIES_WEEKLY_ADJUSTED' else data['Monthly Time Series']
                # Convert the data to a DataFrame
                df = pd.DataFrame.from_dict(time_series_data, orient='index')
                # Convert date strings to datetime objects
                df.index = pd.to_datetime(df.index)
                # Convert data values to numeric types
                df = df.apply(pd.to_numeric)
                # Add a 'ticker' column to identify the data for each ticker
                df['ticker'] = ticker
                # Append the data for the current ticker to the combined DataFrame
                combined_data = combined_data.append(df)
            else:
                print(f"Error: Data not found for {ticker}.")
        else:
            print(f"Error: API request failed for {ticker} with status code {response.status_code}.")

        # Add a short delay after every 4 requests to comply with API rate limits
        if idx % 4 == 0 and idx < len(list_of_tickers):
            print("Waiting for 60 seconds...")
            time.sleep(60)

    return combined_data

def get_monthly_adjusted_time_series(api_key, symbol, datatype='json'):
    """
    Fetch monthly adjusted time series data for a specific equity from Alpha Vantage.

    Parameters:
        api_key (str): API key from Alpha Vantage.
        symbol (str): The name of the equity (e.g., 'IBM', 'AAPL', 'MSFT').
        datatype (str): 'json' (default) or 'csv' to specify the data format.

    Returns:
        pandas.DataFrame: DataFrame containing the monthly adjusted time series data.
    """
    base_url = 'https://www.alphavantage.co/query'
    function = 'TIME_SERIES_MONTHLY_ADJUSTED'

    # Build the API request URL
    params = {
        'function': function,
        'symbol': symbol,
        'datatype': datatype,
        'apikey': api_key
    }

    # Make the API call
    response = requests.get(base_url, params=params)

    # Check if the API call was successful
    if response.status_code == 200:
        data = response.json()
        if 'Monthly Adjusted Time Series' in data:
            # Extract the monthly adjusted time series data from the response
            monthly_adjusted_time_series = data['Monthly Adjusted Time Series']
            # Convert the data to a DataFrame
            df = pd.DataFrame.from_dict(monthly_adjusted_time_series, orient='index')
            # Convert date strings to datetime objects
            df.index = pd.to_datetime(df.index)
            # Convert data values to numeric types
            df = df.apply(pd.to_numeric)
            return df
        else:
            print("Error: Data not found in the API response.")
    else:
        print(f"Error: API request failed with status code {response.status_code}.")

    return None


def get_quote_data(api_key, list_of_tickers, datatype='json'):
    """
    Fetch quote data for a list of tickers from Alpha Vantage.

    Parameters:
        api_key (str): API key from Alpha Vantage.
        list_of_tickers (list): List of ticker symbols.
        datatype (str): 'json' (default) or 'csv' to specify the data format.

    Returns:
        pandas.DataFrame: DataFrame containing the quote data for all tickers.
    """
    base_url = 'https://www.alphavantage.co/query'
    function = 'GLOBAL_QUOTE'

    # Initialize an empty DataFrame to store the combined data
    combined_data = pd.DataFrame()

    for idx, ticker in enumerate(list_of_tickers, 1):
        # Build the API request URL for the current ticker
        params = {
            'function': function,
            'symbol': ticker,
            'datatype': datatype,
            'apikey': api_key
        }

        # Make the API call
        response = requests.get(base_url, params=params)

        # Check if the API call was successful
        if response.status_code == 200:
            data = response.json()
            if 'Global Quote' in data:
                # Extract the quote data from the response
                quote_data = data['Global Quote']
                # Convert the data to a DataFrame
                df = pd.DataFrame([quote_data])
                # Convert data values to numeric types
                df = df.apply(pd.to_numeric)
                # Add a 'ticker' column to identify the data for each ticker
                df['ticker'] = ticker
                # Append the data for the current ticker to the combined DataFrame
                combined_data = combined_data.append(df)
            else:
                print(f"Error: Data not found for {ticker}.")
        else:
            print(f"Error: API request failed for {ticker} with status code {response.status_code}.")

        # Add a short delay after every 5 requests to comply with API rate limits
        if idx % 5 == 0 and idx < len(list_of_tickers):
            print("Waiting for 15 seconds...")
            time.sleep(15)

    return combined_data

def symbol_search(api_key, keywords, datatype='json'):
    """
    Perform symbol search based on keywords using the Alpha Vantage API.

    Parameters:
        api_key (str): API key from Alpha Vantage.
        keywords (str): A text string of keywords for symbol search.
        datatype (str): 'json' (default) or 'csv' to specify the data format.

    Returns:
        pandas.DataFrame: DataFrame containing the search results with symbols and market information.
    """
    base_url = 'https://www.alphavantage.co/query'
    function = 'SYMBOL_SEARCH'

    # Build the API request URL
    params = {
        'function': function,
        'keywords': keywords,
        'datatype': datatype,
        'apikey': api_key
    }

    # Make the API call
    response = requests.get(base_url, params=params)

    # Check if the API call was successful
    if response.status_code == 200:
        data = response.json()
        if 'bestMatches' in data:
            # Extract the search results from the response
            search_results = data['bestMatches']
            # Convert the data to a DataFrame
            df = pd.DataFrame(search_results)
            # Convert data values to appropriate data types
            df[['5. marketCap', '8. currency']] = df[['5. marketCap', '8. currency']].apply(pd.to_numeric)
            return df
        else:
            print("Error: Search results not found in the API response.")
    else:
        print(f"Error: API request failed with status code {response.status_code}.")

    return None

def get_market_status(api_key):
    """
    Fetch the current market status (open vs. closed) from Alpha Vantage.

    Parameters:
        api_key (str): API key from Alpha Vantage.

    Returns:
        pandas.DataFrame: DataFrame containing the current market status.
    """
    base_url = 'https://www.alphavantage.co/query'
    function = 'MARKET_STATUS'

    # Build the API request URL
    params = {
        'function': function,
        'apikey': api_key
    }

    # Make the API call
    response = requests.get(base_url, params=params)

    # Check if the API call was successful
    if response.status_code == 200:
        data = response.json()
        if 'marketStatus' in data:
            # Extract the market status data from the response
            market_status_data = data['marketStatus']
            # Convert the data to a DataFrame
            df = pd.DataFrame([market_status_data])
            return df
        else:
            print("Error: Market status data not found in the API response.")
    else:
        print(f"Error: API request failed with status code {response.status_code}.")

    return None

import requests
import pandas as pd

# Aplha Intelligence APIs

def get_news_sentiment(api_key, tickers=None, topics=None, time_from=None, time_to=None, sort='LATEST', limit=50):
    """
    Fetch live and historical market news & sentiment data from Alpha Vantage.

    Parameters:
        api_key (str): API key from Alpha Vantage.
        tickers (str or list, optional): Stock/crypto/forex symbols of your choice.
        topics (str or list, optional): News topics of your choice.
        time_from (str, optional): Time range start in YYYYMMDDTHHMM format.
        time_to (str, optional): Time range end in YYYYMMDDTHHMM format.
        sort (str, optional): Sort order for the results ('LATEST', 'EARLIEST', or 'RELEVANCE').
        limit (int, optional): Number of results to retrieve (default is 50).

    Returns:
        pandas.DataFrame: DataFrame containing the news & sentiment data.
    """
    base_url = 'https://www.alphavantage.co/query'
    function = 'NEWS_SENTIMENT'

    # Convert tickers and topics to comma-separated strings if they are lists
    tickers = ','.join(tickers) if isinstance(tickers, list) else tickers
    topics = ','.join(topics) if isinstance(topics, list) else topics

    # Build the API request URL
    params = {
        'function': function,
        'apikey': api_key,
        'tickers': tickers,
        'topics': topics,
        'time_from': time_from,
        'time_to': time_to,
        'sort': sort,
        'limit': limit
    }

    # Make the API call
    response = requests.get(base_url, params=params)

    # Check if the API call was successful
    if response.status_code == 200:
        data = response.json()
        if 'articles' in data:
            # Extract the news & sentiment data from the response
            news_sentiment_data = data['articles']
            # Convert the data to a DataFrame
            df = pd.DataFrame(news_sentiment_data)
            return df
        else:
            print("Error: News & sentiment data not found in the API response.")
    else:
        print(f"Error: API request failed with status code {response.status_code}.")

    return None

import requests
import pandas as pd
import time
from datetime import datetime

def get_top_gainers_losers_most_active(api_key):
    """
    Fetch the top 20 gainers, losers, and most actively traded tickers in the US market from Alpha Vantage.

    Parameters:
        api_key (str): API key from Alpha Vantage.

    Returns:
        pandas.DataFrame: DataFrame containing the top gainers, losers, and most active tickers data.
    """
    base_url = 'https://www.alphavantage.co/query'
    function = 'TOP_GAINERS_LOSERS'

    # Build the API request URL
    params = {
        'function': function,
        'apikey': api_key
    }

    # Make the API call
    response = requests.get(base_url, params=params)

    # Check if the API call was successful
    if response.status_code == 200:
        data = response.json()
        if 'mostGainerStock' in data and 'mostLoserStock' in data and 'mostActiveStock' in data:
            # Extract the top gainers, losers, and most active tickers data from the response
            top_gainers_data = data['mostGainerStock']
            top_losers_data = data['mostLoserStock']
            most_active_data = data['mostActiveStock']

            # Convert the data to DataFrames
            df_gainers = pd.DataFrame(top_gainers_data)
            df_losers = pd.DataFrame(top_losers_data)
            df_most_active = pd.DataFrame(most_active_data)

            return df_gainers, df_losers, df_most_active
        else:
            print("Error: Data not found in the API response.")
    else:
        print(f"Error: API request failed with status code {response.status_code}.")

    return None, None, None

def run_top_gainers_losers_most_active(api_key):
    while True:
        # Get the current hour in 24-hour format
        current_hour = datetime.now().hour

        # Check if the current hour is within the trading hours (9 am to 4 pm)
        if 9 <= current_hour < 16:
            # Fetch top gainers, losers, and most active tickers
            gainers, losers, most_active = get_top_gainers_losers_most_active(api_key)

            # Print or process the data as per your requirement
            if gainers is not None and losers is not None and most_active is not None:
                print("Top Gainers:")
                print(gainers)
                print("Top Losers:")
                print(losers)
                print("Most Active:")
                print(most_active)

        # Wait for 1 hour before making the next API call
        time.sleep(3600)

## Fundamental Data APIs


def get_company_overview(api_key, tickers):
    """
    Fetch company information, financial ratios, and key metrics for multiple tickers.

    Parameters:
        api_key (str): API key from Alpha Vantage.
        tickers (list): List of ticker symbols.

    Returns:
        dict: Dictionary with ticker symbols as keys and company overview data as values.
    """
    base_url = 'https://www.alphavantage.co/query'
    function = 'OVERVIEW'
    data_dict = {}

    for ticker in tickers:
        params = {
            'function': function,
            'symbol': ticker,
            'apikey': api_key
        }
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            data_dict[ticker] = data
        else:
            print(f"Error fetching data for {ticker}: status code {response.status_code}")

    return data_dict

def get_income_statement(api_key, tickers):
    """
    Fetch annual and quarterly income statements for multiple tickers.

    Parameters:
        api_key (str): API key from Alpha Vantage.
        tickers (list): List of ticker symbols.

    Returns:
        dict: Dictionary with ticker symbols as keys and income statement data as values.
    """
    base_url = 'https://www.alphavantage.co/query'
    function = 'INCOME_STATEMENT'
    data_dict = {}

    for ticker in tickers:
        params = {
            'function': function,
            'symbol': ticker,
            'apikey': api_key
        }
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            data_dict[ticker] = data
        else:
            print(f"Error fetching data for {ticker}: status code {response.status_code}")

    return data_dict

def get_balance_sheet(api_key, tickers):
    """
    Fetch annual and quarterly balance sheets for multiple tickers.

    Parameters:
        api_key (str): API key from Alpha Vantage.
        tickers (list): List of ticker symbols.

    Returns:
        dict: Dictionary with ticker symbols as keys and balance sheet data as values.
    """
    base_url = 'https://www.alphavantage.co/query'
    function = 'BALANCE_SHEET'
    data_dict = {}

    for ticker in tickers:
        params = {
            'function': function,
            'symbol': ticker,
            'apikey': api_key
        }
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            data_dict[ticker] = data
        else:
            print(f"Error fetching data for {ticker}: status code {response.status_code}")

    return data_dict

def get_earnings_data(symbols, api_key):
    dfs = []
    for symbol in symbols:
        url = f'https://www.alphavantage.co/query?function=EARNINGS&symbol={symbol}&apikey={api_key}'
        response = requests.get(url)
        data = response.json()
        df = pd.DataFrame(data['annualEarnings'] + data['quarterlyEarnings'])
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def get_cash_flow_data(symbols, api_key):
    dfs = []
    for symbol in symbols:
        url = f'https://www.alphavantage.co/query?function=CASH_FLOW&symbol={symbol}&apikey={api_key}'
        response = requests.get(url)
        data = response.json()
        df = pd.DataFrame(data['annualReports'] + data['quarterlyReports'])
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def get_listing_status_data(api_key):
    url = f'https://www.alphavantage.co/query?function=LISTING_STATUS&apikey={api_key}'
    response = requests.get(url)
    data = pd.read_csv(pd.compat.StringIO(response.text))
    return data

def get_earnings_calendar_data(api_key, symbols=None, horizon='3month'):
    if symbols is None:
        url = f'https://www.alphavantage.co/query?function=EARNINGS_CALENDAR&apikey={api_key}'
    else:
        symbols_str = ','.join(symbols)
        url = f'https://www.alphavantage.co/query?function=EARNINGS_CALENDAR&symbol={symbols_str}&apikey={api_key}'
    url += f'&horizon={horizon}'
    
    response = requests.get(url)
    data = pd.read_csv(pd.compat.StringIO(response.text))
    return data

def get_ipo_calendar_data(api_key):
    url = f'https://www.alphavantage.co/query?function=IPO_CALENDAR&apikey={api_key}'
    response = requests.get(url)
    data = pd.read_csv(pd.compat.StringIO(response.text))
    return data

# Forex (FX)

def get_currency_exchange_rate(from_currency, to_currency, api_key):
    url = f'https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={from_currency}&to_currency={to_currency}&apikey={api_key}'
    response = requests.get(url)
    data = response.json()
    
    df = pd.DataFrame(data['Realtime Currency Exchange Rate'], index=[0])
    return df


def get_fx_daily_data(from_symbol, to_symbol, api_key, outputsize='compact', datatype='json'):
    url = f'https://www.alphavantage.co/query?function=FX_DAILY&from_symbol={from_symbol}&to_symbol={to_symbol}&outputsize={outputsize}&datatype={datatype}&apikey={api_key}'
    response = requests.get(url)
    
    if datatype == 'json':
        data = response.json()
        df = pd.DataFrame(data['Time Series FX (Daily)']).T
        df.index = pd.to_datetime(df.index)
    elif datatype == 'csv':
        df = pd.read_csv(pd.compat.StringIO(response.text))
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    else:
        raise ValueError("Invalid datatype. Please choose 'json' or 'csv'.")
    
    return df

def get_fx_weekly_data(from_symbol, to_symbol, api_key, datatype='json'):
    url = f'https://www.alphavantage.co/query?function=FX_WEEKLY&from_symbol={from_symbol}&to_symbol={to_symbol}&datatype={datatype}&apikey={api_key}'
    response = requests.get(url)
    
    if datatype == 'json':
        data = response.json()
        df = pd.DataFrame(data['Time Series FX (Weekly)']).T
        df.index = pd.to_datetime(df.index)
    elif datatype == 'csv':
        df = pd.read_csv(pd.compat.StringIO(response.text))
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    else:
        raise ValueError("Invalid datatype. Please choose 'json' or 'csv'.")
    
    return df

def get_fx_monthly_data(from_symbol, to_symbol, api_key, datatype='json'):
    url = f'https://www.alphavantage.co/query?function=FX_MONTHLY&from_symbol={from_symbol}&to_symbol={to_symbol}&datatype={datatype}&apikey={api_key}'
    response = requests.get(url)
    
    if datatype == 'json':
        data = response.json()
        df = pd.DataFrame(data['Time Series FX (Monthly)']).T
        df.index = pd.to_datetime(df.index)
    elif datatype == 'csv':
        df = pd.read_csv(pd.compat.StringIO(response.text))
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    else:
        raise ValueError("Invalid datatype. Please choose 'json' or 'csv'.")
    
    return df

#crypto
def get_currency_exchange_rates(currency_pairs, api_key):
    base_url = 'https://www.alphavantage.co/query'
    function = 'CURRENCY_EXCHANGE_RATE'
    output_format = 'json'
    
    rates_data = []
    
    for pair in currency_pairs:
        from_currency, to_currency = pair.split('/')
        url = f'{base_url}?function={function}&from_currency={from_currency}&to_currency={to_currency}&apikey={api_key}'
        response = requests.get(url)
        data = response.json()
        
        if 'Realtime Currency Exchange Rate' in data:
            rate = data['Realtime Currency Exchange Rate']['5. Exchange Rate']
            rates_data.append({'From_Currency': from_currency, 'To_Currency': to_currency, 'Exchange_Rate': rate})
    
    if len(rates_data) == 0:
        raise ValueError("No exchange rates found for the provided currency pairs.")
    
    df = pd.DataFrame(rates_data)
    
    return df

# Example usage


# Example usage:
if __name__ == "__main__":
    api_key = "D03VFL5XJ25QWELA"
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'FB']  # Add more tickers as needed

    # Call each function and print the result
    print("1. get_ticker_data:")
    print(get_ticker_data('AAPL', 'daily'))

    print("2. get_ticker_data_graph_item:")
    print(get_ticker_data_graph_item('AAPL'))

    print("3. get_earnings_data:")
    print(get_earnings_data(api_key, tickers))

    print("4. get_daily_time_series:")
    print(get_daily_time_series(api_key, 'AAPL'))

    print("5. get_weekly_time_series:")
    print(get_weekly_time_series(api_key, tickers))

    print("6. get_time_series:")
    print(get_time_series(api_key, 'TIME_SERIES_MONTHLY_ADJUSTED', tickers))

    print("7. get_monthly_adjusted_time_series:")
    print(get_monthly_adjusted_time_series(api_key, 'AAPL'))

    print("8. get_quote_data:")
    print(get_quote_data(api_key, tickers))

    print("9. symbol_search:")
    print(symbol_search(api_key, 'AAPL'))

    print("10. get_market_status:")
    print(get_market_status(api_key))

    print("11. get_news_sentiment:")
    print(get_news_sentiment(api_key, tickers=['AAPL'], topics=['business', 'technology']))

    print("12. get_top_gainers_losers_most_active:")
    gainers, losers, most_active = get_top_gainers_losers_most_active(api_key)
    print("Top Gainers:")
    print(gainers)
    print("Top Losers:")
    print(losers)
    print("Most Active:")
    print(most_active)

    print("13. run_top_gainers_losers_most_active:")
    run_top_gainers_losers_most_active(api_key)

    print("14. get_company_overview:")
    print(get_company_overview(api_key, tickers))

    print("15. get_income_statement:")
    print(get_income_statement(api_key, tickers))

    print("16. get_balance_sheet:")
    print(get_balance_sheet(api_key, tickers))

    print("17. get_earnings_data:")
    print(get_earnings_data(tickers, api_key))

    print("18. get_cash_flow_data:")
    print(get_cash_flow_data(tickers, api_key))

    print("19. get_listing_status_data:")
    print(get_listing_status_data(api_key))

    print("20. get_earnings_calendar_data:")
    print(get_earnings_calendar_data(api_key, symbols=tickers))

    # Fetch real-time currency exchange rate
    from_currency = 'USD'
    to_currency = 'EUR'
    exchange_rate_data = get_currency_exchange_rate(from_currency, to_currency, api_key)
    print(f"Exchange Rate ({from_currency}/{to_currency}):")
    print(exchange_rate_data)

    # Fetch daily foreign exchange data
    from_symbol = 'USD'
    to_symbol = 'EUR'
    daily_fx_data = get_fx_daily_data(from_symbol, to_symbol, api_key, outputsize='compact', datatype='json')
    print(f"Daily FX Data ({from_symbol}/{to_symbol}):")
    print(daily_fx_data)

    # Fetch weekly foreign exchange data
    weekly_fx_data = get_fx_weekly_data(from_symbol, to_symbol, api_key, datatype='json')
    print(f"Weekly FX Data ({from_symbol}/{to_symbol}):")
    print(weekly_fx_data)

    # Fetch monthly foreign exchange data
    monthly_fx_data = get_fx_monthly_data(from_symbol, to_symbol, api_key, datatype='json')
    print(f"Monthly FX Data ({from_symbol}/{to_symbol}):")
    print(monthly_fx_data)

    # Fetch real-time currency exchange rates for multiple currency pairs
    currency_pairs = ['USD/EUR', 'USD/JPY', 'GBP/USD']
    exchange_rates_data = get_currency_exchange_rates(currency_pairs, api_key)
    print("Real-time Exchange Rates:")
    print(exchange_rates_data)