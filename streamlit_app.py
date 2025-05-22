import streamlit as st
from datetime import datetime, timedelta
from datetime import time
from time import sleep
from zoneinfo import ZoneInfo
import pandas_ta as ta
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
#from concurrent.futures import ProcessPoolExecutor
import uuid
import psycopg2
from psycopg2 import pool
#from psycopg2.extras import execute_values

query_data = st.secrets["query"]["data"]
query_ticker = st.secrets["query"]["ticker"]

# Define local timezone. Get current UTC time and convert to local time
local_timezone = ZoneInfo("Asia/Bangkok")
# Define 9:00 AM in GMT+7 for the current day
now = datetime.now(local_timezone)
local_time = now.strftime("%H:%M:%S")
local_today = datetime.now(local_timezone).date()

# Initialize connection pool (do this once at the app start)
if 'db_pool' not in st.session_state:
    st.session_state.db_pool = psycopg2.pool.SimpleConnectionPool(
        1, 20,  # Min 1 and max 20 connections in the pool
        user="postgres",
        password="J,@CC2@oCa{R'^O]",
        host="34.142.206.64",
        port="5432",
        database="postgres"
    )

# Function to get a connection from the pool
def get_connection():
    return st.session_state.db_pool.getconn()

# Function to return a connection to the pool
def return_connection(conn):
    st.session_state.db_pool.putconn(conn)

# Fetch distinct stocks
def get_ticker():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(query_ticker)
    results = cursor.fetchall()
    return [row[0] for row in results]  # Extract single column values

# Function to query the database using connection pooling
def get_stock_data(stock_name, fromtime, totime):
    conn = get_connection()
    cursor = conn.cursor()
    #print(cursor.mogrify(query_thma, (stock_name.strip(), fromdate, todate)).decode())
    cursor.execute(query_data, (stock_name, fromtime, totime))
    stock_data = cursor.fetchall()
    
    # Get column names dynamically from the cursor description
    column_names = [desc[0] for desc in cursor.description]    
    # Convert the stock data to a pandas DataFrame with dynamic column names
    df = pd.DataFrame(stock_data, columns=column_names)

    cursor.close()
    return_connection(conn)  # Return the connection back to the pool
    
    return df

def group_backward(df, interval):
    data = df.copy()    
    if interval == 1:
        return data[['unixtime', 'date', 'time', 'priceaverage', 'priceclose', 'priceopen', 'pricehigh', 'pricelow', 'dealvolume']]        
    else:
        # Calculate a group index for each row based on reverse order
        data.loc[:, 'group'] = (data.index // interval)[::-1]        
        # Group by the 'group' column and calculate the weighted average price ('priceaverage')
        df = data.set_index('group').groupby('group').apply(
            lambda g: pd.Series({
                't': g['unixtime'].iloc[-1],       # Take the most recent unix_time in each group
                'date': g['date'].iloc[-1],            # Take the most recent date
                'time': g['time'].iloc[-1],            # Take the most recent time
                #'priceaverage': (g['priceaverage'] * g['dealvolume']).sum() / g['dealvolume'].sum(),  # Weighted average price
                'priceclose': g['priceclose'].iloc[-1],
                'priceopen': g['priceopen'].iloc[0],         # First open price in the group
                'pricehigh': g['pricehigh'].max(),           # Highest price in the group
                'pricelow': g['pricelow'].min(),           # Lowest price in the group
                'dealvolume': g['dealvolume'].sum()
            })
        ).reset_index(drop=True)
        # Reverse the order of rows in the DataFrame
        df = df.iloc[::-1].reset_index(drop=True)
        result_df = pd.DataFrame(df)
        
        return result_df

def render_chart(data):
    fig = go.Figure()
    #color_dict = {8: 'blue', 15: 'green', 20: 'orange',30: 'indigo'}
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['priceclose'],
        mode='lines',
        line=dict(color='green', width=2),
        customdata=data['date'].astype(str)+': ' + data['time'].astype(str),
        name=f"{data['priceclose'].iloc[-1]}",
        hovertemplate='close: %{y}<br>%{customdata}<extra></extra>', #<br>
        yaxis='y2'  # Map to second y-axis
    ))

    # Add price line on second y-axis
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['priceaverage'],
        mode='lines',
        line=dict(color='yellow', width=2),
        customdata=data['date'].astype(str)+' '+data['time'].astype(str),
        name=f"{data['priceaverage'].iloc[-1]}",
        hovertemplate='avg: %{y}<extra></extra>',
        yaxis='y2'  # Map to second y-axis
    ))
    
    # Add RSI line trace to the left y-axis
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['rsi']*24800/100, # 43225/100,
        mode='lines',
        line=dict(color='orangered', width=2),
        customdata=f'_{interval}'+ ': ' + data['rsi'].fillna(0).astype(int).astype(str),
        name='rsi',
        hovertemplate= 'rsi%{customdata}%<extra></extra>',
        yaxis='y'  # Left y-axis
    ))

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data[f'l_{interval}'],
        mode='lines',
        line=dict(color='white'),  # Use variable interval
        fillcolor='rgba(0, 0, 0, 0)',  # Transparent fill
        name=f'L_{interval}',
        stackgroup='chance',  # Dynamic stack group based on interval
        groupnorm=''  # Disable normalization
    ))

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data[f'n_{interval}'],
        mode='lines',
        line=dict(width=1.0, color='blue'),  # Use variable interval
        name=f'N_{interval}',
        stackgroup='chance',  # Dynamic stack group based on interval
    ))
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data[f's_{interval}'],
        mode='lines',
        line=dict(color='white'),  # Use variable interval
        name=f'S_{interval}',
        stackgroup='chance',  # Dynamic stack group based on interval
    ))

    # Customize layout
    fig.update_xaxes(visible=False, showticklabels=False)
    fig.update_layout(
        #title="Track history index",
        xaxis_title=None,  # Common x-axis
        yaxis=dict(
            title=None,  # Primary y-axis for signals
            side="left"
        ),
        yaxis2=dict(
            title=None,  # Secondary y-axis for price
            overlaying="y",  # Overlay on the same plot
            side="right",  # Display on the right
            showgrid=False  # Optional: Hide gridlines for secondary y-axis
        ),
        legend=dict(title=None),
        template="plotly_white",
        showlegend=False,
        margin=dict(l=0, r=0, t=20, b=0),  # Set margins for wide mode
        height=540,
        hovermode="x unified"  # Unified hover mode for better readability)
    )

    st.plotly_chart(fig, use_container_width=True, key = uuid.uuid4())

def render_hollow(df):    
    df['pC'] = df['priceclose'].shift(1)
    # Define color based on close and previous close
    df['color'] = np.where(df['priceclose'] > df['pC'], "lightgreen", "orangered")
    # Set fill to transparent if close > open and the previously defined color otherwise
    df['fill'] = np.where(df['priceclose'] > df['priceopen'], "rgba(128,128,128,0.5)", df['color'])

    # Initialize empty plot with marginal subplots
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes="columns",
        shared_yaxes="rows",
        #column_width=[0.8, 0.2],
        row_heights=[0.8, 0.2],
        horizontal_spacing=0,
        vertical_spacing=0,
    )
   
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['priceopen'],
            high=df['pricehigh'],
            low=df['pricelow'],
            close=df['priceclose'],
            showlegend=False,
            name = None
            ),
        row=1,
        col=1
    )
    fig.update_traces(name='', selector=dict(type='candlestick'))
    showlegend = False

    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['dealvolume'],
            showlegend=False,
            marker_line_color=df['color'],
            marker_color=df['fill'],
            name=f"{df['dealvolume'].iloc[-1]}",
            hovertemplate='%{y}<extra></extra>',
        ),
        col=1,
        row=2,
    )

    fig.update_xaxes(visible=False, showticklabels=False)
    fig.update_layout(
        xaxis_title=None,  # Common x-axis
        yaxis=dict(
            title=None,  # Primary y-axis for signals
            side="right"
        ),
        showlegend=False,
        margin=dict(l=0, r=0, t=20, b=0),  # Set margins for wide mode
        xaxis_rangeslider_visible=False,
        height=540,
        hovermode="x unified"  # Unified hover mode for better readability)
    )
    st.plotly_chart(fig, use_container_width=True, key = uuid.uuid4())

# Set page configuration
st.set_page_config(page_title="Bros-Fu Price Tracker", layout="wide")

# Load options
tickers = get_ticker()

with st.sidebar:
    with st.form("form_key"):
        symbol = st.selectbox("symbol", options=tickers, index=tickers.index('VN30') if 'VN30' in tickers else 0)
        fromdate = st.date_input("From date:", value=local_today - timedelta(days=365), max_value=local_today)
        todate = st.date_input("To date:", value = local_today, max_value=local_today)
        length = st.selectbox('Group of ticks:', options=[5, 1, 2, 3, 8, 13])
        interval = st.selectbox('ENGINE/ RSI interval:', options=[5, 8, 13])
        submit_btn = st.form_submit_button("Submit")

entrydate = fromdate - timedelta(weeks=1)

# Placeholder for the chart
stick_placeholder = st.empty()
vol_placeholder = st.empty()
chart_placeholder = st.empty()
error_message = st.empty()

if "df" not in st.session_state:
    st.session_state['df'] = None
    st.session_state['selected_stock'] = None

try:
    if fromdate <= todate:
        while (todate == local_today and ((now.replace(hour=8, minute=45, second=00, microsecond=000) <= now <= now.replace(hour=11, minute=31, second=59, microsecond=999)) 
                                            or (now.replace(hour=13, minute=00, second=00, microsecond=000) <= now <= now.replace(hour=14, minute=45, second=59, microsecond=999))
                                            or (now.replace(hour=20, minute=29, second=59, microsecond=999) <= now <= now.replace(hour=23, minute=59, second=59, microsecond=999)))):
            now = datetime.now(local_timezone)
            totime = now.time()
            df = get_stock_data(symbol, fromdate, todate)
            print(df)

            st.session_state['df'] = pd.DataFrame(df.loc[df['date'] >= fromdate, :])
            st.session_state['df'] = st.session_state['df'].reset_index(drop=True)
            st.session_state['df']['rsi'] = ta.rsi(st.session_state['df']['priceaverage'], length=interval, mamode='ema')

            with stick_placeholder:
                st.session_state['stick'] = group_backward(st.session_state['df'], length)
                render_hollow(st.session_state['stick'])
            
            with chart_placeholder:
                render_chart(st.session_state['df'])

            print('sleeping 20')
            sleep(20)  # Wait for 60 seconds before the next update

        if (st.session_state['df'] is None 
            or st.session_state['selected_stock'] != symbol 
            or st.session_state['fromdate'] != fromdate or st.session_state['todate'] != todate):
            
            st.session_state['selected_stock'] = symbol
            st.session_state['fromdate'] = fromdate
            st.session_state['todate'] = todate
        
        df = get_stock_data(symbol, fromdate, todate)
        
        print('not while')
        st.session_state['df'] = pd.DataFrame(df.loc[df['date'] >= st.session_state['fromdate'], :])
        st.session_state['df'] = st.session_state['df'].reset_index(drop=True)

        st.session_state['df'].loc[:, 'rsi'] = ta.rsi(st.session_state['df'].loc[:, 'priceaverage'], length=interval, mamode='ema')
        
        st.session_state['stick'] = group_backward(st.session_state['df'], length)

        render_hollow(st.session_state['stick'])
        
        render_chart(st.session_state['df'])

    else:
        st.warning('From date must be before end date!')

except TypeError:
    error_message.error("No Data to show, please check your input settings")