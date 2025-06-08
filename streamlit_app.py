import streamlit as st
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas_ta as ta
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import uuid
import psycopg2
from psycopg2 import pool

pd.set_option('future.no_silent_downcasting', True)

query_data = st.secrets.query.data
query_ticker = st.secrets.query.ticker

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
        user=st.secrets.database.user,
        password=st.secrets.database.password,
        host=st.secrets.database.host,
        port=st.secrets.database.port,
        database=st.secrets.database.dbname
    )

# Fetch distinct stocks
def get_ticker():
    conn = st.session_state.db_pool.getconn()
    cursor = conn.cursor()
    cursor.execute(query_ticker)
    results = cursor.fetchall()
    cursor.close()
    st.session_state.db_pool.putconn(conn)
    return [row[0] for row in results]  # Extract single column values

# Function to query the database using connection pooling
def get_stock_data(stock_name, fromtime, totime):
    conn = st.session_state.db_pool.getconn()
    cursor = conn.cursor()
    #print(cursor.mogrify(query_thma, (stock_name.strip(), fromdate, todate)).decode())
    cursor.execute(query_data, (stock_name, fromtime, totime))
    stock_data = cursor.fetchall()
    
    # Get column names dynamically from the cursor description
    column_names = [desc[0] for desc in cursor.description]    
    
    cursor.close()
    st.session_state.db_pool.putconn(conn)
    # Return the connection back to the pool
    
    # Convert the stock data to a pandas DataFrame with dynamic column names
    df = pd.DataFrame(stock_data, columns=column_names)
    
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
        y=data['rsi']*29920/100, # 43225/100,
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
            tickfont=dict(family="Arial Black"),
            overlaying="y",  # Overlay on the same plot
            side="right",  # Display on the right
            showgrid=False  # Optional: Hide gridlines for secondary y-axis
        ),
        legend=dict(title=None),
        template="plotly_white",
        showlegend=False,
        margin=dict(l=0, r=0, t=20, b=0),  # Set margins for wide mode
        height=480,
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
    #fig = make_subplots(
    #    rows=2,
    #    cols=1,
    #    shared_xaxes="columns",
    #    shared_yaxes="rows",
    #    #column_width=[0.8, 0.2],
    #    row_heights=[0.8, 0.2],
    #    horizontal_spacing=0,
    #    vertical_spacing=0,
    #)
    fig = go.Figure()
    
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['priceopen'],
            high=df['pricehigh'],
            low=df['pricelow'],
            close=df['priceclose'],
            showlegend=False,
            name = None
            )#,
        #row=1,
        #col=1
    )
    fig.update_traces(name='', selector=dict(type='candlestick'))
    showlegend = False

    #fig.add_trace(
    #    go.Bar(
    #        x=df.index,
    #        y=df['dealvolume'],
    #        showlegend=False,
    #        marker_line_color=df['color'],
    #        marker_color=df['fill'],
    #        name=f"{df['dealvolume'].iloc[-1]}",
    #        hovertemplate='%{y}<extra></extra>',
    #   ),
    #    col=1,
    #    row=2,
    #)

    fig.update_xaxes(visible=False, showticklabels=False)
    fig.update_yaxes(tickfont_family="Arial Black")
    fig.update_layout(
        xaxis_title=None,  # Common x-axis
        yaxis=dict(
            title=None,  # Primary y-axis for signals
            side="right"
        ),
        showlegend=False,
        margin=dict(l=0, r=0, t=20, b=0),  # Set margins for wide mode
        xaxis_rangeslider_visible=False,
        height=360,
        hovermode="x unified"  # Unified hover mode for better readability)
    )
    st.plotly_chart(fig, use_container_width=True, key = uuid.uuid4())

def render_volume(df):

    df['pC'] = df['priceclose'].shift(1)
    # Define color based on close and previous close
    df['color'] = np.where(df['priceclose'] > df['pC'], "lightgreen", "orangered")
    # Set fill to transparent if close > open and the previously defined color otherwise
    df['fill'] = np.where(df['priceclose'] > df['priceopen'], "rgba(128,128,128,0.5)", df['color'])
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['dealvolume'],
        showlegend=False,
        marker_line_color=df['color'],
        marker_color=df['fill'],
        name=f'{df['dealvolume'].iloc[-1]}',
        hovertemplate='vol: %{y}<extra></extra>',
        yaxis='y' 
    ))
    # Add RSI line trace to the left y-axis
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['atr'], # 43225/100,
        mode='lines',
        line=dict(color='blue', width=2),
        customdata=f'_{interval}'+ ': ' + df['atr'].fillna(0).astype(int).astype(str),
        name='atr',
        hovertemplate= 'atr%{customdata}<extra></extra>',
        yaxis='y2'  # Left y-axis
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
            tickformat=".2f",
            side="right",  # Display on the right
            showgrid=False,
        ),
        legend=dict(title=None),
        template="plotly_white",
        showlegend=False,
        margin=dict(l=0, r=0, t=20, b=0),  # Set margins for wide mode
        height=240,
        hovermode="x unified"  # Unified hover mode for better readability)
    )

    st.plotly_chart(fig, use_container_width=True, key = uuid.uuid4())

# Set page configuration
st.set_page_config(page_title="Bros-Chess Tracker", layout="wide")

# Load options
tickers = get_ticker()

with st.sidebar:
    with st.form("form_key"):
        symbol = st.selectbox("symbol", options=tickers, index=tickers.index('VN30') if 'VN30' in tickers else 0)
        st.divider()
        fromdate = st.date_input("From date:", value=local_today - timedelta(weeks=26), max_value=local_today)
        todate = st.date_input("To date:", value = local_today, max_value=local_today)
        st.divider()
        interval = st.selectbox('BLOCK engine/rsi/tick length:', options=[5, 8, 13])
        #length = st.selectbox('Tick interval (days):', options=[5, 1, 2, 3, 8, 13])
        st.divider()
        submit_btn = st.form_submit_button("Submit")

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
        
        # update every 5 mins
        st_autorefresh(interval=5 * 60 * 1000, key="dataframerefresh")

        if (st.session_state['df'] is None 
            or st.session_state['selected_stock'] != symbol 
            or st.session_state['fromdate'] != fromdate or st.session_state['todate'] != todate):
            
            st.session_state['selected_stock'] = symbol
            st.session_state['fromdate'] = fromdate
            st.session_state['todate'] = todate
        
        df = get_stock_data(symbol, fromdate, todate)
        
        st.session_state['df'] = pd.DataFrame(df.loc[df['date'] >= st.session_state['fromdate'], :])
        st.session_state['df'] = st.session_state['df'].reset_index(drop=True)
        st.session_state['df'].loc[:, 'rsi'] = ta.rsi(st.session_state['df'].loc[:, 'priceaverage'], length=interval, mamode='ema')        
        st.session_state['df'].loc[:, 'atr'] = ta.atr(st.session_state['df'].loc[:, 'pricehigh'], st.session_state['df'].loc[:, 'pricelow'], st.session_state['df'].loc[:, 'priceclose'], length=interval)
        st.session_state['stick'] = group_backward(st.session_state['df'], interval)

        render_chart(st.session_state['df'])
        render_volume(st.session_state['df'])
        render_hollow(st.session_state['stick'])        

    else:
        st.warning('From date must be before end date!')

except TypeError:
    error_message.error("No Data to show, please check your input settings")