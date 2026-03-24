import streamlit as st
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas_ta as ta
import pandas as pd
import plotly.graph_objects as go
#from plotly.subplots import make_subplots
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

# Fetch distinct stocks - CACHED to avoid repeated DB queries
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_ticker():
    conn = st.session_state.db_pool.getconn()
    cursor = conn.cursor()
    cursor.execute(query_ticker)
    results = cursor.fetchall()
    cursor.close()
    st.session_state.db_pool.putconn(conn)
    return [row[0] for row in results]  # Extract single column values

# Function to query the database using connection pooling - CACHED
#@st.cache_data(ttl=1800)  # Cache for 30 minutes
# Option query is stored in Streamlit secrets so it can be changed without code updates
query_options = st.secrets.query.options

@st.cache_data(ttl=300)
def get_option_table(underlying_symbol):
    """Fetch options/derivatives table for a given underlying symbol."""
    conn = st.session_state.db_pool.getconn()
    cursor = conn.cursor()
    try:
        cursor.execute(query_options, (underlying_symbol,))
        result = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]
    finally:
        cursor.close()
        st.session_state.db_pool.putconn(conn)

    df = pd.DataFrame(result, columns=column_names)
    return df


def get_stock_data(stock_name, fromtime, totime):
    """Fetch stock data from database with caching"""
    conn = st.session_state.db_pool.getconn()
    cursor = conn.cursor()
    try:
        cursor.execute(query_data, (stock_name, fromtime, totime))
        stock_data = cursor.fetchall()
        
        # Get column names dynamically from the cursor description
        column_names = [desc[0] for desc in cursor.description]    
    finally:
        cursor.close()
        st.session_state.db_pool.putconn(conn)
    
    # Convert the stock data to a pandas DataFrame with dynamic column names
    df = pd.DataFrame(stock_data, columns=column_names)
    return df

def group_backward(df, interval):
    """Group data backwards with optimized vectorized operations"""
    data = df.copy()    
    if interval == 1:
        return data[['unixtime', 'date', 'time', 'priceaverage', 'priceclose', 'priceopen', 'pricehigh', 'pricelow', 'dealvolume']]        
    else:
        # Calculate group index in reverse order more efficiently
        n = len(data)
        group_indices = np.arange(n) // interval
        group_indices = (n - 1 - np.arange(n)) // interval
        
        # Use groupby instead of apply for better performance
        grouped = data.assign(group=group_indices).groupby('group', sort=False)
        
        result_data = {
            'unixtime': grouped['unixtime'].last().values,
            'date': grouped['date'].last().values,
            'time': grouped['time'].last().values,
            'priceclose': grouped['priceclose'].last().values,
            'priceopen': grouped['priceopen'].first().values,
            'pricehigh': grouped['pricehigh'].max().values,
            'pricelow': grouped['pricelow'].min().values,
            'dealvolume': grouped['dealvolume'].sum().values,
        }
        
        result_df = pd.DataFrame(result_data)
        # Reverse to match original logic
        result_df = result_df.iloc[::-1].reset_index(drop=True)
        
        return result_df

def render_chart(data):
    fig = go.Figure()
    
    # Pre-compute hover text once for better performance
    hover_date_time = data['date'].astype(str) + ': ' + data['time'].astype(str)
#    rsi_customdata = ': ' + (data[f'rsi_{interval}']/100).astype(str)
    p_rsi_customdata = ': ' + (data[f'p_rsi_{interval}']/100).astype(str)
#    p_l_customdata = ': ' + (data[f'p_l_{interval}']/100).astype(str)
#    p_n_customdata = ': ' + (data[f'p_n_{interval}']/100).astype(str)
#    p_s_customdata = ': ' + (data[f'p_s_{interval}']/100).astype(str)
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['priceclose'],
        mode='lines',
        line=dict(color='yellow', width=2),
        customdata=hover_date_time,
        name=f"{data['priceclose'].iloc[-1]}",
        hovertemplate='close: %{y}<br>%{customdata}<extra></extra>',
        yaxis='y2'  # Map to second y-axis
    ))
    
    # Add RSI line trace to the left y-axis
#    fig.add_trace(go.Scatter(
#        x=data.index,
#        y=data[f'rsi_{interval}'],
#        mode='lines',
#        line=dict(color='orangered', width=2),
#        customdata=rsi_customdata,
#        name='rsi',
#        hovertemplate= 'rsi%{customdata}%<extra></extra>',
#        yaxis='y'  # Left y-axis
#    ))

    # Add p_RSI line trace to the left y-axis
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data[f'p_rsi_{interval}'],
        mode='lines',
        line=dict(color='blue', width=2),
        customdata=p_rsi_customdata,
        name='p_rsi',
        hovertemplate= 'p_rsi%{customdata}%<extra></extra>',
        yaxis='y'  # Left y-axis
    ))
    
        # Add p_l line trace to the left y-axis
#    fig.add_trace(go.Scatter(
#        x=data.index,
#        y=data[f'p_l_{interval}'],
#        mode='lines',
#        line=dict(color='green', width=2),
#        customdata=p_l_customdata,
#        name='p_l',
#        hovertemplate= 'p_l%{customdata}%<extra></extra>',
#        yaxis='y'  # Left y-axis
#    ))
#    
#    # Add p_n line trace to the left y-axis
#    fig.add_trace(go.Scatter(
#        x=data.index,
#        y=data[f'p_n_{interval}'],
#        mode='lines',
#        line=dict(color='yellow', width=2),
#        customdata=p_n_customdata,
#        name='p_n',
#        hovertemplate= 'p_n%{customdata}%<extra></extra>',
#        yaxis='y'  # Left y-axis
#    ))
#    
#    # Add p_s line trace to the left y-axis
#    fig.add_trace(go.Scatter(
#       x=data.index,
#       y=data[f'p_s_{interval}'],
#        mode='lines',
#        line=dict(color='red', width=2),
#        customdata=p_s_customdata,
#        name='p_s',
#        hovertemplate= 'p_s%{customdata}%<extra></extra>',
#        yaxis='y'  # Left y-axis
#    ))
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data[f'l_{interval}'],
        mode='lines',
        line=dict(color='white'),  # Use variable interval
        fillcolor='rgba(0, 0, 0, 0)',  # Transparent fill
#        name=f'L_{interval}',
        name='L',
        stackgroup='chance',  # Dynamic stack group based on interval
        groupnorm=''  # Disable normalization
    ))

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data[f'n_{interval}'],
        mode='lines',
        line=dict(width=1.0, color='grey'),  # Use variable interval
#        name=f'N_{interval}',
        name='N',
        stackgroup='chance',  # Dynamic stack group based on interval
    ))
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data[f's_{interval}'],
        mode='lines',
        line=dict(color='white'),  # Use variable interval
#        name=f'S_{interval}',
        name='S',
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
        hoverlabel=dict(bgcolor="white", font_size=16),
        hovermode="x unified"  # Unified hover mode for better readability
    )

    st.plotly_chart(fig, use_container_width=True, key = uuid.uuid4())

def render_hollow(df):    
    df['pC'] = df['priceclose'].shift(1)
    # Define color based on close and previous close
    df['color'] = np.where(df['priceclose'] > df['pC'], "lightgreen", "orangered")
    # Set fill to transparent if close > open and the previously defined color otherwise
    df['fill'] = np.where(df['priceclose'] > df['priceopen'], "rgba(128,128,128,0.5)", df['color'])

    fig = go.Figure()
    
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['priceopen'],
            high=df['pricehigh'],
            low=df['pricelow'],
            close=df['priceclose'],
            showlegend=False,
            name = None,
            yaxis='y' 
            )#,
        #row=1,
        #col=1
    )
    fig.update_traces(name='', selector=dict(type='candlestick'))
    showlegend = False
        # Add RSI line trace to the left y-axis
    
#    fig.add_trace(go.Scatter(
#        x=df.index,
#        y=df['dealvolume'],
#        mode='lines',
#        line=dict(color='blue', width=2),
        
#        name=f"{df['dealvolume'].iloc[-1]}",
#        hovertemplate='vol: %{y}<extra></extra>',
#        yaxis='y2'  # Map to second y-axis
#    ))

    fig.update_xaxes(visible=False, showticklabels=False)
    fig.update_yaxes(tickfont_family="Arial Black")
    fig.update_layout(
        xaxis_title=None,  # Common x-axis
        yaxis=dict(
            title=None,  # Primary y-axis for signals
            side="right"
        ),
#        yaxis2=dict(
#            title=None,  # Secondary y-axis for price
#            overlaying="y",  # Overlay on the same plot
            #tickformat=".2f",
#            side="left",  # Display on the right
#            showgrid=False,
#        ),
        showlegend=False,
        margin=dict(l=0, r=0, t=20, b=0),  # Set margins for wide mode
        xaxis_rangeslider_visible=False,
        height=360,
        hoverlabel=dict(bgcolor="white", font_size=16),
        hovermode="x unified"  # Unified hover mode for better readability)
    )
    st.plotly_chart(fig, use_container_width=True, key = uuid.uuid4())

#def render_volume(df):

#    df['pC'] = df['priceclose'].shift(1)
    # Define color based on close and previous close
#    df['color'] = np.where(df['priceclose'] > df['pC'], "lightgreen", "orangered")
    # Set fill to transparent if close > open and the previously defined color otherwise
#    df['fill'] = np.where(df['priceclose'] > df['priceopen'], "rgba(128,128,128,0.5)", df['color'])
#    fig = go.Figure()
    
#    fig.add_trace(go.Bar(
#        x=df.index,
#        y=df['dealvolume'],
#        showlegend=False,
#        marker_line_color=df['color'],
#        marker_color=df['fill'],
#        name=f"{df['dealvolume'].iloc[-1]}",
#        hovertemplate='vol: %{y}<extra></extra>',
#        yaxis='y' 
#    ))
    # Add RSI line trace to the left y-axis
#    fig.add_trace(go.Scatter(
#        x=df.index,
#        y=df['atr'], # 43225/100,
#        mode='lines',
#        line=dict(color='blue', width=2),
#        customdata=f'_{interval}'+ ': ' + df['atr'].fillna(0).astype(int).astype(str),
#        name='atr',
#        hovertemplate= 'atr%{customdata}<extra></extra>',
#        yaxis='y2'  # Left y-axis
#    ))
    
    # Customize layout
#    fig.update_xaxes(visible=False, showticklabels=False)
#    fig.update_layout(
        #title="Track history index",
#        xaxis_title=None,  # Common x-axis
#        yaxis=dict(
#            title=None,  # Primary y-axis for signals
#            side="left"
#        ),
#        yaxis2=dict(
#            title=None,  # Secondary y-axis for price
#            overlaying="y",  # Overlay on the same plot
#            tickformat=".2f",
#            side="right",  # Display on the right
#            showgrid=False,
#        ),
#        legend=dict(title=None),
#        template="plotly_white",
#        showlegend=False,
#        margin=dict(l=0, r=0, t=20, b=0),  # Set margins for wide mode
#        height=240,
#        hoverlabel=dict(bgcolor="white", font_size=16),
#        hovermode="x unified"  # Unified hover mode for better readability)
#    )

#    st.plotly_chart(fig, use_container_width=True, key = uuid.uuid4())

# Set page configuration
st.set_page_config(page_title="Bros-Chess Tracker", layout="wide")

# Load options
tickers = get_ticker()

with st.sidebar:
    with st.form("form_key"):
        symbol = st.selectbox("symbol", options=tickers, index=tickers.index('ACB') if 'ACB' in tickers else 0)
        st.divider()
        fromdate = st.date_input("From date:", value=local_today - timedelta(days=60), max_value=local_today)
        todate = st.date_input("To date:", value = local_today, max_value=local_today)
        st.divider()
        interval = st.selectbox('BLOCK engine/rsi/tick length:', options=[233, 34, 55,89, 144, 199, 377], index=0)
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
        #st_autorefresh(interval=1 * 60 * 1000, key="dataframerefresh")
        
        if (st.session_state['df'] is None 
            or st.session_state['selected_stock'] != symbol 
            or st.session_state['fromdate'] != fromdate or st.session_state['todate'] != todate):
            
            st.session_state['selected_stock'] = symbol
            st.session_state['fromdate'] = fromdate
            st.session_state['todate'] = todate
        
        df = get_stock_data(symbol, fromdate, todate)
        df = df.astype({col: 'float64' for col in ['pricehigh', 'pricelow', 'priceclose']})
        st.session_state['df'] = pd.DataFrame(df.loc[df['date'] >= st.session_state['fromdate'], :])
        st.session_state['df'] = st.session_state['df'].reset_index(drop=True)
        #st.session_state['df'].loc[:, 'rsi'] = round(ta.rsi(st.session_state['df'].loc[:, 'priceclose'], length=interval, mamode='ema')*100)
        #st.session_state['df'].loc[:, 'atr'] = ta.atr(st.session_state['df'].loc[:, 'pricehigh'], st.session_state['df'].loc[:, 'pricelow'], st.session_state['df'].loc[:, 'priceclose'], length=interval, mamode='ema')
        st.session_state['stick'] = group_backward(st.session_state['df'], interval)
        #st.session_state['stick'].loc[:, 'atr'] = ta.atr(st.session_state['stick'].loc[:, 'pricehigh'], st.session_state['stick'].loc[:, 'pricelow'], st.session_state['stick'].loc[:, 'priceclose'], length=interval, mamode='ema')
        #st.dataframe((st.session_state['df']))
        render_chart(st.session_state['df'])

        # Visual separator between chart and table
        st.divider()

        # Display the related CWVNLAST option/derivative table for the selected underlying
        options_df = get_option_table(symbol)
        # Shorten column names for display
        options_df = options_df.rename(columns={
            'underlying_symbol': 'Symbol',
            'issuer_name': 'Issuer',
            'stock_symbol': 'CW',
            'exercise_price': 'Execute',
            'exercise_ratio': 'Ratio',
            'days_left': 'D_left',
            'last_trading_date': 'Last_date',
            'ref_price': 'Reference',
            'celing': 'Ceiling',
            'floor': 'Floor',
            'matched_price': 'Last_price',
            'matched_volume': 'Last_vol',
            'price_change': 'Change',
            'price_change_percent': 'Change_p',
        })

        # Keep numeric columns numeric but format them for display via pandas Styler.
        # This avoids Streamlit formatting errors and still allows right-alignment.
        if options_df.empty:
            st.info(f"No derivatives found for '{symbol}'")
        else:
            # Ensure numeric columns are numeric for formatting
            # Ensure numeric columns are numeric for proper formatting
            for col in [
                'Execute',
                'Reference',
                'Ceiling',
                'Floor',
                'Last_price',
                'Last_vol',
                'Change',
                'Change_p',
            ]:
                if col in options_df.columns:
                    options_df[col] = pd.to_numeric(options_df[col], errors='coerce')

            # Ensure Last_date is datetime for formatting
            if 'Last_date' in options_df.columns:
                options_df['Last_date'] = pd.to_datetime(options_df['Last_date'], errors='coerce')

            format_map = {
                'Execute': '{:,.0f}',
                'Reference': '{:,.0f}',
                'Ceiling': '{:,.0f}',
                'Floor': '{:,.0f}',
                'Last_price': '{:,.0f}',
                'Last_vol': '{:,.0f}',
                'Change': '{:,.0f}',
                'Change_p': '{:,.2f}%',
                'Last_date': '{:%d-%m-%Y}',
            }

            right_align_cols = [
                'Execute',
                'Ratio',
                'Reference',
                'Ceiling',
                'Floor',
                'Last_price',
                'Last_vol',
                'Change',
                'Change_p',
            ]
            cols_to_align = [c for c in right_align_cols if c in options_df.columns]

            styled = options_df.style.format({k: v for k, v in format_map.items() if k in options_df.columns})

            # Color entire row based on the Change column value
            if 'Change' in options_df.columns:
                def _row_color(r):
                    c = r.get('Change')
                    if pd.isna(c):
                        return ['' for _ in r]
                    if c < 0:
                        color = 'red'
                    elif c > 0:
                        color = 'green'
                    else:
                        color = 'yellow'
                    return [f'color: {color}'] * len(r)

                styled = styled.apply(_row_color, axis=1)

            if cols_to_align:
                styled = styled.set_properties(subset=cols_to_align, **{'text-align': 'right'})

            st.write(styled)
#        render_volume(st.session_state['df'])
#        render_hollow(st.session_state['stick'])        

    else:
        st.warning('From date must be before end date!')

except TypeError:
    error_message.error("No Data to show, please check your input settings")