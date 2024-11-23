import streamlit as st
#import datetime as dt
from datetime import datetime, timedelta
import pandas as pd
import pandas_ta as ta
import numpy as np
import time
import yaml
import requests
import plotly.graph_objs as go
import uuid
from zoneinfo import ZoneInfo

# Define local timezone
local_timezone = ZoneInfo("Asia/Bangkok")

# Get current UTC time and convert to local time
# Define 9:00 AM in GMT+7 for the current day
local_timezone = ZoneInfo("Asia/Bangkok")
now = datetime.now(local_timezone)
local_today = datetime.now(local_timezone).date()
#now = dt.now(ZoneInfo("UTC")).astimezone(local_timezone)

# Load YAML configuration
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Set page configuration
st.set_page_config(page_title="Intraday Stock Price Tracker", layout="wide")
#now = dt.datetime.now()

with st.sidebar:
    with st.form("form_key"):
        contract = st.session_state.selected_stock = st.text_input("INPUT STOCK (only stock):", 'HPG')
        interval = st.selectbox("Interval", options=['1D', '60', '30', '15'])
        fromdate = st.date_input("From date:", value=local_today - timedelta(days=365), max_value=local_today)
        todate = st.date_input("To date:", value = local_today, max_value=local_today)
        strategy = st.selectbox("Trading strategy:", options=['neutral', 'long', 'short'])
        bperiod = st.number_input('Buy Pperiod:', placeholder="in range 4 - 45", min_value=4, max_value=45)
        speriod = st.number_input('Sell period:', placeholder="in range 4 - 45", min_value=4, max_value=45)
        bshield = st.selectbox('Buy shield:', options=[0, 10, 20, 50, 100, 150, 200])
        sshield = st.selectbox('Sell shield:', options=[0, 10, 20, 50, 100, 150, 200])
        submit_btn = st.form_submit_button("Submit")