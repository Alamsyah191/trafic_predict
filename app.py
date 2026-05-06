import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
from sqlalchemy import create_engine

st.set_page_config(page_title="Smart Traffic AI", layout="wide", page_icon="🚦")

st.title("🚦 Smart Traffic Prediction & AI Assistant")
st.write("This application predicts traffic density and provides an AI Assistant for analysis.")

DB_URL = 'postgresql://postgres:Alamsyah_01@localhost:5432/traffic_db'
engine = create_engine(DB_URL)

@st.cache_data(ttl=600)
def load_data_from_db():
    try:
        query = "SELECT * FROM traffic_table"
        df = pd.read_sql(query, engine)
        
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df['Date'] = df['DateTime'].dt.date
        df['Hour'] = df['DateTime'].dt.hour
        df['DayOfWeek'] = df['DateTime'].dt.day_name()
        
        cats = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        df['DayOfWeek'] = pd.Categorical(df['DayOfWeek'], categories=cats, ordered=True)
        return df
    except Exception as e:
        st.error(f"Failed to connect to Database: {e}")
        return None

df = load_data_from_db()

tab1, tab2, tab3 = st.tabs(["📊 Analysis Dashboard", "🔮 Traffic Prediction", "🤖 AI Chat (Data Connected)"])

with tab1:
    if df is not None:
        st.header("Comprehensive Vehicle Volume Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Data Records", f"{len(df):,}")
        col2.metric("Avg Vehicles/Hour", f"{int(df['Vehicles'].mean())}")
        col3.metric("Max Record", f"{df['Vehicles'].max()}")
        col4.metric("Junctions", f"{df['Junction'].nunique()}")
        
        st.markdown("---")
        
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.subheader("📈 Daily Density Trend (Time Series)")
            daily_trend = df.groupby('Date')['Vehicles'].mean()
            st.line_chart(daily_trend)
            
        with col_chart2:
            st.subheader("⏰ Average Density per Hour (Peak Hours)")
            hourly_trend = df.groupby('Hour')['Vehicles'].mean()
            st.area_chart(hourly_trend)

        st.markdown("---")

        col_chart3, col_chart4 = st.columns(2)
        with col_chart3:
            st.subheader("📅 Density by Day")
            day_trend = df.groupby('DayOfWeek')['Vehicles'].mean()
            st.bar_chart(day_trend)
            
        with col_chart4:
            st.subheader("📍 Junction Comparison")
            junction_trend = df.groupby('Junction')['Vehicles'].sum()
            st.bar_chart(junction_trend)

        with st.expander("View Raw Data from PostgreSQL"):
            st.dataframe(df.head(100), use_container_width=True)

with tab2:
    st.header("Traffic Condition Prediction")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Input Parameters")
        junction = st.selectbox("Select Junction", [1, 2, 3, 4])
        hour = st.slider("Hour", 0, 23, 12)
        day_of_week_str = st.selectbox("Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        
        days_mapping = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
        day_of_week_num = days_mapping[day_of_week_str]
        
        if st.button("Run Prediction", type="primary"):
            st.session_state.predict = True
    
    with col2:
        st.subheader("Prediction Result")
        
        try:
            model = joblib.load('traffic_model.pkl')
            
            input_data = pd.DataFrame({
                'Junction': [junction],
                'Hour': [hour],
                'DayOfWeek': [day_of_week_num]
            })
            
            prediction = model.predict(input_data)[0]
            prediction = int(max(0, prediction))
            
            st.metric(label="Estimated Vehicle Count", value=f"{prediction} Vehicles")
            
            if prediction > 80:
                st.error("⚠️ Status: HEAVY TRAFFIC")
            elif prediction > 50:
                st.warning("🟡 Status: MODERATE TRAFFIC")
            else:
                st.success("✅ Status: SMOOTH")
                
        except FileNotFoundError:
            st.error("Error: 'traffic_model.pkl' not found. Please run training script first.")
        except Exception as e:
            st.error(f"Prediction error: {e}")

    try:
        if 'model' in locals():
            st.markdown("---")
            st.subheader(f"24-Hour Forecast Trend for Junction {junction} on {day_of_week_str}")
            
            hours_list = list(range(24))
            trend_data = pd.DataFrame({
                'Junction': [junction] * 24,
                'Hour': hours_list,
                'DayOfWeek': [day_of_week_num] * 24
            })
            
            trend_predictions = model.predict(trend_data)
            trend_predictions = [int(max(0, p)) for p in trend_predictions]
            
            chart_df = pd.DataFrame({
                'Hour': hours_list,
                'Predicted Vehicles': trend_predictions
            })
            
            st.line_chart(chart_df.set_index('Hour'))
    except Exception:
        pass

with tab3:
    st.header("AI Traffic Data Assistant")
    
    data_context = ""
    if df is not None:
        data_context = f"""
        Traffic Dataset Information (PostgreSQL):
        - Total records: {len(df)}
        - Average vehicles per hour: {int(df['Vehicles'].mean())}
        - Number of junctions: {df['Junction'].nunique()}
        - Max vehicles in an hour: {df['Vehicles'].max()}
        Role: Data Analyst. Answer concisely based on the data provided above.
        """

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about the data (e.g., When is the peak hour?)"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_prompt = f"{data_context}\n\nUser Question: {prompt}"
            
            try:
                response = requests.post(
                    'http://localhost:11434/api/generate',
                    json={"model": "qwen3:8b", "prompt": full_prompt, "stream": False},
                    timeout=3000
                )
                
                if response.status_code == 200:
                    ai_reply = response.json()['response']
                else:
                    ai_reply = f"Ollama Error: {response.status_code}"
            except requests.exceptions.ConnectionError:
                ai_reply = "AI connection failed. Ensure Ollama is running at localhost:11434."
                
            message_placeholder.markdown(ai_reply)
            
        st.session_state.messages.append({"role": "assistant", "content": ai_reply})