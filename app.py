import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
from sqlalchemy import create_engine

st.set_page_config(page_title="Smart Traffic AI", layout="wide", page_icon="🚦")

with st.sidebar:
    st.markdown("### 🚦 Project Information")
    st.markdown("**Course:** Big Data & Gen AI")
    st.markdown("**Institution:** President University")
    st.markdown("**Class:** IS 20241")
    st.markdown("**Developer:** Mohamad Rahman Alamsyah (Alam)")
    st.markdown("---")
    st.success("Database Connection: Online 🟢")
    st.success("AI Engine: Ready 🟢")

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
    except Exception:
        return None

df = load_data_from_db()

if df is None:
    st.error("Failed to connect to the Database. Please ensure PostgreSQL is running.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["📊 Analysis Dashboard", "🔮 Traffic Prediction", "🤖 AI Chat (Data Connected)"])

with tab1:
    st.header("Comprehensive Vehicle Volume Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Data Records", f"{len(df):,}")
    col2.metric("Avg Vehicles/Hour", f"{int(df['Vehicles'].mean())}")
    col3.metric("Max Record", f"{df['Vehicles'].max()}")
    col4.metric("Junctions", f"{df['Junction'].nunique()}")
    
    st.markdown("---")
    
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.subheader("📈 Daily Density Trend")
        daily_trend = df.groupby('Date')['Vehicles'].mean()
        st.line_chart(daily_trend)
        
    with col_chart2:
        st.subheader("⏰ Average Density per Hour")
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
        
        predict_btn = st.button("Run Prediction", type="primary")
    
    with col2:
        st.subheader("Live Prediction Result")
        try:
            model = joblib.load('traffic_model.pkl')
            
            input_data = pd.DataFrame({
                'Junction': [junction],
                'Hour': [hour],
                'DayOfWeek': [day_of_week_num]
            })
            
            prediction = int(max(0, model.predict(input_data)[0]))
            st.metric(label="Estimated Vehicle Count", value=f"{prediction} Vehicles")
            
            if prediction > 80:
                st.error("⚠️ Status: HEAVY TRAFFIC")
            elif prediction > 50:
                st.warning("🟡 Status: MODERATE TRAFFIC")
            else:
                st.success("✅ Status: SMOOTH")
                
        except Exception as e:
            st.error(f"Prediction Error: {e} - Ensure 'traffic_model.pkl' exists.")

    try:
        if 'model' in locals():
            st.markdown("---")
            st.subheader("📊 In-Depth Prediction Analysis")
            
            col_pred1, col_pred2 = st.columns(2)
            
            with col_pred1:
                st.markdown(f"**24-Hour Forecast (Junction {junction}, {day_of_week_str})**")
                hours_list = list(range(24))
                trend_data = pd.DataFrame({'Junction': [junction]*24, 'Hour': hours_list, 'DayOfWeek': [day_of_week_num]*24})
                preds_24h = [int(max(0, p)) for p in model.predict(trend_data)]
                df_24h = pd.DataFrame({'Hour': hours_list, 'Predicted Vehicles': preds_24h}).set_index('Hour')
                st.area_chart(df_24h)
                
            with col_pred2:
                st.markdown(f"**Weekly Trend for {hour}:00 (Junction {junction})**")
                days_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                weekly_data = pd.DataFrame({'Junction': [junction]*7, 'Hour': [hour]*7, 'DayOfWeek': list(range(7))})
                preds_weekly = [int(max(0, p)) for p in model.predict(weekly_data)]
                df_weekly = pd.DataFrame({'Day': days_names, 'Predicted Vehicles': preds_weekly}).set_index('Day')
                st.bar_chart(df_weekly)
                
            st.markdown("---")
            col_pred3, col_pred4 = st.columns(2)
            
            with col_pred3:
                st.markdown(f"**Junction Comparison at {hour}:00 on {day_of_week_str}**")
                junc_list = [1, 2, 3, 4]
                junc_data = pd.DataFrame({'Junction': junc_list, 'Hour': [hour]*4, 'DayOfWeek': [day_of_week_num]*4})
                preds_junc = [int(max(0, p)) for p in model.predict(junc_data)]
                df_junc = pd.DataFrame({'Junction': [f"Junc {j}" for j in junc_list], 'Predicted Vehicles': preds_junc}).set_index('Junction')
                st.bar_chart(df_junc)
                
            with col_pred4:
                st.markdown("**📥 Export Prediction Data**")
                st.write("Download the 24-hour forecast data for further reporting.")
                csv = df_24h.to_csv().encode('utf-8')
                st.download_button(
                    label="Download 24H Forecast (CSV)",
                    data=csv,
                    file_name=f'forecast_junc{junction}_{day_of_week_str}.csv',
                    mime='text/csv',
                )
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
        
        Role: Smart City Traffic Expert. Answer concisely based on the data.
        
        IMPORTANT INSTRUCTION FOR CHART RENDERING:
        If the user asks about "trend", "daily", or "hari", include the exact text [CHART_TREND].
        If the user asks about "peak hour", "time", or "jam", include the exact text [CHART_HOUR].
        If the user asks about "junction", "location", or "persimpangan", include the exact text [CHART_JUNCTION].
        You can use multiple tags in a single response if the user requests multiple charts.
        """

    def render_chat_chart(chart_type, dataframe):
        if chart_type == "trend":
            st.markdown("**📊 Daily Traffic Trend**")
            st.line_chart(dataframe.groupby('Date')['Vehicles'].mean())
        elif chart_type == "hour":
            st.markdown("**⏰ Average Vehicles per Hour**")
            st.area_chart(dataframe.groupby('Hour')['Vehicles'].mean())
        elif chart_type == "junction":
            st.markdown("**📍 Total Vehicles per Junction**")
            st.bar_chart(dataframe.groupby('Junction')['Vehicles'].sum())

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("chart_types") and df is not None:
                for c_type in message["chart_types"]:
                    render_chat_chart(c_type, df)

    if prompt := st.chat_input("Ask about the traffic data..."):
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
            
            detected_charts = []
            if "[CHART_TREND]" in ai_reply:
                detected_charts.append("trend")
                ai_reply = ai_reply.replace("[CHART_TREND]", "")
            if "[CHART_HOUR]" in ai_reply:
                detected_charts.append("hour")
                ai_reply = ai_reply.replace("[CHART_HOUR]", "")
            if "[CHART_JUNCTION]" in ai_reply:
                detected_charts.append("junction")
                ai_reply = ai_reply.replace("[CHART_JUNCTION]", "")
                
            message_placeholder.markdown(ai_reply.strip())
            
            if detected_charts and df is not None:
                for chart in detected_charts:
                    render_chat_chart(chart, df)
            
        st.session_state.messages.append({
            "role": "assistant", 
            "content": ai_reply.strip(),
            "chart_types": detected_charts
        })