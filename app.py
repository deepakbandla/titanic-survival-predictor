import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Titanic Predictor", page_icon="🚢", layout="centered")

@st.cache_resource 
def load_model():
    return joblib.load('titanic_model.pkl')

model = load_model()

st.title("Titanic Survival Predictor")
st.markdown("Adjust the passenger details below to see how different factors affect survival probabilities.")
st.divider()

st.subheader("Passenger Profile")
col1, col2 = st.columns(2)

with col1:
    name = st.text_input("Passenger Name", "Braund, Mr. Owen Harris")
    pclass = st.selectbox("Passenger Class", [1, 2, 3], help="1 = 1st Class, 2 = 2nd Class, 3 = 3rd Class")
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.slider("Age", 0, 100, 25)
    embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"], help="C = Cherbourg, Q = Queenstown, S = Southampton")

with col2:
    sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
    parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
    fare = st.slider("Fare Price (£)", 0.0, 500.0, 32.2)
    cabin = st.text_input("Cabin", "", help="Leave blank if unknown (e.g., C85)")
    ticket = st.text_input("Ticket Number", "A/5 21171")


# --- FEATURE ENGINEERING ---
def apply_feature_engineering(df):
    df = df.copy()

    df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    title_mapping = {"Mr":"Mr", "Mrs":"Mrs", "Miss":"Miss", "Master":"Master"}
    df["Title"] = df["Title"].map(title_mapping).fillna("Rare")

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["FamilyCategory"] = pd.cut(
        df["FamilySize"],
        bins=[0, 1, 4, 20],
        labels=["Alone", "Small", "Large"],
        include_lowest=True
    )

    df["Deck"] = df["Cabin"].str[0].fillna("U")
    if df["Deck"].iloc[0] == "": 
        df["Deck"] = "U"

    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 12, 18, 35, 60, 120],
        labels=["Child", "Teen", "YoungAdult", "Adult", "Senior"]
    )

    df["FareBin"] = pd.cut(
        df["Fare"],
        bins=[-0.001, 7.91, 14.454, 31.0, 600.0],
        labels=["Q1", "Q2", "Q3", "Q4"]
    )

    df["TicketPrefix"] = df["Ticket"].str.extract(r"^([A-Za-z./]+)")
    df["TicketPrefix"] = df["TicketPrefix"].fillna("NUM")

    return df

st.divider()

# --- PREDICTION AND DASHBOARD LOGIC ---
if st.button("Predict Survival", type="primary", use_container_width=True):
    
    raw_data = pd.DataFrame({
        'Name': [name], 'Pclass': [pclass], 'Sex': [sex], 'Age': [age],
        'SibSp': [sibsp], 'Parch': [parch], 'Fare': [fare],
        'Cabin': [cabin], 'Ticket': [ticket], 'Embarked': [embarked]
    })
    
    processed_data = apply_feature_engineering(raw_data)
    
    try:
        probabilities = model.predict_proba(processed_data)[0]
        survival_prob = probabilities[1]
        death_prob = probabilities[0]
        
        st.subheader("Prediction Results")
        
        if survival_prob > 0.5:
            st.success(f"**Outcome: Likely to Survive** 🚢")
        else:
            st.error(f"**Outcome: Unlikely to Survive** 🧊")
            
        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric(label="Probability of Survival", value=f"{survival_prob * 100:.1f}%")
        metric_col2.metric(label="Probability of Perishing", value=f"{death_prob * 100:.1f}%")
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=survival_prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Survival Confidence", 'font': {'size': 20}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "black"},
                'steps': [
                    {'range': [0, 50], 'color': "#ff4b4b"}, 
                    {'range': [50, 100], 'color': "#09ab3b"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        st.subheader("What drove this prediction?")
        st.write("This chart shows how much weight the Random Forest model gave to different features across the dataset:")
        
        clf = model.named_steps["classifier"]
        feature_names = model.named_steps["preprocessor"].get_feature_names_out()
        
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": clf.feature_importances_
        }).sort_values(by="Importance", ascending=False).head(10)
        
        importance_df["Feature"] = importance_df["Feature"].str.replace("num__", "").str.replace("cat__", "")
        
        st.bar_chart(data=importance_df, x="Feature", y="Importance", color="#1f77b4")
        
        with st.expander("View Extracted Features (For Debugging)"):
            st.dataframe(processed_data)
            
    except Exception as e:
        st.error(f"Prediction Error: {e}")