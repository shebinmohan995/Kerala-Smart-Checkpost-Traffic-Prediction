import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
if 'input_data' not in st.session_state:
    st.session_state['input_data']=None
df=pd.read_csv("Kerala_Government_Smart_CheckPost_Dataset_10000.csv")

st.set_page_config(
    page_title="Kerala Smart Check Post",
    layout="wide"
)


@st.cache_resource
def load_model():
    with open("decision_tree_model.pkl","rb")as f:
        model=pickle.load(f)
    with open("OneHot_Encoder.pkl","rb")as f:
        coder=pickle.load(f)
    return model,coder
model,coder=load_model()

st.title("Kerala Smart Check Post")
st.subheader("Traffic Density Prediction System")


st.sidebar.markdown("Traffic Information")
with st.sidebar.form(key="Traffic_Form"):
    district=st.selectbox(
        "Distirct",
        ["Select District","Thiruvananthapuram","Kollam","Pathanamthitta","Alappuzha","Kottayam",
         "Idukki","Ernakulam","Thrissur","Palakkad","Malappuram",
         "Kozhikode","Wayanad","Kannur","Kasaragod"],index=0
        )

    check_post=st.selectbox(
        "Check_post",
        ["Check_Post 1","Check_Post 2","Check_post 3"]
        )    

    vehicle_type=st.selectbox(
        "Vehicle_Type",
        ["Car","Bus","Truck","Bike"]
        )

    vehicle_age=st.number_input(
        "Vehicle_Age (Years)",
        min_value=0,
        max_value=50,
        value=5
        )

    engine_cc=st.number_input(
        "Engine Capacity (CC)",
        min_value=0,
        max_value=10000,
        value=1500
        )

    passengers=st.number_input(
        "Number of Passengers",
        min_value=0,
        max_value=100,
        value=2
        )

    weather=st.selectbox(
        "Weather",
        ["Sunny","Cloudy","Rainy"]
        )

    road_condition=st.selectbox(
        "Road Condition",
        ["Good","Moderate","Poor"]
        )
    submit=st.form_submit_button("Prepare Prediction")
if submit:
    st.write("### Selected Traffic Information")
    st.write("District:",district)
    st.write("Check Post:",check_post)
    st.write("Vehicle Type:",vehicle_type)
    st.write("Vehicle Age:",vehicle_age)
    st.write("Engine CC:",engine_cc)
    st.write("Passengers:",passengers)
    st.write("Weather:",weather)
    #st.write("Road Condition:"road_condition)
    st.info("Input form is working.Next,we will connect it.")

if submit:
    categorical_columns = [
        'Date', 'Time', 'District', 'Check_Post',
        'Vehicle_Type', 'Vehicle_Brand', 'Fuel_Type',
        'Load_Status', 'Weather', 'Road_Condition',
        'Peak_Hour', 'Festival_Season', 'Holiday',
        'Direction'
    ]

    numerical_columns = [
        'Vehicle_Age_Years', 'Engine_CC', 'Gross_Weight_kg',
        'Passengers', 'Driver_Age_Years',
        'Driver_Experience_Years', 'Temperature_C',
        'Humidity_Percent', 'Rainfall_mm', 'Visibility_km',
        'Average_Speed_kmph', 'Queue_Length_Vehicles',
        'Waiting_Time_Minutes'
    ]

    date=st.date_input("Select Date")
    time=st.time_input("Select Date")
    vehicle_brand=st.selectbox("Vehicle Brand",df["Vehicle_Brand"].unique())
    fuel_type=st.selectbox("Fuel_Type",df["Fuel_Type"].unique())
    gross_weight=st.number_input("Gross_Weight",min_value=0.0)
    load_status=st.selectbox("Load_Status",df["Load_Status"].unique())
    driver_age=st.number_input("Driver_Age",min_value=18,max_value=100,value=30)
    driver_experience=st.number_input("Driver_Experience(Years)",min_value=0,max_value=80,value=5)

    temperature = st.number_input("Temperature (°C)", value=30.0)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=70.0)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=0.0)
    visibility = st.number_input("Visibility (km)", min_value=0.0, value=5.0)

    peak_hour = st.selectbox("Peak Hour", df["Peak_Hour"].dropna().unique())
    festival_season = st.selectbox("Festival Season", df["Festival_Season"].dropna().unique())
    holiday = st.selectbox("Holiday", df["Holiday"].dropna().unique())
    direction = st.selectbox("Direction", df["Direction"].dropna().unique())

    average_speed = st.number_input("Average Speed (km/h)", min_value=0.0, value=30.0)
    queue_length = st.number_input("Queue Length (Vehicles)", min_value=0, value=5)
    waiting_time = st.number_input("Waiting Time (Minutes)", min_value=0.0, value=5.0)


    input_data = pd.DataFrame([{
        'Date': str(date),
        'Time': time.strftime("%H:%M:%S"),
        'District': district,
        'Check_Post': check_post,
        'Vehicle_Type': vehicle_type,
        'Vehicle_Brand': vehicle_brand,
        'Fuel_Type': fuel_type,
        'Vehicle_Age_Years': vehicle_age,
        'Engine_CC': engine_cc,
        'Gross_Weight_kg': gross_weight,
        'Load_Status': load_status,
        'Passengers': passengers,
        'Driver_Age_Years': driver_age,
        'Driver_Experience_Years': driver_experience,
        'Weather': weather,
        'Temperature_C': temperature,
        'Humidity_Percent': humidity,
        'Rainfall_mm': rainfall,
        'Visibility_km': visibility,
        'Road_Condition': road_condition,
        'Peak_Hour': peak_hour,
        'Festival_Season': festival_season,
        'Holiday': holiday,
        'Direction': direction,
        'Average_Speed_kmph': average_speed,
        'Queue_Length_Vehicles': queue_length,
        'Waiting_Time_Minutes': waiting_time
    }])

    st.write("Input data prepared:", input_data.shape)
    st.session_state['input_data']=input_data
    st.dataframe(input_data)
    district_unique=input_data['District'].iloc[0]

    if st.button("Predicted Traffic Density",key="btn_prediction"):
        encoded=coder.transform(st.session_state['input_data'])
        prediction=model.predict(encoded)

        st.success(f"Predicted Traffic Density:{prediction[0]}")
        st.write("Model expects features:", model.n_features_in_,
                 'Input columns',len(input_data.columns))

if st.session_state['input_data'] is not None:

    input_data = st.session_state.get("input_data")

    if input_data is not None:
       st.write("Input columns:", len(input_data.columns))
else:
    st.info('Please fill in the input data above to view predictions and visualizations.')


st.subheader("District Dataset")

district_data = df[
    df["District"].astype(str).str.strip().str.lower()
    == str(district).strip().lower()
]

if not district_data.empty:
    st.dataframe(
        district_data,
        use_container_width=True,
        hide_index=True
    )
else:
    st.warning("No dataset records available for this district.")


st.subheader("District-wise Traffic Visualization")

district_data = df[
    df["District"].astype(str).str.strip().str.lower()
    == str(district).strip().lower()
]

if not district_data.empty:
    traffic_counts = district_data["Traffic_Density"].value_counts()

    plt.figure(figsize=(8, 4))
    plt.bar(traffic_counts.index.astype(str), traffic_counts.values)

    plt.title(f"Traffic Density in {district}")
    plt.xlabel("Traffic Density")
    plt.ylabel("Number of Records")
    plt.xticks(rotation=0)
    plt.tight_layout()

    st.pyplot(plt.gcf())
    plt.close()
else:
    st.warning(f"No records found for {district}.")

st.subheader("Traffic by Time of Day")

df["Hour"] = pd.to_datetime(
    df["Time"].astype(str),
    format="%H:%M",
    errors="coerce"
).dt.hour

df["Time_Period"] = pd.cut(
    df["Hour"],
    bins=[-1, 4, 11, 16, 20, 23],
    labels=["Night", "Morning", "Afternoon", "Evening", "Night 2"]
)

time_counts = df["Time_Period"].value_counts().reindex(
    ["Morning", "Afternoon", "Evening", "Night", "Night 2"],
    fill_value=0
)

plt.figure(figsize=(8, 4))
plt.bar(time_counts.index, time_counts.values)
plt.title("Traffic Distribution by Time of Day")
plt.xlabel("Time Period")
plt.ylabel("Number of Vehicles")
plt.xticks(rotation=0)
plt.tight_layout()

st.pyplot(plt)
plt.close()


st.subheader("Vehicle Type Count by District")

district_data = df[
    df["District"].astype(str).str.strip().str.lower()
    == str(district).strip().lower()
]

if not district_data.empty:

    vehicle_counts = district_data["Vehicle_Type"].value_counts()

    plt.figure(figsize=(8, 8))
    plt.pie(
        vehicle_counts.values,
        labels=vehicle_counts.index,
        autopct="%1.1f%%",
        startangle=90
    )

    plt.title(f"Vehicle Type Distribution in {district}")
    plt.tight_layout()

    st.pyplot(plt.gcf())
    plt.close()

else:
    st.warning(f"No records found for {district}.")


st.subheader("Traffic Density Visualization")
df1=df["Traffic_Density"].value_counts()
plt.bar(df1.index,df1.values)
plt.title("Traffic Density")
plt.xlabel("Traffic")
plt.ylabel("Count")
st.pyplot(plt.gcf())
plt.close()


st.subheader("Total District Wise Traffic Visualization")
df2=df["District"].value_counts()
plt.bar(df2.index,df2.values)
plt.title("Traffic by District")
plt.xlabel("District")
plt.ylabel("Vehicle_Count")
plt.xticks(rotation=45,ha="right")
plt.tight_layout()
st.pyplot(plt.gcf())
plt.close()
