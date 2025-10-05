import streamlit as st
import pickle
import numpy as np

# Load the model
model = pickle.load(open(r'D:\data science\DATA GURU\PROJECT\Airlines customer satisfaction  prediction model\NOTEBOOK\lgbm_model.pkl', 'rb'))


st.title("✈️ Airline Passenger Satisfaction Prediction 😃")

# UI inputs (unchanged)



type_of_travel = st.selectbox('Type of Travel 🧳', ['Business travel 💼', 'Personal Travel 🏖️'])
class_type = st.selectbox('Class 🪑', ['Eco 🧾', 'Eco Plus 🌟', 'Business 💼'])
flight_distance = st.slider('Flight Distance 📏', 100, 5000, 500)
inflight_wifi = st.slider('Inflight Wifi Service 📶', 0, 5, 3)
ease_of_online_booking = st.slider('Ease of Online Booking 🖥️', 0, 5, 3)
online_boarding = st.slider('Online Boarding 🛫', 0, 5, 3)
seat_comfort = st.slider('Seat Comfort 💺', 0, 5, 3)
inflight_entertainment = st.slider('Inflight Entertainment 🎬', 0, 5, 3)
onboard_service = st.slider('On-board Service 🤝', 0, 5, 3)
leg_room_service = st.slider('Leg Room Service 🦵', 0, 5, 3)
cleanliness = st.slider('Cleanliness 🧼', 0, 5, 3)
departure_delay = st.slider('Departure Delay in Minutes ⏳', 0, 1000, 0)
arrival_delay = st.slider('Arrival Delay in Minutes 🕒', 0, 1000, 0)

if st.button('Predict 🎯'):
    # Feature mapping with original labels
    class_map = {'Eco 🧾': 0, 'Eco Plus 🌟': 1, 'Business 💼': 2}
    travel_map = {'Business travel 💼': 0, 'Personal Travel 🏖️': 1}

    input_data = np.array([[
        online_boarding,
        (arrival_delay + departure_delay) / (flight_distance + 1),
        inflight_wifi,
        class_map[class_type],
        travel_map[type_of_travel],
        inflight_entertainment,
        flight_distance,
        seat_comfort,
        leg_room_service,
        onboard_service,
        ease_of_online_booking,
        cleanliness
    ]])

    prediction = model.predict(input_data)

    st.subheader("Prediction Result 📢") 
    if prediction[0] == 1:
        st.success("✅ The passenger is **satisfied** 🙂")
    else:
        st.error("❌ The passenger is **not satisfied** 🙁")
st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center;'>Developed by <b>Shreyank Pandey</b> ❤️</div>", unsafe_allow_html=True)
