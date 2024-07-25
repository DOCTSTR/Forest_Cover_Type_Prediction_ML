# import streamlit as st
# import pickle
# import numpy as np
# from PIL import Image

# rfc = pickle.load(open('rfc.pkl', 'rb'))

# # Creating web app
# st.title('Forest Cover Type Prediction')
# image = Image.open('img.png')
# st.image(image, caption='myimage', use_column_width=True)
# user_input = st.text_input('Input Features')

# if user_input:
#     user_input = user_input.split(',')
#     features = np.array([user_input], dtype=np.float64)
#     output = rfc.predict(features).reshape(1, -1)

#     # Create the cover type dictionary
#     cover_type_dict = {
#         1: {"name": "Spruce/Fir", "image": "img_1.png"},
#         2: {"name": "Lodgepole Pine", "image": "img_2.png"},
#         3: {"name": "Ponderosa Pine", "image": "img_3.png"},
#         4: {"name": "Cottonwood/Willow", "image": "img_4.png"},
#         5: {"name": "Aspen", "image": "img_5.png"},
#         6: {"name": "Douglas-fir", "image": "img_6.png"},
#         7: {"name": "Krummholz", "image": "img_7.png"}
#     }

#     # Convert the output to integer
#     predicted_cover_type = int(output[0])
#     cover_type_info = cover_type_dict.get(predicted_cover_type)

#     if cover_type_info is not None:
#         cover_type_name = cover_type_info["name"]
#         cover_type_image_path = cover_type_info["image"]

#         # Display the cover type card
#         col1, col2 = st.columns([2, 3])

#         with col1:
#             st.write("Predicted Cover Type:")
#             st.write(f"<h1 style='font-size: 40px; font-weight: bold;'>{cover_type_name}</h1>", unsafe_allow_html=True)

#         with col2:
#             cover_type_image = Image.open(cover_type_image_path)
#             st.image(cover_type_image, caption=cover_type_name, use_column_width=True)
#     else:
#         st.write("Unable to make a prediction")
import streamlit as st
import pickle
import numpy as np
from PIL import Image

# Load the pre-trained model
rfc = pickle.load(open('rfc2.pkl', 'rb'))

# Creating web app
st.title('Forest Cover Type Prediction')
image = Image.open('img.png')
st.image(image, caption='myimage', use_column_width=True)

# Manually input features
st.header('Input Features')

# Input fields for the features
Elevation = st.number_input('Elevation')
Aspect = st.number_input('Aspect')
Slope = st.number_input('Slope')
Horizontal_Distance_To_Hydrology = st.number_input('Horizontal Distance To Hydrology')
Vertical_Distance_To_Hydrology = st.number_input('Vertical Distance To Hydrology')
Horizontal_Distance_To_Roadways = st.number_input('Horizontal Distance To Roadways')
Hillshade_9am = st.number_input('Hillshade 9am')
Hillshade_Noon = st.number_input('Hillshade Noon')
Hillshade_3pm = st.number_input('Hillshade 3pm')
Horizontal_Distance_To_Fire_Points = st.number_input('Horizontal Distance To Fire Points')

# Dropdowns for Wilderness Areas
wilderness_areas = ['1 - Rawah Wilderness Area', '2 - Neota Wilderness Area', '3 - Comanche Peak Wilderness Area', '4 - Cache la Poudre Wilderness Area']
selected_wilderness_area = st.selectbox('Wilderness Area', wilderness_areas)

# Create Wilderness Area inputs
Wilderness_Area1 = 1 if selected_wilderness_area == '1 - Rawah Wilderness Area' else 0
Wilderness_Area2 = 1 if selected_wilderness_area == '2 - Neota Wilderness Area' else 0
Wilderness_Area3 = 1 if selected_wilderness_area == '3 - Comanche Peak Wilderness Area' else 0
Wilderness_Area4 = 1 if selected_wilderness_area == '4 - Cache la Poudre Wilderness Area' else 0

# Dropdowns for Soil Types
soil_types = [
    '1 - Cathedral family - Rock outcrop complex, extremely stony.',
    '2 - Vanet - Ratake families complex, very stony.',
    '3 - Haploborolis - Rock outcrop complex, rubbly.',
    '4 - Ratake family - Rock outcrop complex, rubbly.',
    '5 - Vanet family - Rock outcrop complex complex, rubbly.',
    '6 - Vanet - Wetmore families - Rock outcrop complex, stony.',
    '7 - Gothic family.',
    '8 - Supervisor - Limber families complex.',
    '9 - Troutville family, very stony.',
    '10 - Bullwark - Catamount families - Rock outcrop complex, rubbly.',
    '11 - Bullwark - Catamount families - Rock land complex, rubbly.',
    '12 - Legault family - Rock land complex, stony.',
    '13 - Catamount family - Rock land - Bullwark family complex, rubbly.',
    '14 - Pachic Argiborolis - Aquolis complex.',
    '15 - unspecified in the USFS Soil and ELU Survey.',
    '16 - Cryaquolis - Cryoborolis complex.',
    '17 - Gateview family - Cryaquolis complex.',
    '18 - Rogert family, very stony.',
    '19 - Typic Cryaquolis - Borohemists complex.',
    '20 - Typic Cryaquepts - Typic Cryaquolls complex.',
    '21 - Typic Cryaquolls - Leighcan family, till substratum complex.',
    '22 - Leighcan family, till substratum, extremely bouldery.',
    '23 - Leighcan family, till substratum - Typic Cryaquolls complex.',
    '24 - Leighcan family, extremely stony.',
    '25 - Leighcan family, warm, extremely stony.',
    '26 - Granile - Catamount families complex, very stony.',
    '27 - Leighcan family, warm - Rock outcrop complex, extremely stony.',
    '28 - Leighcan family - Rock outcrop complex, extremely stony.',
    '29 - Como - Legault families complex, extremely stony.',
    '30 - Como family - Rock land - Legault family complex, extremely stony.',
    '31 - Leighcan - Catamount families complex, extremely stony.',
    '32 - Catamount family - Rock outcrop - Leighcan family complex, extremely stony.',
    '33 - Leighcan - Catamount families - Rock outcrop complex, extremely stony.',
    '34 - Cryorthents - Rock land complex, extremely stony.',
    '35 - Cryumbrepts - Rock outcrop - Cryaquepts complex.',
    '36 - Bross family - Rock land - Cryumbrepts complex, extremely stony.',
    '37 - Rock outcrop - Cryumbrepts - Cryorthents complex, extremely stony.',
    '38 - Leighcan - Moran families - Cryaquolls complex, extremely stony.',
    '39 - Moran family - Cryorthents - Leighcan family complex, extremely stony.',
    '40 - Moran family - Cryorthents - Rock land complex, extremely stony.'
]
selected_soil_type = st.selectbox('Soil Type', soil_types)

# Create Soil Type inputs
soil_type_values = [1 if soil == selected_soil_type else 0 for soil in soil_types]

# Collect all features into a list
features = [
    Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology,
    Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm,
    Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3,
    Wilderness_Area4
] + soil_type_values

# Convert the features to a numpy array
features_array = np.array([features], dtype=np.float64)

if st.button('Predict'):
    output = rfc.predict(features_array).reshape(1, -1)

    # Create the cover type dictionary
    cover_type_dict = {
        1: {"name": "Spruce/Fir", "image": "img_1.png"},
        2: {"name": "Lodgepole Pine", "image": "img_2.png"},
        3: {"name": "Ponderosa Pine", "image": "img_3.png"},
        4: {"name": "Cottonwood/Willow", "image": "img_4.png"},
        5: {"name": "Aspen", "image": "img_5.png"},
        6: {"name": "Douglas-fir", "image": "img_6.png"},
        7: {"name": "Krummholz", "image": "img_7.png"}
    }

    # Convert the output to integer
    predicted_cover_type = int(output[0])
    cover_type_info = cover_type_dict.get(predicted_cover_type)

    if cover_type_info is not None:
        cover_type_name = cover_type_info["name"]
        cover_type_image_path = cover_type_info["image"]

        # Display the cover type card
        col1, col2 = st.columns([2, 3])

        with col1:
            st.write("Predicted Cover Type:")
            st.write(f"<h1 style='font-size: 40px; font-weight: bold;'>{cover_type_name}</h1>", unsafe_allow_html=True)

        with col2:
            cover_type_image = Image.open(cover_type_image_path)
            st.image(cover_type_image, caption=cover_type_name, use_column_width=True)
    else:
        st.write("Unable to make a prediction")
