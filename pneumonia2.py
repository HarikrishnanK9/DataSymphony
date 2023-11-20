import streamlit as st
import mysql.connector
from PIL import Image
import io

# Connect to MySQL
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Password@123",
    database="DL_Projects"
)
cursor = db.cursor()

# Create a table if it doesn't exist
table_creation_query = """
CREATE TABLE IF NOT EXISTS patient_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    patient_name VARCHAR(255),
    age INT,
    sex VARCHAR(10),
    patient_email VARCHAR(255),
    doctor_name VARCHAR(255),
    result VARCHAR(100)
);
"""
cursor.execute(table_creation_query)
db.commit()

# Function to insert data into the MySQL table
def insert_data(patient_name, age, sex, patient_email, doctor_name, result):
    insert_query = """
    INSERT INTO patient_data (patient_name, age, sex, patient_email, doctor_name, result)
    VALUES (%s, %s, %s, %s, %s, %s)
    """
    cursor.execute(insert_query, (patient_name, age, sex, patient_email, doctor_name, result))
    db.commit()

# Streamlit App
st.title("Pneumonia Detection App")

# Sidebar for user input
st.sidebar.header("User Input")
patient_name = st.sidebar.text_input("Patient's Name")
age = st.sidebar.number_input("Patient's Age", min_value=0, max_value=150, value=25)
sex = st.sidebar.radio("Sex", options=["Male", "Female"])

patient_email = st.sidebar.text_input("Patient's Email")
doctor_name = st.sidebar.text_input("Doctor's Name")
# doctor_email = st.sidebar.text_input("Doctor's Email")

uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "jpeg", "png"])

if st.button("Detect Pneumonia"):
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-ray Image", use_column_width=True)

        # Perform pneumonia detection using your CNN model here
        # Replace the following line with your actual prediction logic
        result = "Pneumonia Detected" if True else "No Pneumonia Detected"

        # Insert data into the MySQL table
        insert_data(patient_name, age, sex, patient_email, doctor_name, result)  #add doctor_email here

        # Display the result
        st.success(f"Result: {result}")

# Display the table
st.subheader("Patient Data Table")
cursor.execute("SELECT * FROM patient_data")
data = cursor.fetchall()
for row in data:
    st.write(row)

# Close MySQL connection
cursor.close()
db.close()