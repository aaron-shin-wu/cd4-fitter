import streamlit as st

st.title("🎈 My new app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)

# User input boxes
num1 = st.number_input("Box 1", value = 0.0)
num2 = st.number_input("Box 2", value = 0.0)

# Calculate sum
result = num1 + num2

# Display result
st.write("The sum is:", result)