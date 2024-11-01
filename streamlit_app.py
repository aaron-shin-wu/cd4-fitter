import streamlit as st
import logging
import traceback
import time
from solver_simplex import cd4_fitter, get_sample_dist_values

def run_fitter(mean, sd, upper):
    """ Runs back end (optimization fitter in solver_simplex.py). Prints runtime of backend to the terminal.
        Catches errors in the back end and prints traceback to log file.
        Prints inputs, outputs, runtime, and error info to the log file.
        Returns result of optimization if successful, or returns an Exception object if an error occurs. """

    try:
        logging.info('Solver started running w/ target inputs: mean = %.2f, sd = %.2f, upper = %.2f', mean, sd, upper)

        start_time = time.time()

        try:
            result = cd4_fitter(mean, sd, upper)
        except Exception as e:
            end_time = time.time()
            logging.error("An exception occurred while running solver.", exc_info=True)
            logging.info("Time spent running solver before error occurred: %.2f sec", end_time-start_time)
            return -1, e

        end_time = time.time()

        print("Time to run:", end_time - start_time, "seconds")
        logging.info('Solver terminated successfully and took %.2f seconds to run', end_time-start_time)
        logging.info('Solver output was: mean = %.8f, sd = %.8f', result[0], result[1])


        return result

    except Exception as e:
        logging.error("An exception occurred in the application (run_fitter() function).", exc_info=True)



st.title("CD4 Square Root Transformation Fitter Tool")
# adding sidebar for logging
st.sidebar.title("Log")
error_log = st.sidebar.empty()

#initialize a list to store error messages
error_messages = []
st.write(
    "This is the web version of the MPEC Modelling CD4 Fitter Tool v1.01 2024-04-02"
)
st.write("Web version powered by Streamlit. Deployed on 2024-11-01.")
st.write(
    "Please input the desired mean, standard deviation, and upper limit for your target distribution."
)

# User input boxes
target_mean_input = st.number_input("Target Mean:", value=0.0)
target_stdv_input = st.number_input("Target Std Dev:", value=0.0)
upper_limit_input = st.number_input("Upper limit:", value=0.0)


# Calculate button
if st.button("Click to Run"):
    try:
        # Convert inputs to floats
        target_mean = float(target_mean_input)
        target_stdv = float(target_stdv_input)
        upper_limit = float(upper_limit_input)
        
        # Calculate the sum
        result = run_fitter(target_mean, target_stdv, upper_limit)

        # Display the result
        st.write("The sum is:", result)

    except ValueError:
        # Display an error if inputs are not valid numbers
        if target_mean == "" or target_stdv == "":
            st.error("Target mean cannot be blank")
        if target_stdv == "":
            st.error("Target std dev cannot be blank")
        if upper_limit == "":
            st.error("Upper limit cannot be blank")
        else:
            st.error("Please enter valid numbers in both boxes.")



