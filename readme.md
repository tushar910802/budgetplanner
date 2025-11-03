📈 MMM Budget Optimizer Streamlit App

This application provides an interactive web interface for a Marketing Mix Model (MMM) Budget Optimizer.

You can set a total marketing budget, define constraints for brand-level and channel-level spending, and run an optimization to find the best budget allocation to maximize your Key Performance Indicator (KPI).

files-in-this-project">Files in this Project

app.py: The main Streamlit application file containing all Python code for the UI and the MMMBudgetOptimizer class.

requirements.txt: A list of all necessary Python libraries.

final_kpi_weekly_reduced.csv: The (required) sample data file used by the optimizer. This file must be in the same directory as app.py.

🚀 How to Run

Install Dependencies:
Open your terminal and install all required libraries from the requirements.txt file.

pip install -r requirements.txt


Run the App:
Make sure you are in the same directory as app.py and your data file. Run the following command:

streamlit run app.py


Streamlit will open the application in a new browser tab.

🏃‍♀️ How to Use

Open the App: Once you run the command above, your browser will open to the app.

Set Parameters: Use the sidebar on the left to configure your optimization:

Total Annual Budget: The total amount (€) you wish to allocate.

Brand Constraints: Set the minimum and maximum percentage of the total budget that any single brand can receive.

Channel Constraints: Set the flexibility for individual channel spending, based on historical min/max values.

Run Optimization: Click the "🚀 Run Budget Optimization" button.

Review Results: The main panel will update with:

High-level metrics (Total KPI, Historical KPI, % Improvement).

The Budget Allocation Plot comparing historical vs. optimal brand budgets.

Tabs containing detailed channel-level plots and model performance graphs.

A full data table of the optimization plan, which you can download as a CSV.