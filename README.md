# Asset-Dashboard
This app is used to determine certain values, filter out assets, and predict when an asset should be replaced within a company.

# User Manual (PLEASE READ)!!!!
1. Upload Excel file or provide excel file path.
2. Select the sheet of the excel that should be analysed.
3. Scroll down on the sidebar and apply the filters you would like:
     - All assets
     - All assets -> all assets (grouped) ///Used for finding and GENERATING EMAILS for SPECIFIED USERS
     - Users with multiple assets ///All users and their assets are in one large spreadsheet
     - Assets aged > N years ///Scroll down on the sidebar and enter N
     - Immediate replacement required
4. Charts are also generated to help visualize the data.
5. Emails to be sent page: Filters out most important emails that should be sent automatically.

## Quick Start (Recommended)
**Windows:** double-click `run_app.bat`  
**macOS:** double-click `run_app.command`  
The app opens at `http://localhost:8501`.

## Manual Start (Alternative)
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
``
