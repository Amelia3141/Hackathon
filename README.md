# Scleroderma Prediction Hackathon Project

This repository contains code and resources for predicting scleroderma from patient data using machine learning and NLP, as well as a Shiny frontend for interactive predictions.

## Project Structure
- `scleroderma_api.py`: FastAPI backend for predictions and recommendations
- `app.R`: R Shiny frontend for user interaction
- `requirements.txt`: Python dependencies
- `.gitignore`: Excludes large data/model files from Git
- `*.joblib`: Model and preprocessing objects (not tracked by Git)
- `*.csv`, `*.txt`: Patient data files (not tracked by Git)

## How to Use
1. **Backend**: Install Python dependencies (`pip install -r requirements.txt`), start FastAPI with `python scleroderma_api.py`.
2. **Frontend**: Open R, install required packages (`shiny`, `httr`, `jsonlite`), and run `shiny::runApp('app.R')`.
3. **Deployment**: Deploy the Shiny app to shinyapps.io for a public frontend; deploy the backend to a public server if needed.

## Notes
- Sensitive patient data and large model files are excluded from version control.
- Update API URLs in `app.R` if deploying backend elsewhere.

---

**For more details or questions, contact the repo owner.**
