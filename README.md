# Nifty 500 Trend Scanner

A Streamlit application for the Nifty 500 Minervini trend scan with sector RRG filtering and sector-wise stock batches.

## Run on Streamlit Community Cloud

1. Open the repository on GitHub.
2. Open **Streamlit Community Cloud** and create a new app.
3. Select `vidyasagarsingh2133/stockscanmivini`, branch `main`, and set the main file to `streamlit_app.py`.
4. Deploy.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app downloads the latest Nifty 500 symbol list and daily prices, applies the market, sector, trend, and relative-strength filters, and creates downloadable stock batches by sector.
