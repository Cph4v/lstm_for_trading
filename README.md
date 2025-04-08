# LSTM_FOR_TRADING.GIT

An end-to-end pipeline for real-time cryptocurrency price prediction using a combination of ARIMA and Transformer models. The system fetches minute-level OHLCV (Open, High, Low, Close, Volume) data for Bitcoin from Bybit, preprocesses the data, and displays predictions with animated plots.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Adding API Keys](#adding-api-keys)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

LSTM_FOR_TRADING.GIT is a real-time cryptocurrency forecasting project. It combines traditional time-series models (ARIMA) with deep learning (Transformer) to predict Bitcoin prices. The pipeline fetches minute-level OHLCV data via the Bybit API, processes the data, applies forecasting models, and visualizes the results using animated plots.

---

## Features

- **Real-Time Data Fetching:** Retrieves live Bitcoin price data from the Bybit API.
- **ARIMA Forecasting:** Uses ARIMA for short-term prediction.
- **Transformer Predictions:** Incorporates a Transformer model for advanced forecasting.
- **Dynamic Visualization:** Real-time, animated graphs for forecasting.
- **Scalable:** Can be adapted to other cryptocurrencies by modifying the data source.

---

## Project Structure

```sh
lstm_for_trading.git/
├── main.py                      # Main execution script
├── requirements.txt             # Project dependencies
└── src
    ├── data_preprocessing.py    # Data preprocessing routines
    ├── models                   # Model definitions and pre-trained models
    │   ├── layers.py            # Model layer definitions
    │   ├── transformer.py       # Transformer model implementation
    │   └── 128seq_1min_ohlcv.keras  # Pre-trained model file
    └── utils
        └── postprocessing.py    # Utility functions for result postprocessing

```
## Getting Started

### Prerequisites

- **Python 3.8+**
- **Pip**

### Installation

1. **Clone the Repository:**

   ```sh
   git clone https://github.com/Cph4v/lstm_for_trading.git
   cd lstm_for_trading
   ```
Install Dependencies:

Copy
```sh
pip install -r requirements.txt
```
Adding API Keys
Add your broker's API key and secret by creating an environment file (e.g., .env):
Copy
```sh
BROKER_API_KEY=your_api_key_here
BROKER_SECRET_KEY=your_secret_key_here
```
Usage
Run the main script to start the prediction pipeline:

Copy
```sh
python main.py
```
Contributing
Contributions are welcome!

Report Issues: Submit bugs or feature requests via the Issues page.

Submit Pull Requests: Please review our Contributing Guidelines before submitting any changes.

## License
### This project is licensed under the MIT License.
