# AI-Powered Traffic Congestion Prediction

A deep learning based traffic flow prediction system using LSTM, GRU, and Hybrid LSTM-GRU models.

## Overview

This project predicts future traffic flow using historical traffic data. Three deep learning architectures are implemented and compared:

- LSTM
- GRU
- Hybrid LSTM-GRU

The system performs multi-step traffic forecasting and evaluates the models using MAE, RMSE, and R².

## Dataset

The project uses a traffic dataset containing:

- DateTime
- Junction
- Vehicles
- ID

The current implementation focuses on Junction 1 and uses the Vehicles column for traffic-flow prediction.

## Forecasting Setup

- Previous 12 hourly vehicle counts are used as input.
- Next 5 hourly vehicle counts are predicted.
- Forecast horizons: H0, H1, H2, H3, H4.

## Models

### LSTM

Long Short-Term Memory network for learning temporal dependencies in traffic data.

### GRU

Gated Recurrent Unit network for traffic forecasting.

### Hybrid LSTM-GRU

Combines LSTM and GRU representations to predict future traffic flow.

## Model Evaluation

The models are evaluated using:

- MAE
- RMSE
- R²

In the current test run, the Hybrid LSTM-GRU achieved:

- MAE: 8.1785
- RMSE: 11.2163
- R²: 0.7953

## Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
