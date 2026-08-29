# UK Electricity Demand Forecasting and Anomaly Detection

Final Year Project — B.Sc. (Hons) Computer Science, London South Bank University (2024-2025)

---

## What this project is about

Electricity demand is constantly shifting. A heatwave pushes air conditioning usage up, a major concert fills an arena, a bank holiday empties offices. These fluctuations are hard to predict but critical to manage. If the grid undersupplies, there are blackouts. If it oversupplies, energy is wasted. I chose this project because I wanted to work on a real, live problem with real consequences, not a textbook dataset. The National Grid publishes half-hourly demand data for the whole of England and Wales, which gave me 17,568 genuine records to work with. Beyond the immediate forecasting problem, there is a bigger picture too: as the population grows and the push toward renewable energy accelerates, accurate demand prediction becomes even more important. Renewables like solar and wind are intermittent. You cannot just burn more gas when demand spikes. Knowing what demand will be in advance is what makes a renewable grid actually manageable

---

## Dataset

- Source: National Grid demanddata_2024 (publicly available half-hourly settlement data)
- 17,568 records across 22 features
- Time range: January 2024 to December 2024
- Key columns used: ENGLAND_WALES_DEMAND, Net Demand (ND), Total System Demand (TSD)

---

## What I built

I started with raw half-hourly settlement data from National Grid covering the full year 2024. The first task was just making it usable. I constructed proper timestamps from the settlement date and period columns, selected the relevant demand columns, and capped outliers at the 99th percentile. For anomaly detection I used Isolation Forest, an unsupervised method that flagged 176 unusual demand readings across the year without needing labelled examples of what an anomaly looks like.

For feature engineering I extracted the hour of day, day of week, and month from the timestamps, then added lag features capturing demand from one hour and one day prior. These lag features ended up being the most important inputs to the LSTM because electricity demand follows strong daily and weekly cycles.

I trained four models to compare: Ridge Regression and Lasso as linear baselines, Random Forest as a non-linear benchmark, and a two-layer LSTM as the main model. The LSTM trained in around two minutes on AWS EC2 with the data stored on S3, using early stopping and learning rate reduction to find the optimal stopping point rather than guessing the number of epochs upfront. Random Forest achieved the lowest raw error on the test set (MAE 662 MW, RMSE 1,395 MW). The LSTM achieved MAE 1,022 MW but is architecturally better suited for time series because it learns sequential dependencies that static models cannot capture.

Finally I built a Streamlit dashboard that simulates real-time IoT sensor data streaming, letting users input live sensor readings for demand and temperature and see the values plotted as they come in. The full pipeline including the trained LSTM model was deployed on AWS EC2 with data stored on S3 for the duration of the project. The EC2 instance was shut down after submission to avoid ongoing costs

---

## Models compared

| Model | MAE | RMSE | Performance Notes |
| :--- | :--- | :--- | :--- |
| **Random Forest** | 662.60 MW | 1,395.60 MW | Lowest overall raw error metrics across the test partition space. |
| **LSTM Network** | 1,022.21 MW | 1,813.01 MW | Captured sequential temporal shapes beautifully. Early-stopped at epoch 24. |
| **Ridge Regression** | 928.59 MW | 1,657.67 MW | Linear regularized baseline framework. |
| **Lasso Regression** | 928.60 MW | 1,657.71 MW | Linear framework with automated feature sparsity boundaries. |

---

## Results

![Predictions vs Actual](plots/predictions_full.png)

![Training History](plots/training_history.png)

![Anomaly Detection](plots/anomaly_detection.png)

![Demand Patterns](plots/demand_patterns.png)

![Hourly Distribution](plots/hourly_distribution.png)

---

## Tech stack

- Python, Pandas, NumPy
- scikit-learn (Ridge, Lasso, Random Forest)
- TensorFlow/Keras (LSTM)
- Matplotlib, Seaborn
- Streamlit (interactive dashboard)
- AWS EC2 and S3 (training and deployment)

---

## How to run it

```bash
# Install dependencies
pip install -r requirements.txt

# Run the ML pipeline
python src/main.py

# Run the Streamlit dashboard
streamlit run app.py
```

Note: You need to download demanddata_2024.csv from National Grid's public data portal and place it in the root folder before running.

---

## What I learned

The most important thing I learned was that a model can look fine in training and be completely broken in production. My Streamlit dashboard was running and the predictions were technically being generated, but no matter what parameters I changed on the sliders, the output was always stuck around 240W, nowhere near the real 20,000 to 45,000 MW range of UK demand. It took a long time to figure out that the problem was in how I was scaling the data. I had one scaler doing the job of two. The features and the target were being processed together, which meant the inverse transformation was producing garbage. Once I split them into two independent MinMaxScalers, the predictions jumped from an MAE of 22,456 MW down to 1,022 MW. The model had not changed at all. The bug was in the pipeline, not the network. That is something no textbook really prepares you for.

---

## Author

Syed Muhammad Hassan — B.Sc. (Hons) Computer Science, First Class, London South Bank University
