# Event Attendance Prediction

Free events have a familiar problem: many people register, but far fewer come. This project forecasts **how many registered participants will actually show up** at AIESEC in Astana events. The team can then plan food, seating and materials for the real number of guests instead of the registration count.

A CatBoost model is trained on past registrations with check-in data, then applied to the registration forms of later events.

## How it works

```
Higgsfield registrations + check-in  ─►  CatBoost model  ─►  turnout forecast for Creative One and Business Cup
   (Oct 2025, ~1,800 people)                                  (Jan 2026, 568 regs)  (Apr 2026, 535 regs)
```

1. **Training data:** registrations for *Guest Talks: Higgsfield Edition* (18 Oct 2025), labelled with who checked in.
2. **Features:** education level, university, major group, how early the person registered, age, participation type, and which AIESEC projects they joined before.
3. **Transfer to new events:** each event used a different form, and Creative One's was bilingual (Kazakh / Russian). The notebooks map every form to the same features: universities are normalised, degrees mapped, and free-text majors grouped with keyword rules in three languages.
4. **Forecast:** the model gives each registrant a probability of attending. The headcount forecast is the number of people above a chosen threshold.

## Results

**Model quality on held-out Higgsfield registrations**

| Version | Accuracy | AUC |
|---|---:|---:|
| Full feature set (notebook 1) | — | 0.72 |
| Neutral features shared with Creative One (notebook 2) | 64.4% | 0.69 |
| Unified model without age (notebook 3) | 65.2% | 0.69 |

**Turnout forecasts vs. real attendance**

A registrant is counted as an expected attendee when the model gives them a probability above **0.6**. The same rule was used for both events.

| Event | Registered | Forecast | Actual attendance | Accuracy |
|---|---:|---:|---:|---:|
| Creative One (24 Jan 2026) | 568 | 229 | 201 | 86% |
| Business Cup (5 Apr 2026, `ZimranApril3.csv`) | 535 | 278 | 280 | **99%** |

Accuracy = 100 − |forecast − actual| ÷ actual × 100. The Business Cup forecast was made before the event. Lower thresholds overestimated turnout (for Creative One: 365 at 0.5, 490 at 0.3). The forecasts were used to plan both events.

**Why a 65%-accurate model can still forecast headcount well:** accuracy measures whether the model is right about *each person*. Some people it expects will come stay home, and some it doubts do come. Over hundreds of registrations these errors largely cancel out, so the *total* is much more reliable than any single prediction, and the total is what event planning needs.

**What drives attendance** (Higgsfield model): how early someone registered (18%), education level (18%), university (17%), participation type (16%), age (13%) and major (13%). Previous AIESEC projects mattered very little (under 2% each).

## Notebooks

| Notebook | What it does |
|---|---|
| `1_train_higgsfield_model.ipynb` | Cleans Higgsfield registrations, engineers features, trains CatBoost, plots feature importance |
| `2_forecast_creative_one.ipynb` | Trains on features shared with the Creative One form, validates, forecasts Creative One turnout |
| `3_forecast_zimran.ipynb` | Unified model without age; forecasts Business Cup (Zimran data) and Creative One, reports accuracy and feature importance |

## Data and privacy

Registration exports contain personal data, so **no data is included in this repository** and all notebook outputs have been cleared. Names are dropped or never used as features.

To run the notebooks, put the exports in `data/`:

- `Higgsfield_Registrations.csv`: training data, including the `CheckIn` column
- `C1.csv`: Creative One registrations
- `ZimranApril3.csv`: Business Cup registrations

## How to run

```bash
pip install -r requirements.txt
jupyter notebook
```

## Tech stack

Python, pandas, NumPy, CatBoost, scikit-learn, Matplotlib.
