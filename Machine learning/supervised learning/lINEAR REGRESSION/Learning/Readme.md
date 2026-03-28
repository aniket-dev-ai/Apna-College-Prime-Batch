![Rows](https://img.shields.io/badge/Rows-1338-blueviolet) ![Model](https://img.shields.io/badge/Model-Linear%20Regression-green) ![Status](https://img.shields.io/badge/Status-Exploratory%20+%20Visual-orange)

# Insurance Charges Linear Regression
![Rows](https://img.shields.io/badge/Rows-1338-blueviolet) ![Model](https://img.shields.io/badge/Model-Linear%20Regression-green) ![Status](https://img.shields.io/badge/Status-Exploratory%20%2B%20Visual-orange)

# Insurance Charges Linear Regression

> **Notebook destination:** `Machine learning/supervised learning/lINEAR REGRESSION/Learning/code.ipynb`  
> **Data source:** `Machine learning/supervised learning/lINEAR REGRESSION/Learning/insurance.csv`

---

## 🌈 Visual Story
- **Scatterboard:** `sns.scatterplot(x=bmi, y=charges, hue=smoker)` paints a vivid two-tone gradient that immediately flags how smoking status magnifies medical costs as BMI rises.
- **Feature spotlight:** Age, BMI, children, sex, and smoking status are the regressors after dropping `region`. Numeric encoding (`female→1`, `male→0`, `yes→1`, `no→0`) feeds the regression engine.
- **Target:** `charges` (annual medical insurance cost) remains untouched so predictions are comparable to raw dollars.

## 🧾 Data Inventory
| Column | Type | Role | Why it matters |
| --- | --- | --- | --- |
| `age` | numeric | predictor | Older policyholders usually cost more, so it anchors the baseline. |
| `sex` | categorical→binary | predictor | Controls for average rate differences between males and females. |
| `bmi` | numeric | predictor | Body Mass Index has a nonlinear cost signature; visualization isolates its slope. |
| `children` | numeric | predictor | Additional dependents raise premiums slightly. |
| `smoker` | categorical→binary | predictor | The dominant signal: converted to 0/1 and reflected by the hue in the scatterplot. |
| `charges` | numeric | target | What the model is trying to predict (dollars billed). |

## 🎨 Visualization Highlights
- **BMI vs. Charges scatter (code cell 3):** Hue toggles between smokers and non-smokers; clustering happens at the top right for smokers, hinting at interaction effects the plain linear model must absorb.
- **Color cues:** Use the default Seaborn palette to keep the legend readable inside Jupyter and let the `hue` parameter serve as the 3rd dimension.
- **Next graphic idea:** Add `sns.lmplot(..., col="smoker")` as a follow-up to compare slopes side-by-side and justify a nonlinear upgrade.

## 🧠 Notebook Flow
1. **Imports:** `pandas`, `seaborn`, and `LinearRegression` (cells 0, 10).
2. **Data ingestion:** `pd.read_csv("insurance.csv")` (cell 1) and immediate preview of full frame (cell 2).
3. **EDA:** Visualize BMI versus charges with smoking hue (cell 3) to color-code risk tiers.
4. **Feature engineering:** Drop `charges`/`region`, map `sex` & `smoker` to binaries, and keep the remaining 5 predictors (cells 4–7).
5. **Train/test split:** 80/20 with `random_state=42` ensures reproducibility (cell 8).
6. **Model training:** Fit `LinearRegression()` on the training fold (cell 11).
7. **Inference & evaluation:** Predict on test set (cells 12–13) and compute `r2_score` plus adjusted R² (cells 14–15).

## 🛠️ Setup & Dependencies
```bash
pip install pandas seaborn scikit-learn matplotlib
```
- Use Python 3.9+ inside the same virtual environment that runs Jupyter Notebooks.
- Launch the notebook via `jupyter lab` or `jupyter notebook` from the root directory and navigate to the Learning folder.

## 🚀 How to Reproduce the Run
1. `cd "Machine learning/supervised learning/lINEAR REGRESSION/Learning"`
2. `jupyter notebook code.ipynb`
3. Run cells from top to bottom; no hidden state or external script is required.
4. Recreate the scatterplot and metrics to verify the story each time you tweak a feature.

## 📈 Metrics Snapshot
- **R² (test):** `0.7811` — the model explains ~78% of the variance in charges.
- **Adjusted R²:** `0.7770` — after penalizing for the five predictors, the model still retains strong explanatory power.
- **Residual check idea:** Plot `y_test - y_pred` to ensure residuals stay centered around zero.

## 📂 Files & Path Map
- `code.ipynb` – the runnable notebook with cells numbered and documented.
- `insurance.csv` – the canonical dataset with 1,338 rows covering age, sex, BMI, children, smoking, region, and charges.
- `Readme.md` – this visual guide so future collaborators know how to navigate the analysis at a glance.

## 🔮 Suggested Next Steps
1. **Feature scaling & polynomial terms:** Normalize BMI and age, then add quadratic BMI or interaction terms to capture curvature.
2. **Smoker stratified models:** Train separate regressions for smokers vs. non-smokers to compare coefficients directly.
3. **Visual upgrades:** Export the scatterplot to PNG via `plt.savefig(...)` and embed it in this README once finalized.

## 🧭 Credits
- Inspired by the classic Kaggle insurance cost dataset; adapted for classroom practice.
