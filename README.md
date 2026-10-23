# Baseline Predictive Pipeline -- ETAI
## 20231699 Inês Lopes

# Week 2:

(DT) Decision Tree:
    Train accuracy: 0.829
    Test accuracy:  0.628

The Decision Tree has a high training accuracy of 82.9%, but its test accuracy drops to 62.8%. This large difference suggests that the model is overfitting the training data and does not generalize as well to unseen data!

(LR) Logistic Regression:
    Train accuracy: 0.679
    Test accuracy:  0.679

Logistic Regression has the same training and test accuracy of 67.9%. This suggests that the model generalizes more consistently and shows little evidence of overfitting.

So overall, LR performs better on unseen data and is more stable, while DT performs better on the training data but suffers from overfitting.


# Week 3:

After preprocessing, overall test accuracy decreased slightly for both models. The Decision Tree decreased from 62.8% to 61.1%, while Logistic Regression decreased from 67.9% to 65.7%. The Decision Tree still shows substantial overfitting, although its train–test gap decreased from 20.1 to 18.0 percentage points. The most important change appears in class-1 recall: Decision Tree recall decreased from 0.49 to 0.45, while Logistic Regression recall decreased more noticeably from 0.61 to 0.48. This means both models became worse at identifying actual class-1 cases after preprocessing. The class-1 F1-score also decreased for both models. On the other hand, Logistic Regression improved its recall for class 0 from 0.74 to 0.80. Preprocessing also standardized inconsistent race labels, making the false-positive-rate fairness analysis much more reliable.

# Week 4
We compared four classifiers using stratified 5-fold cross-validation on the development set. Logistic Regression achieved the highest mean validation accuracy (0.673) and had a very small train-validation gap (+0.003), indicating stable generalization with little evidence of overfitting. Random Forest reached a lower validation accuracy (0.637) and showed a much larger gap (+0.097), suggesting overfitting. The Decision Tree performed worse, with a validation accuracy of 0.604 and a similarly large gap (+0.096). The Dummy classifier achieved 0.549 accuracy and serves as the baseline, showing that the real models learn useful patterns beyond simply predicting the majority class. Overall, Logistic Regression provided the strongest and most stable cross-validation performance among the tested models.

| Model               | Train mean | Validation mean | Gap  (gap = train perf - valid perf)
| Logistic Regression | 0.676      | 0.673           | +0.003 |
| Random Forest       | 0.734      | 0.637           | +0.097 |  (overfitting)
| Decision Tree       | 0.699      | 0.604           | +0.096 |
| Dummy               | 0.549      | 0.549           | 0.000  |

# Week 5 Hyperparameter Tuning

This week, we applied hyperparameter tuning using Optuna and nested 5-fold cross-validation to improve model performance while avoiding overly optimistic evaluation. Tuning substantially improved the tree-based models: Decision Tree accuracy increased from 60.1% to 67.4%, while Random Forest improved from 63.7% to 68.2%, becoming our best-performing model. Logistic Regression remained stable at 67.3%, suggesting that its original configuration was already effective. The train–validation gaps also decreased significantly for the tree-based models, indicating reduced overfitting and better generalization. Overall, we learned that hyperparameter tuning can improve performance, but its benefits depend on the model. The locked test set remained untouched to preserve an unbiased final evaluation.


# Week 6 – Ensembles and Feature Engineering

This week, I extended the machine learning pipeline by implementing two additional ensemble methods: AdaBoost and an Oblique Random Forest approximation using random projections and bagged decision trees. I also introduced three engineered features: `priors_per_adult_year`, `has_prior_offences`, and `juvenile_offences_total`, calculated within the pipeline to ensure consistent processing during cross-validation. I evaluated the impact of feature engineering across four models using five-fold stratified cross-validation. The results showed that engineered features did not consistently improve predictive performance: Logistic Regression improved slightly from 0.673 to 0.675, while AdaBoost decreased from 0.676 to 0.674. Random Forest and Oblique Random Forest also showed small or inconsistent changes. Separately, Optuna hyperparameter tuning improved the performance of the ensemble models, with AdaBoost achieving 0.684 and Oblique Random Forest achieving 0.682 mean nested cross-validation accuracy. Overall, these experiments demonstrated that adding features does not necessarily improve generalization and that model complexity should be evaluated through validation rather than training performance alone. The locked test set remained untouched throughout these experiments.


# Week 7 – Metrics, Selection Metric and Scoreboard

This week, I added three new models (Gradient Boosting, HistGradientBoosting and Gaussian Naive Bayes) and changed how every model is evaluated. The pipeline now keeps out-of-fold probabilities instead of only yes/no predictions, and reports eight metrics per run (accuracy, balanced accuracy, precision, recall, F1, F2, ROC AUC and average precision), together with a confusion matrix. Every run also adds one row to a single scoreboard file (`results/runs.csv`), next to the no-model floor (a Dummy classifier scored on the same folds), so all runs can be compared directly.

**Scoreboard** (ROC AUC, mean ± std over 5 stratified folds; tuned = nested cross-validation, 30 Optuna trials):

| Model                  | Untuned       | Tuned (nested) |
|------------------------|---------------|----------------|
| Gradient Boosting      | 0.726 ± 0.013 | 0.727 ± 0.013  |
| Random Forest          | 0.690 ± 0.018 | 0.727 ± 0.013  |
| Oblique Random Forest  | 0.679 ± 0.014 | 0.727 ± 0.013  |
| AdaBoost               | 0.722 ± 0.015 | 0.726 ± 0.012  |
| HistGradientBoosting   | 0.716 ± 0.015 | 0.725 ± 0.015  |
| Logistic Regression    | 0.722 ± 0.012 | 0.721 ± 0.011  |
| Naive Bayes            | 0.704 ± 0.010 | 0.719 ± 0.009  |
| Decision Tree          | 0.596 ± 0.009 | 0.717 ± 0.012  |
| Dummy (floor)          | 0.500 ± 0.000 | –              |

Every model is well above the floor of 0.500. Tuning helped most for the models that overfit with their default settings: the Decision Tree improved from 0.596 to 0.717, and Random Forest and Oblique Random Forest from about 0.68–0.69 to 0.727. After tuning, the top five models are all within 0.002 of each other (0.725–0.727), which is smaller than their fold-to-fold standard deviation (about 0.013), so none of them is clearly better than the others. Logistic Regression did not gain from tuning (0.722 vs 0.721), which shows that a simple linear model already captures most of the signal. The locked test set remained untouched.

**Selection metric: ROC AUC.** I chose ROC AUC because it measures how well the model ranks people who reoffend above people who do not, across every possible threshold, so it does not depend on the arbitrary 0.5 cut-off. Unlike accuracy, it is not inflated by the class imbalance (a model predicting "no reoffence" for everyone already gets 0.549 accuracy, but only 0.500 ROC AUC). Since the threshold can be chosen later depending on the cost of false positives and false negatives, which matters for the fairness analysis, a threshold-free metric is the right one for comparing and tuning models.

## Project structure

```
.
├── main.py                # entry point: run the whole pipeline
├── config.yaml             # all tunable settings live here
├── requirements.txt
├── src/
│   ├── data.py             # loading
│   ├── preprocessing.py    # cleaning + train/test split
│   ├── model.py             # model construction
│   ├── evaluate.py         # accuracy metrics + fairness check
│   └── results.py          # saves each run's report to disk
├── results/                # created automatically -- one file per run (not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    └── README.md            # problem description + full data dictionary
```

## Pipeline progress

This table is updated after each practical class, so you can always see what changed in the pipeline and why -- it's a running log, not a fixed syllabus.

| Week | Practical class focus | Added to the pipeline |
|------|------------------------|------------------------|
| 2 | Introduction & baseline pipeline | Initial project structure, single train/test split, simple preprocessing, Logistic Regression and Decision Tree baselines, classification metrics, fairness check and automatic result saving. |
| 3 | Data cleaning & preprocessing | Diagnosis-driven cleaning, placeholder and invalid-value handling, category normalization, missing-value indicators, imputation, encoding and scaling inside a leak-safe preprocessing pipeline. |
| 4 | Cross-validation & model comparison | Replaced the ordinary train/test evaluation with a locked 20% final test set and stratified 5-fold cross-validation on the 80% development set. Added out-of-fold predictions, fold-by-fold train/validation scores, mean ± standard deviation, train-validation gap, Dummy baseline and Random Forest. The final model is refitted on the full development set after cross-validation. |
| 7 | Metrics, selection metric & scoreboard | Added Gradient Boosting, HistGradientBoosting and Gaussian Naive Bayes. Out-of-fold probabilities instead of predictions, eight metrics per run (`cv.metrics`) with a confusion matrix, one selection metric (`cv.selection_metric: roc_auc`) that tuning maximises, a no-model floor scored on the same folds, a corrected paired t-test, and a scoreboard (`results/runs.csv`) with one row per run. Optional `VoteSelector` feature-selection step (`feature_selection` in `config.yaml`). |

## Environment setup

You only need to do this once per machine.

### macOS / Linux
```bash
python3 -m venv venv                 # creates an isolated Python environment in a folder called "venv"
source venv/bin/activate             # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```

### Windows -- PowerShell
```powershell
python -m venv venv                  # creates an isolated Python environment in a folder called "venv"
venv\Scripts\activate                # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```
If PowerShell blocks the activation script, run this once first:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Windows -- cmd.exe
Same three steps as above, just with cmd's own activation command:
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

Once the environment is active you'll see `(venv)` at the start of your prompt. To leave it later, run `deactivate` (same command on every OS).

### Every time after the first

Creating the environment and installing packages only needs to happen once, ever. Every other time you sit down to work -- a new terminal window, the next practical class, tomorrow -- you don't repeat any of the steps above. From the project's root folder, you just need to:

**macOS / Linux**
```bash
source venv/bin/activate
python main.py
```

**Windows**
```powershell
venv\Scripts\activate
python main.py
```

That's it -- activate, then run. If you don't see `(venv)` at the start of your prompt, the environment isn't active and `python main.py` may use the wrong Python (or fail to find a package) entirely.

## Running the pipeline

With the environment active (see above), from the project's root
folder, on any OS:
```bash
python main.py
```

This loads `config.yaml`, loads and preprocesses the data, trains the model, and prints:
- **train accuracy and test accuracy, side by side.** Comparing the two is how you catch overfitting: if the model looks much better on the data it was trained on than on data it's never seen, it has memorised rather than learned something that generalises. 
- a classification report on the test set
- a false-positive-rate-by-race comparison between our model and
  COMPAS's own score

All of this is also saved to a timestamped file in `results/` (e.g.`results/run_20260916_143012.txt`), so it doesn't just scroll past in your terminal -- open it later, or change something in `config.yaml` (like the model type) and compare the new file to the last one.
`results/` is created automatically the first time you run the
pipeline, and isn't tracked in git (see `.gitignore`) since it's
generated output, not source.

You're free to improve on this structure or restructure it entirely -- what matters is that your project stays runnable end-to-end with a single command, and that each piece (data, preprocessing, model, evaluation) stays easy to find and change independently.

## Dataset

See `data/README.md`.
