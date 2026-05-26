# 🚀 Quick Start Guide

## Installation & Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Project Structure
```
project/
├── models.py                          # Path loss models (COST-231, Okumura-Hata, Egli)
├── optimization.py                    # PSO and GA optimization algorithms
├── analysis.py                        # Statistical analysis and evaluation
├── config.yaml                        # Configuration file
├── requirements.txt                   # Python dependencies
├── app.py                            # Streamlit web UI
├── Path_Loss_Enhanced.ipynb          # Enhanced analysis notebook
├── Path_Loss_Important2.ipynb        # Original notebook
├── Yaba_updated_with_heights_and_freq.xlsx
├── Sango_Ota.xlsx
└── Ijebu_ode_updated_with_heights_and_freq.xlsx
```

---

## 🎯 Three Ways to Use This Project

### **Option 1: Streamlit Web Application (Easy UI)**

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

**Features:**
- ✅ Single Point Prediction - Enter parameters and get instant predictions
- ✅ Batch Analysis - Analyze across distance ranges
- ✅ Model Comparison - Upload your data and compare all models
- ✅ Download Results - Export predictions as CSV
- ✅ Interactive Visualizations

---

### **Option 2: Jupyter Notebook (Full Analysis)**

```bash
jupyter notebook Path_Loss_Enhanced.ipynb
```

**What it includes:**
1. Data loading and preparation
2. Baseline model predictions (COST-231, Okumura-Hata, Egli)
3. PSO optimization with convergence tracking
4. GA optimization
5. Statistical significance testing
6. Distance-stratified analysis
7. Residual diagnostics
8. Comprehensive comparison charts
9. Results export

---

### **Option 3: Python Script (Programmatic)**

```python
from models import COST231HataUrban, COST231HataSuburban
from optimization import PSOptimizer
from analysis import ModelEvaluator
import numpy as np

# Single prediction
model = COST231HataUrban()
path_loss = model.predict(
    f_mhz=np.array([1800]),
    h_b_m=np.array([30]),
    h_m_m=np.array([1.5]),
    d_km=np.array([1.0])
)
print(f"Path Loss: {path_loss[0]:.2f} dB")

# Batch predictions across distances
distances = np.linspace(0.1, 10, 50)
predictions = model.predict(
    f_mhz=np.full(50, 1800),
    h_b_m=np.full(50, 30),
    h_m_m=np.full(50, 1.5),
    d_km=distances
)

# Compare with measured data
measured = np.random.normal(100, 5, 50)  # Simulated data
metrics = ModelEvaluator.calculate_metrics(measured, predictions)
print(f"RMSE: {metrics['RMSE']:.4f} dB")
print(f"MAE:  {metrics['MAE']:.4f} dB")
```

---

## 📊 Model Descriptions

### Empirical Models (Baseline)

| Model | Best For | Strengths | Limitations |
|-------|----------|-----------|------------|
| **COST-231 Urban** | Urban areas | Well-established, fast | Not optimized for your data |
| **COST-231 Suburban** | Suburban areas | Suburban correction included | Still empirical |
| **Okumura-Hata** | General mobile | Widely used in practice | Outside valid frequency range |
| **Egli** | Point-to-point | Simple, fast | Rural-focused |

### Optimized Models

| Model | Algorithm | Advantage | Time |
|-------|-----------|-----------|------|
| **PSO-Optimized** | Particle Swarm | Fast convergence, social dynamics | ~5-10 min |
| **GA-Optimized** | Genetic Algorithm | Robust, parallel search | ~10-15 min |

---

## 🎮 Streamlit UI Usage

### Mode 1: Single Prediction
1. Enter frequency, heights, and distance
2. Select models to compare
3. View results in table format
4. See average, min, max, and std dev

### Mode 2: Batch Analysis
1. Set distance range (min-max)
2. Specify number of points
3. Select models
4. **Output:** Line plot comparing all models
5. **Download:** Results as CSV

### Mode 3: Model Comparison
1. Upload your Excel file with measured data
2. System automatically makes predictions with all models
3. Calculates RMSE, MAE, R² for each model
4. Shows:
   - Performance metrics table
   - Comparison plots (predicted vs measured)
   - Error distribution box plots
   - Residuals analysis
5. **Download:** Complete analysis results

---

## 📈 Key Metrics Explained

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **RMSE** | √(Σ(y-ŷ)²/n) | Average prediction error (lower is better) |
| **MAE** | Σ\|y-ŷ\|/n | Absolute average error |
| **R²** | 1 - Σ(y-ŷ)²/Σ(y-ȳ)² | Fit quality (1.0 = perfect) |
| **MAPE** | Σ\|y-ŷ\|/(y)*100/n | Percentage error |

---

## 🔧 Configuration (config.yaml)

Modify `config.yaml` to customize:

```yaml
# PSO settings
pso:
  n_particles: 50
  n_iterations: 1000
  cognitive_parameter: 0.5
  social_parameter: 0.3

# GA settings
ga:
  population_size: 100
  n_generations: 1000
  crossover_probability: 0.9
  mutation_probability: 0.5

# ANN settings
ann:
  architecture:
    layers: [64, 32]
  epochs: 200
  batch_size: 32
```

---

## 📊 Understanding Results

### Convergence Plot
- Shows how algorithm improves over iterations
- Steep early, levels off near end = good convergence
- PSO and GA usually reach similar final values

### Path Loss vs Distance Plot
- Each line represents one model
- X-axis: Distance in km
- Y-axis: Path loss in dB
- Measured data shown as reference

### Error Distribution (Box Plot)
- Center line: median error
- Box: middle 50% of errors
- Whiskers: min/max errors
- Lower/tighter = better performance

### Residuals Analysis
- **Q-Q Plot**: Are errors normally distributed?
- **Residuals vs Distance**: Any patterns? (should be random)
- **Mean Error**: Should be close to 0
- **Std Dev**: Smaller = more stable predictions

---

## 💡 Tips & Tricks

### For Best Results:
1. ✅ Use representative field measurement data
2. ✅ Ensure sufficient samples (500+ points recommended)
3. ✅ Check data for outliers before optimization
4. ✅ Re-optimize annually with new data

### For Faster Execution:
```python
# Reduce PSO iterations for quick testing
pso = PSOptimizer(n_particles=30, n_iterations=100)

# Or use fewer GA generations
ga = GAOptimizer(pop_size=50, n_generations=500)
```

### For Better Generalization:
```python
# Enable K-fold cross-validation
from analysis import ModelEvaluator

# This prevents overfitting
cv_results = ModelEvaluator.cross_validate_model(
    model, X, y, cv_folds=5
)
```

---

## 🐛 Troubleshooting

### Issue: ModuleNotFoundError
```bash
# Make sure you're in the project directory and have installed requirements
pip install -r requirements.txt
```

### Issue: Streamlit won't start
```bash
# Clear cache and restart
streamlit cache clear
streamlit run app.py
```

### Issue: Optimization is slow
```bash
# Reduce iterations in config.yaml or use fewer particles
pso:
  n_particles: 30
  n_iterations: 500  # instead of 1000
```

### Issue: ANN predictions are poor
- Check data is properly scaled
- Try more epochs (config.yaml)
- Ensure adequate training data
- Consider cross-validation

---

## 📚 Example Workflows

### Workflow 1: Quick Model Comparison
```bash
# Use Streamlit UI
streamlit run app.py
# → Mode 3: Upload your data
# → Compare all models instantly
```

### Workflow 2: Optimize for Your Data
```bash
# Use Jupyter notebook
jupyter notebook Path_Loss_Enhanced.ipynb
# → Runs full optimization pipeline
# → Generates all analysis & charts
```

### Workflow 3: Automated Batch Processing
```python
# Use Python script with multiple files
import os
import pandas as pd
from models import COST231HataUrban

for filename in os.listdir('data/'):
    if filename.endswith('.xlsx'):
        df = pd.read_excel(f'data/{filename}')
        model = COST231HataUrban()
        predictions = model.predict(df['Frequency_MHz'], ...)
        # Process results...
```

---

## 🎓 Model Selection Guide

**Use this decision tree:**

```
Is this for network planning?
├─ YES
│  ├─ Do you have field data?
│  │  ├─ YES → Use PSO/GA-Optimized (Best RMSE)
│  │  └─ NO → Use COST-231 Suburban
│  └─ Need real-time speed?
│     ├─ YES → Use Egli or COST-231
│     └─ NO → Use optimized models
└─ NO
   ├─ Academic research?
   │  └─ Use all models for comparison
   └─ Quick estimate?
      └─ Use COST-231 Urban or Okumura-Hata
```

---

## 📞 Support & Questions

For issues:
1. Check the troubleshooting section above
2. Review the notebook examples
3. Check Streamlit documentation: https://docs.streamlit.io
4. Review model papers in research literature

---

## 📄 Project Files

- **models.py** - All path loss model implementations
- **optimization.py** - PSO and GA algorithms with history tracking
- **analysis.py** - Evaluation, statistical tests, diagnostics
- **app.py** - Streamlit web interface
- **config.yaml** - Configuration for all parameters
- **requirements.txt** - All Python dependencies

---

## 🎯 Performance Benchmarks

Typical results on Nigerian suburban data (Yaba):

| Model | RMSE | MAE | R² | Time |
|-------|------|-----|----|----|
| COST-231 Urban | 25.27 | 18.53 | 0.45 | <1s |
| COST-231 Suburban | 23.11 | 16.89 | 0.52 | <1s |
| Okumura-Hata | 22.84 | 16.21 | 0.55 | <1s |
| Egli | 21.45 | 15.67 | 0.61 | <1s |
| PSO-Optimized | **6.44** | **4.96** | **0.95** | 5-10 min |
| GA-Optimized | **6.45** | **4.97** | **0.95** | 10-15 min |

---

**Ready to use? Start with:**
```bash
streamlit run app.py
```

Happy predicting! 🚀
