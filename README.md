# Optimization of Radio Propagation Path Loss Models for Suburban Areas

## 📌 Project Overview

This project focuses on enhancing the predictive accuracy of radio propagation models in suburban environments. By analyzing drive test data from **Yaba, Sango Ota, and Ijebu Ode, Nigeria** at **1800 MHz**, this study optimizes the **COST-231 Hata model** using metaheuristic techniques.

The goal is to provide telecommunication engineers with high-precision tools for network planning, reducing the gap between theoretical predictions and real-world signal attenuation.

---

## 🌍 Study Locations

* 
**Locations:** Yaba, Sango Ota, and Ijebu Ode (Lagos and Ogun State, Nigeria).


* 
**Frequency:** 1800 MHz.


* 
**Dataset:** Real-world drive test campaign measurements including distance, elevation, and path loss.



---

## 🎯 Research Objectives

The project workflow was designed to solve the following:

1. Implement baseline empirical models (**Okumura-Hata, Egli, COST-231**).


2. Evaluate the accuracy of these models against field measurements using **RMSE** and **MAE**.


3. Optimize the COST-231 Hata model parameters using **Particle Swarm Optimization (PSO)**.


4. Optimize the same model using a **Genetic Algorithm (GA)** for 1000 generations.


5. Benchmark the optimized results against traditional models to determine the most effective approach.



---

## 🛠️ Technical Workflow & Implementation

### 1. Data Preparation & Environment Setup

* 
**Library Installation:** Implementation relies on `pyswarms` for optimization and `scikit-learn` for metrics.


* 
**Data Loading:** Measurements are pulled from Excel files (e.g., `Yaba_updated_with_heights_and_freq.xlsx`).


* 
**Feature Engineering:** Essential parameters like Base Station Height (30m) and Mobile Height (1.5m) are integrated into the calculation dataframe.



### 2. Baseline Modeling

* Standard COST-231 Hata Urban/Suburban path loss predictions are generated as a reference point.



### 3. Metaheuristic Optimization (The Core)

The project utilizes two advanced optimization strategies to minimize the error (RMSE) between predicted and actual data:

* **Particle Swarm Optimization (PSO):**
* Utilizes the `pyswarms` library to search for global best coefficients for the path loss equation.


* Achieved the lowest error rates across the study areas.




* **Genetic Algorithm (GA):**
* Processed over **1000 generations** to evolve the model parameters.


* Successfully converged with a best RMSE of approximately **6.445**.





---

## 📊 Results & Performance Insights

### Key KPIs (Results Summary)

* 
**Best GA RMSE:** ~6.445 


* 
**Top Performer:** Particle Swarm Optimization (PSO) 


* 
**Optimization Generations:** 1000 (for Genetic Algorithm) 



### 🧠 Key Insights

* 
**Significant Error Reduction:** Both metaheuristic techniques drastically reduced prediction errors compared to standard models like Egli and Okumura-Hata.


* 
**Environment Adaptation:** The optimized models proved to be more reliable for the specific suburban terrains of Nigeria.


* 
**PSO vs. GA:** While both were effective, PSO performed slightly better in minimizing signal attenuation prediction errors.



---

## 🛠️ Tools & Technologies Used

* **Python:** Primary programming language.
* 
**Pandas & NumPy:** Data manipulation and numerical analysis.


* 
**Pyswarms:** Metaheuristic optimization for PSO.


* 
**Scikit-Learn:** Statistical evaluation (RMSE/MAE).


* 
**Openpyxl:** Handling complex Excel datasets.



---

## 🚀 Conclusion

This project demonstrates a complete technical workflow for wireless network optimization. By transitioning from standard empirical models to metaheuristic-optimized models, the study provides a robust framework for deploying more dependable communication systems in suburban environments.
