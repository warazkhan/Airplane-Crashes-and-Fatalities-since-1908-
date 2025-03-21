# ✈️ Airplane Crashes and Fatalities Analysis (1908 - Present)

This project explores over a century of **airplane crash data (1908 - present)** to analyze trends in **aviation accidents, fatalities, and safety improvements**. Using **Exploratory Data Analysis (EDA)** and **data visualization**, we extract insights into the evolution of air travel safety.  

<center> <img alt="Insight logo" src="https://github.com/warazkhan/Data-Analysis-Project/blob/main/Flow%20Of%20Exploratory%20Data%20Analysis%20(EDA).gif?raw=true" align="center" hspace="10px" vspace="10px" width=60% height=100% > </center>

---

## 🚀 Project Overview  
- **Dataset**: Historical records of airplane crashes from **1908 onwards**, including crash location, airline, aircraft type, number of fatalities, and possible causes.  
- **Objective**:  
  - Analyze **historical aviation accident trends** and crash frequencies.  
  - Identify **factors affecting fatalities** in crashes.  
  - Explore **changes in aviation safety** over time.  
  - Utilize **data visualization & statistical analysis** to extract key insights.  

---

## 📊 Key Insights & Analysis  
### 1️⃣ **Exploratory Data Analysis (EDA)**  
- 📅 **Yearly Trends**: Examined crash frequencies and fatality rates per year.  
- 🌍 **Geographic Distribution**: Identified **high-risk regions** for airplane crashes.  
- 🏢 **Airline & Aircraft Type Impact**: Analyzed which **operators & aircraft models** had the most accidents.  
- ☠️ **Fatalities & Survivors**: Investigated **survival rates** in crashes over time.  

### 2️⃣ **Statistical & Machine Learning Approaches**  
- 📈 **Time Series Analysis**: Studied **trends in aviation safety improvements**.  
- 🔍 **Feature Correlation**: Identified **key factors influencing fatal crashes**.  
- 🎯 **Predictive Modeling**: Attempted **crash severity prediction** based on flight details.  

---

## 📂 Dataset Overview  
The dataset contains **aviation accident records** with key details:  

| Column Name     | Description |
|----------------|-------------|
| `Date`         | Date of the crash |
| `Time`         | Approximate time of the crash (if available) |
| `Location`     | City, state, or country where the crash occurred |
| `Operator`     | Airline or military branch operating the aircraft |
| `Flight #`     | Flight number (if applicable) |
| `Route`        | Flight path or intended route |
| `Type`         | Aircraft model/type |
| `Registration` | Aircraft registration number |
| `cn/In`        | Aircraft serial or construction number |
| `Aboard`       | Total number of people (passengers + crew) |
| `Fatalities`   | Number of deaths |
| `Ground`       | Number of people killed on the ground |
| `Summary`      | Brief description of the crash event |

---

## 📊 Visualizations & Findings  
- **Yearly crash trends** show **significant safety improvements** in recent decades.  
- **High-fatality crashes** often involve **specific aircraft models & operators**.  
- **Geographic analysis** helps identify **regions with higher accident occurrences**.  
- **Advancements in aviation technology** correlate with reduced crash rates and fatalities.  

---

## 🛠️ Tools & Libraries Used  
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')


