# ⚡ P2S‑WARNING‑SYSTEM  
**Earthquake Early Warning using P‑Wave Detection & Time‑to‑Failure Prediction**

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange?logo=mlflow&logoColor=white)
![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-brightgreen?logo=dvc&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit&logoColor=white)
![Render](https://img.shields.io/badge/Render-Deployed-purple?logo=render&logoColor=white)

---

## 📖 Overview

Earthquakes don’t happen instantly – they unfold in stages.  
When the Earth’s crust suddenly shifts, **two main types of seismic waves** are generated:

- **P‑Waves (Primary Waves)** – fastest, arrive first, usually **not destructive** (a gentle tap).  
- **S‑Waves (Secondary Waves)** – slower, cause **strong shaking** that damages buildings and injures people.

The time gap between detecting a **P‑wave** and the arrival of the **S‑wave** is called **Time to Failure (TTF)**.  
This small window – sometimes just a few seconds – can be enough to:

- 🚂 Stop trains and elevators  
- 🔥 Shut down gas lines  
- 📢 Alert people to take cover  
- 💚 Save lives and reduce damage

---

## 🎯 What This Project Does

This project uses **Machine Learning** to predict:

### 1️⃣ P‑wave Detection (Classification)  
Given live seismic sensor readings, determine if a **P‑wave** has been detected.  
**Output:** `p_wave_detected` (0 = No, 1 = Yes)

### 2️⃣ Time‑to‑Failure Prediction (Regression)  
If a P‑wave is detected, estimate how many seconds remain before the destructive S‑wave arrives.  
**Output:** `ttf_seconds` (e.g., 3.5 seconds)

---

## ⚙️ How It Works

1. **Sensors** collect real‑time seismic data – vibration intensity, background noise, SNR, etc.  
2. **Step 1 – Classification:** The system checks if the signal matches patterns of a P‑wave.  
3. **Step 2 – Regression:**  
   - If **No P‑wave** detected → TTF = 0 seconds.  
   - If **Yes** → Estimate TTF using historical earthquake data patterns.  
4. **Early Warning Alert:** If TTF is above a threshold, an alert is triggered.

---

## 📊 Dataset Used

**Features:**  
- `sensor_reading` – vibration intensity  
- `noise_level` – background noise around the sensor  
- `rolling_avg` – smoothed average of readings over time  
- `reading_diff` – change in vibration between readings  
- `pga` – peak ground acceleration  
- `snr` – signal‑to‑noise ratio  

**Targets:**  
- `p_wave_detected` (classification)  
- `ttf_seconds` (regression)

---

## 🧰 Tech Stack & MLOps Tools

| Category               | Tools                                                                 |
|-------------------- ---|-----------------------------------------------------------------------|
| **Languages**          | Python                                                                |
| **ML & Data**          | Scikit‑learn, Pandas, NumPy                                           |
| **Experiment Tracking* | MLflow                                                                |
| **Data Versioning**    | DVC (Data Version Control)                                            |
| **Containerization**   | Docker                                                                |
| **Frontend / UI**      | Streamlit                                                             |
| **Deployment**         | Render (Cloud)                                                        |

> ✅ The project follows **full MLOps practices** – from data versioning to experiment tracking to containerised deployment.

---

## 🚀 Live Demo (Streamlit)

🔗 **[P2S‑WARNING‑SYSTEM on Render](https://p2s-earthquake-latest.onrender.com/)**  

The Streamlit app lets you:
- Upload live sensor readings (or use sample data)
- See **real‑time P‑wave detection**  
- Get **TTF predictions**  
- Visualise historical warnings

---

## 📸 Screenshots / Diagrams
<img width="1918" height="877" alt="Screenshot 2026-04-20 141633" src="https://github.com/user-attachments/assets/193a45bc-8c41-450d-9178-b1471cf95925" />
<img width="1919" height="729" alt="Screenshot 2026-04-20 141731" src="https://github.com/user-attachments/assets/5ec7cfdc-78fc-458b-91f8-fdee342e5d4c" />
<img width="1918" height="667" alt="Screenshot 2026-04-20 141746" src="https://github.com/user-attachments/assets/373b9ccd-39fe-4b4e-abfe-96a37f48764c" />

---
📈 Why It Matters
Even a 5–10 second warning before an S‑wave hits can give people enough time to:
Move to safety
Prevent train derailments
Stop critical operations
This system is designed to be fast, lightweight, and accurate – suitable for both research and real‑time applications.

---
👨‍💻 Author ----------
Nirabhay Singh Rathod  ----------
📧 nirbhay105633016@gmail.com
