# 🎓 EduAnalytics Platform

> **Unlocking insights in online education with advanced analytics and interactive dashboards.**

EduAnalytics is a comprehensive analytics platform for exploring, visualizing, and understanding learning effectiveness across major online education platforms. It combines advanced statistics, web scraping, and interactive dashboards to deliver actionable insights for both learners and content creators.

---

## 🚀 Features

- **Learning Intelligence Engine:**
  - Survival analysis of course completion and time-to-mastery
  - Engagement pattern mining and dropout prediction
  - Content difficulty calibration and learning path optimization
- **Content Intelligence Dashboard:**
  - Performance scoring and market gap analysis
  - Quality-price elasticity and competition intelligence
- **Advanced Statistical Analysis:**
  - Survival analysis, ANOVA, chi-square, regression modeling
- **Multi-Platform Data Collection:**
  - YouTube, Coursera, Udemy, Reddit, and more
- **Interactive Visualizations:**
  - Modern, responsive dashboards with Plotly and Streamlit

---

## 🛠️ Tech Stack

- **Backend:** Python, Flask, SQLAlchemy
- **Frontend:** Streamlit, Plotly, Seaborn
- **Database:** Supabase (managed PostgreSQL in the cloud)
- **Data Collection:** Custom scrapers, API integrations
- **Statistical Libraries:** SciPy, Statsmodels, Lifelines, NetworkX

---

## 📊 Dashboard Views

- **🎓 Learner Intelligence:**
  - Analyze optimal learning durations, difficulty success rates, platform effectiveness, and get personalized learning path recommendations.
- **📊 Creator Analytics:**
  - Explore content performance, market opportunities, pricing strategies, and revenue forecasts.
- **🔬 Advanced Insights:**
  - Dive into statistical significance testing, survival analysis, and cross-platform studies.

---

## 📦 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/eduanalytcs.git
cd eduanalytcs
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Secrets
- For local development, create a `.env` file or set environment variables for your database and API keys.
- For Streamlit Cloud, add your secrets to `.streamlit/secrets.toml` (see example in repo).

### 4. Initialize the Database
```bash
python database_setup.py
```
Or use your own PostgreSQL instance and update the `DATABASE_URL` in your secrets.

### 5. Run the Dashboard Locally
```bash
streamlit run dashboards.py
```

### 6. Deploy to Streamlit Cloud
- Push your code to GitHub.
- Go to [Streamlit Cloud](https://streamlit.io/cloud), create a new app, and select `dashboards.py` as the main file.
- Add your secrets in the Streamlit Cloud UI.

---

## 📝 Example Usage

- **test_db.py:** Minimal app to test database connection and schema on Streamlit Cloud.
- **dashboards.py:** Main interactive dashboard for all analytics and visualizations.

---

## 📚 Educational Value

- Advanced statistics and hypothesis testing
- Data engineering and ETL pipelines
- Web scraping and API integration
- Interactive dashboard development
- Business intelligence and data storytelling

---

## 📄 License

See [LICENSE](LICENSE) for details.

---

## 🌐 Live Demo

Try the dashboard online:

[Streamlit Cloud App](https://your-streamlit-app-url.streamlit.app/)

*Replace the above URL with your actual Streamlit Cloud deployment link.*

---

## 🤝 Contributing

Pull requests and issues are welcome! Please open an issue to discuss your ideas or report bugs.
