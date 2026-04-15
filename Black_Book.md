# LEAVE MANAGEMENT AND FORECASTING SYSTEM

## BLACK BOOK PROJECT REPORT

**Project Title:** Leave Management and Forecasting System  
**Domain:** Human Resource Analytics, Workforce Planning, Machine Learning Forecasting  
**Implementation Type:** Python-based data processing, forecasting, and dashboard application  
**Primary Interfaces:** Streamlit dashboard, Flask dashboard, SQL-style analytics dashboard  

---

## **TABLE OF CONTENTS**

| Sr. No. | Title of Chapter | Page No. |
|---|---|---|
| 01 | Introduction | 1 |
| 1.1 | Overview | 1 |
| 1.2 | Motivation | 2 |
| 1.3 | Problem Statement and Objectives | 3 |
| 1.4 | Scope of the Work | 4 |
| 02 | Literature Survey | 5 |
| 2.1 | Review of Recent Literature | 5 |
| 2.2 | Gap Identification / Common Findings from Literature | 6 |
| 03 | Software Requirements Specification | 7 |
| 3.1 | Functional Requirements | 7 |
| 3.1.1 | System Feature 1: Data Ingestion and Consolidation | 8 |
| 3.1.2 | System Feature 2: Data Cleaning and Preprocessing | 9 |
| 3.1.3 | System Feature 3: Forecasting and Model Evaluation | 10 |
| 3.1.4 | System Feature 4: Dashboard Visualization | 11 |
| 3.1.5 | System Feature 5: Reporting and Operational Planning | 12 |
| 3.2 | External Interface Requirements | 13 |
| 3.2.1 | User Interfaces | 13 |
| 3.2.2 | Hardware Interfaces | 14 |
| 3.2.3 | Software Interfaces | 15 |
| 3.2.4 | Communication Interfaces | 16 |
| 3.4 | Nonfunctional Requirements | 17 |
| 3.3.1 | Performance Requirements | 17 |
| 3.3.2 | Safety / Security Requirements | 18 |
| 3.5 | System Requirements | 19 |
| 3.4.1 | Database Requirements | 19 |
| 3.4.2 | Software Requirements | 20 |
| 3.4.3 | Hardware Requirements | 21 |
| 3.6 | SDLC Model to be Applied | 22 |
| 04 | Project Plan | 23 |
| 05 | System Design | 39 |
| 06 | Project Implementation | 44 |
| 07 | Software Testing | 50 |
| 08 | Results | 52 |
| 09 | Conclusions | 55 |
| 10 | References | 58 |
| 11 | Appendix A | 59 |
| 12 | Appendix B | 60 |

---

## 01 INTRODUCTION

### 1.1 Overview

The Leave Management and Forecasting System is a data-driven human resource analytics project designed to convert historical leave records into reliable operational intelligence. The project folder contains raw leave datasets, consolidated CSV and Excel files, generated intermediate outputs, model artifacts, dashboard scripts, visual analysis images, testing files, and supporting documentation. The implemented solution uses Python for data processing, machine learning, forecasting, and dashboard presentation. The major application file, `streamlit_app.py`, builds an interactive dashboard that loads historical leave records, expands leave ranges into daily facts, engineers calendar and workforce features, evaluates saved models, explains predictions, and supports staffing planning. Additional scripts such as `retrain_model.py`, `check_model.py`, `streamlit_sql_visualization.py`, and `web_dashboard.py` provide retraining, validation, SQL-style analytics, and Flask-based visualization capabilities. The system therefore works not only as a leave reporting tool but also as a forecasting decision-support system for HR and operations teams.

### 1.2 Motivation

Organizations frequently face a mismatch between approved employee leave and daily staffing requirements. Traditional leave records are often maintained as spreadsheets, periodic summaries, or isolated HR exports. These sources are useful for administration but limited for proactive planning because they show what already happened rather than what may happen next. The motivation behind this project is to use machine learning and analytics to move from reactive leave tracking to predictive workforce planning. By studying daily leave behavior, department contribution, leave types, holidays, weekends, rolling trends, and employee master data, the system can estimate future leave counts and highlight operational risk periods. The project is especially valuable for departments where absence patterns directly affect production, quality, support coverage, shift planning, or customer service. The dashboards make the results understandable to non-technical users, while the saved model artifacts and generated reports keep the analytical process auditable and repeatable.

### 1.3 Problem Statement and Objectives

The problem addressed by this project is the absence of a centralized, predictive leave intelligence system that can clean historical records, forecast future leave demand, and present the results in a practical dashboard for decision makers. The project aims to solve this by integrating multiple leave datasets, preparing model-ready daily summaries, training forecasting models, evaluating model accuracy, and exposing insights through user-friendly dashboards. The main objectives are to consolidate raw leave data from different periods, clean invalid or inconsistent records, generate daily employee leave facts, engineer time-based and workforce-related features, compare machine learning models such as XGBoost, Gradient Boosting, and Random Forest, select the best forecasting model using metrics such as MAE, RMSE, R2, WAPE, MAPE, and SMAPE, generate future forecasts with prediction intervals, and support staffing decisions through visual analytics. The system also documents model behavior through feature importance, test predictions, model cards, and forecast files.

### 1.4 Scope of the Work

The scope of the work includes data ingestion from CSV and Excel leave sources, historical leave expansion into daily records, model feature generation, forecasting, model validation, dashboard visualization, and operational decision support. The project currently uses file-based datasets such as `Combined_All_Leave_Data.csv`, employee leave period files, relist employee files, and employee master data. It produces processed output files such as `clean_frame.csv`, `expanded_frame.csv`, `feature_df.csv`, and `model_df.csv`, and it stores trained models and metadata under the `artifacts` folder. The dashboards cover daily leave trends, planned versus unplanned leave, cost-centre analysis, special leave and comp-off analysis, leave reason analysis, forecast confidence, risk scoring, and staffing scenarios. The project does not currently implement a full enterprise HRMS login workflow, payroll integration, live ERP synchronization, or role-based production deployment. These items are considered future enhancements and are documented in the report where relevant.

---

## 02 LITERATURE SURVEY

### 2.1 Review of Recent Literature

Recent literature in workforce analytics emphasizes the importance of predictive modeling for absenteeism, leave utilization, and operational capacity planning. Traditional HR analytics relied heavily on descriptive reports, but modern approaches increasingly apply machine learning, time-series analysis, ensemble models, and dashboard-based decision support. Studies on employee absence prediction show that temporal variables such as weekday, month, holiday proximity, seasonal cycles, and past leave behavior are strong predictors of future absenteeism. Research on ensemble learning also shows that tree-based models such as Random Forest, Gradient Boosting, and XGBoost perform well on structured organizational data because they capture nonlinear relationships and feature interactions without requiring strict distribution assumptions. In addition, literature on interpretable machine learning recommends the use of feature importance, residual analysis, validation curves, and explainability methods to help business users trust forecasting systems. The project reflects these ideas through calendar features, lag features, rolling mean features, workforce headcount features, department features, leave type features, XGBoost model training, and visual result validation.

### 2.2 Gap Identification / Common Findings from Literature

Common findings from recent literature show that employee leave and absenteeism are influenced by calendar effects, department workload, policy context, leave type, employee demographics, and previous absence patterns. However, a gap remains between research models and practical HR tools. Many studies demonstrate prediction accuracy in notebooks but do not provide a usable dashboard, retraining process, explainability layer, or operational planning mechanism. Another gap is that many organizations continue to use static spreadsheet reports, making it difficult to forecast staff shortages before they happen. This project addresses these gaps by combining a complete data pipeline with saved model artifacts, model cards, test metrics, future forecast CSV files, interactive Streamlit visualization, DuckDB-based SQL analytics, Flask dashboard support, and staffing scenario planning. The project also identifies areas still needing improvement, such as stronger privacy controls, database-backed deployment, user authentication, automated scheduling, drift monitoring, and formal integration with live HR systems.

---

## 03 SOFTWARE REQUIREMENTS SPECIFICATION

### 3.1 Functional Requirements

The functional requirements describe what the Leave Management and Forecasting System must perform for users and administrators. The system shall ingest historical leave records, validate important columns, clean date fields, expand multi-day leave entries into daily facts, produce model-ready datasets, train or load forecasting models, evaluate model performance, generate future leave predictions, and display insights in dashboards. It shall allow HR users to inspect daily leave count, department-wise contribution, leave type distribution, planned versus unplanned leave, special leave and comp-off patterns, and staffing risks. It shall store generated model artifacts and metadata so that the latest model can be checked independently using `check_model.py`. It shall provide interactive visual charts using Plotly and Streamlit and shall support SQL-style exploration through DuckDB. These requirements are implemented across the primary scripts and supported by generated files in `output`, `artifacts`, and `Images`.

### 3.1.1 System Feature 1: Data Ingestion and Consolidation

The first system feature is data ingestion and consolidation. The project stores raw and combined leave records in folders such as `Data`, `Employee_Leave_Data`, and `relistofemployee`. The system is expected to read large CSV and Excel files containing employee leave details, date ranges, leave types, departments, cost centres, and approval-related attributes. The implementation includes logic for locating project paths, loading source files, reading employee master sheets, and preparing master workforce features. Consolidation is necessary because leave data is split across different historical periods such as October 2022 to December 2022, January 2023 to March 2023, April 2023 to June 2023, and later relist periods. By combining these files, the system creates a single analytical foundation. This feature ensures that downstream forecasting uses a broad historical base instead of isolated monthly or quarterly records.

### 3.1.2 System Feature 2: Data Cleaning and Preprocessing

The second system feature is data cleaning and preprocessing. Leave data frequently contains missing values, inconsistent date formats, spelling differences in leave types, duplicate records, invalid date ranges, and columns that need normalization before model training. The project implements cleaning functions such as `clean_leave_data`, expansion functions such as `expand_leave_records`, and full daily leave expansion logic. The `output` folder shows that the system generates `clean_frame.csv`, `expanded_frame.csv`, `full_expanded_frame.csv`, and `raw_frame.csv`, which indicates a staged preprocessing design. Preprocessing also includes transforming leave ranges into one row per leave day, clipping records to the selected as-of date, preparing employee master information, aligning model feature columns, and ensuring numeric model-ready inputs. This requirement is essential because machine learning accuracy depends strongly on clean and consistent input data.

### 3.1.3 System Feature 3: Forecasting and Model Evaluation

The third system feature is forecasting and model evaluation. The latest artifacts show that the best model is XGBoost, with a training period from 2022-02-18 to 2025-08-26 and a test period from 2025-08-27 to 2026-04-10. The model uses 48 engineered features, including calendar fields, holiday indicators, workforce headcount variables, department encodings, leave type daily variables, lag values, and rolling statistics. The current test metrics show MAE of about 7.85, RMSE of about 14.55, R2 of about 0.9981, WAPE of about 3.77%, and SMAPE of about 13.60%. The retraining script compares XGBoost, Gradient Boosting, Random Forest, and a naive lag baseline using validation and test metrics. The system also creates future forecast files such as `leave_forecast_next_60days_20260410_180739.csv`, making the prediction output available for reporting and planning.

### 3.1.4 System Feature 4: Dashboard Visualization

The fourth system feature is dashboard visualization. The project includes `streamlit_app.py`, `streamlit_sql_visualization.py`, and `web_dashboard.py`. The Streamlit application is the richest dashboard and contains sections for forecast confidence, daily leave dashboard, special leave and comp-off analysis, cost centre wise analysis, planned versus unplanned leave, leave reason analysis, prediction context, and staffing scenario planning. The SQL visualization script uses DuckDB to execute analytical queries over CSV data and presents daily summary, cost centre analysis, leave type distribution, planned versus unplanned events, special leave, reasons, and raw data exploration. The Flask dashboard in `web_dashboard.py` renders Plotly HTML visualizations through a browser template. This feature is important because HR users need visual and interactive tools rather than raw CSV outputs or terminal logs.

### 3.1.5 System Feature 5: Reporting and Operational Planning

The fifth system feature is reporting and operational planning. The system creates model cards, test metrics CSV files, test prediction CSV files, feature importance CSV files, forecast files, and generated visual images such as daily leave trend, monthly leave distribution, weekday leave analysis, department contribution, leave type distribution, holiday leave spikes, forecast comparison, feature importance, SHAP summary, actual versus predicted holdout, residual distribution, residuals versus predicted, and residuals over time. These artifacts support academic reporting and operational auditability. The dashboard also includes a staffing planner that estimates surplus or shortage based on predicted leave, total workforce, required present workforce, known absences, and safety buffer settings. This means the project is not only a prediction engine but also a planning assistant that helps managers convert forecasts into staffing decisions.

### 3.2 External Interface Requirements

External interface requirements define how users, software components, hardware, and communication channels interact with the system. The Leave Management and Forecasting System is primarily file-driven and dashboard-driven. Users interact with the system through Streamlit pages, Flask-rendered pages, dataframes, Plotly charts, sidebar filters, date selectors, multiselect boxes, and numeric inputs. Software components interact through CSV files, Excel files, pickle model artifacts, model metadata, and generated output files. Hardware interaction is indirect because the application runs on a workstation or server using CPU, RAM, disk, and network resources. Communication is currently local HTTP when dashboards are launched, although the same design can be extended to an intranet server. These interfaces make the project flexible enough for academic demonstration and practical HR analytics usage.

### 3.2.1 User Interfaces

The user interface requirement is implemented mainly through Streamlit and supplemented by a Flask HTML dashboard. The Streamlit dashboard provides headings, tabs, expanders, metrics, interactive date inputs, charts, tables, warnings, captions, and filtered views. HR users can inspect the daily forecast, view confidence ranges, compare actual and predicted values, identify risky cost centres, study leave type patterns, and build staffing scenarios. The SQL analytics dashboard provides another interface for users who want structured query-style summaries without writing SQL manually. The Flask dashboard uses a template file under `templates/dashboard.html` and renders Plotly charts as HTML. The interface design prioritizes accessibility and clarity for users who may not be machine learning experts, making complex model outputs understandable through operational language and visual summaries.

### 3.2.2 Hardware Interfaces

The system does not require specialized hardware interfaces such as sensors, biometric devices, or external attendance machines in its current implementation. It operates using standard computing hardware: a laptop, desktop, or server capable of running Python, storing large CSV files, and loading machine learning artifacts. The datasets in the project are large, with some generated CSV files exceeding tens of megabytes, so sufficient disk performance is beneficial. A multi-core CPU improves preprocessing and model training time, while adequate RAM is necessary when reading expanded daily leave records. GPU acceleration is not mandatory because the implemented tree-based models can run on CPU, although a GPU could be useful for future deep learning experiments. Hardware interaction is therefore limited to normal file system, memory, processor, and display usage.

### 3.2.3 Software Interfaces

The software interfaces include Python packages, data files, model artifacts, and dashboard frameworks. The `requirements.txt` file lists pandas, NumPy, Matplotlib, Seaborn, scikit-learn, XGBoost, Plotly, holidays, SHAP, TensorFlow, joblib, Streamlit, Flask, openpyxl, DuckDB, and nbformat. Pandas and NumPy handle tabular data operations, scikit-learn and XGBoost support modeling, Plotly and Streamlit provide visualization, joblib loads and saves model artifacts, openpyxl supports Excel reading, holidays supports calendar feature creation, SHAP supports explainability when available, and DuckDB provides SQL-style analysis over local files. The retraining script interacts with functions from the Streamlit application by parsing selected functions, which shows a practical reuse strategy. Software interfaces are mainly local and file-based, but they form a complete analytics pipeline.

### 3.2.4 Communication Interfaces

The current communication interface is local HTTP communication between the dashboard server and the browser. When Streamlit or Flask is started, users access the application through a browser URL, usually on localhost or a configured network address. Data communication between modules is file-based: raw CSV and Excel files are read from disk, processed CSV files are written to the `output` folder, and model artifacts are loaded from the `artifacts` folder. There is no current external API integration with HRMS, ERP, payroll, email, or notification systems. If implemented in production, communication interfaces should include secure HTTPS, authenticated API endpoints, database connections, scheduled data ingestion jobs, and role-based access. For the academic project, the existing local communication design is sufficient to demonstrate forecasting and dashboard capability.

### 3.4 Nonfunctional Requirements

Nonfunctional requirements define the quality attributes of the system rather than individual functions. This project must be accurate enough for workforce planning, fast enough for interactive dashboard use, understandable enough for HR users, maintainable enough for future enhancement, and secure enough to handle employee-related data responsibly. The model artifacts indicate strong predictive performance, but the system should continue to monitor accuracy because leave behavior can change due to policy changes, production schedules, festivals, hiring, attrition, or unusual events. The dashboard should remain responsive when filtering historical data and generating forecasts. The code should be modular, documented, and testable. Since employee data can contain sensitive information, nonfunctional requirements also include privacy, access control, data minimization, secure storage, and restricted sharing of raw employee-level records.

### 3.3.1 Performance Requirements

Performance requirements include acceptable runtime for data loading, preprocessing, prediction, dashboard rendering, and model retraining. The dashboard should load saved artifacts quickly enough that users can inspect forecasts without waiting for model training. Model training is expected to take longer because it includes feature generation, candidate model fitting, walk-forward validation, and artifact saving. The output files show that preprocessing can create large expanded datasets, so memory-efficient pandas operations and caching are important. Streamlit caching is used in several data access functions to avoid repeated expensive calculations. The model currently produces 60-day future forecasts, which is computationally light once features are prepared. Performance can be improved further by database indexing, incremental data refresh, Parquet storage, scheduled retraining, and separation of heavy training workflows from daily dashboard use.

### 3.3.2 Safety / Security Requirements

Safety and security requirements are important because leave records are employee-related and may reveal absence patterns, departments, reasons, and personal planning behavior. The project should protect raw data files, generated expanded employee-level data, model outputs, and dashboards from unauthorized access. In the current local implementation, security depends on file system permissions and responsible handling by the project owner. For production deployment, the system should add authentication, authorization, HTTPS, audit logs, encrypted storage, privacy masking, role-based dashboards, and data retention rules. The dashboard should avoid exposing unnecessary personal identifiers unless a user has HR permission. Predictions should be presented as planning support, not as a basis for unfair employee monitoring. Ethical use requires transparency, human review, and clear communication that forecasting is aggregate operational intelligence rather than individual judgment.

### 3.5 System Requirements

The system requirements describe the database, software, and hardware environment needed to run the project. The current project is built as a local analytics application using CSV, Excel, and pickle artifacts. It requires Python and the libraries listed in `requirements.txt`. It also requires sufficient storage for raw datasets, generated outputs, images, model files, and notebooks. The application can run on a normal development workstation, but for smoother operation with large CSV files, 16 GB RAM is recommended. The system should support command-line execution for model checking and retraining, browser access for dashboards, and spreadsheet compatibility for data sources. The project is organized so that raw data, output data, documentation, image results, artifacts, tests, templates, and application scripts can be understood separately.

### 3.4.1 Database Requirements

The implemented system currently uses file-based storage rather than a relational database server. The effective database consists of CSV and Excel files in folders such as `Data`, `Employee_Leave_Data`, `relistofemployee`, and `output`. The SQL visualization script uses DuckDB to query CSV data, which gives SQL-like analysis without requiring a separate database installation. For academic and prototype use, this is practical because data files are easy to inspect, share, and regenerate. For production, a proper database design should be implemented with tables for employees, departments, cost centres, leave applications, leave types, daily leave facts, forecast runs, model metrics, and forecast outputs. Indexes should be created on employee ID, leave date, cost centre, department, leave type, and forecast date to support fast filtering and reporting.

### 3.4.2 Software Requirements

The software requirements include Python 3.x and all packages listed in `requirements.txt`: pandas, NumPy, Matplotlib, Seaborn, scikit-learn, XGBoost, Plotly, holidays, SHAP, TensorFlow, joblib, Streamlit, Flask, openpyxl, DuckDB, and nbformat. Streamlit is required to run the main interactive application. Flask is required for the secondary web dashboard. DuckDB is required for SQL-style dashboard analysis. XGBoost and scikit-learn are required for model training and evaluation. Joblib is required for reading and writing `.pkl` artifacts. Openpyxl is required for Excel files such as employee master data. Plotly, Matplotlib, and Seaborn are used for charts and visual evidence. A code editor such as VS Code is useful, and Git is used for version control of scripts and documentation.

### 3.4.3 Hardware Requirements

The recommended hardware includes a 64-bit processor, at least four CPU cores, 8 GB minimum RAM, 16 GB recommended RAM, and several gigabytes of free storage for datasets, generated CSV files, model artifacts, images, and notebooks. Since the project expands leave ranges into daily-level records, memory usage can increase significantly during preprocessing. SSD storage is recommended because large CSV files are repeatedly read and written. A GPU is not mandatory because the current best model is XGBoost and can run effectively on CPU. For deployment to multiple HR users, a server with stable network access, backup storage, and monitoring would be preferable. For academic demonstration, a standard laptop or desktop with Python installed is sufficient to run model checks and dashboards.

### 3.6 SDLC Model to be Applied

An iterative and incremental SDLC model is most suitable for this project. The reason is that machine learning systems improve through repeated cycles of data understanding, preprocessing, feature engineering, model training, validation, dashboard feedback, and refinement. A strict waterfall model would be less effective because requirements evolve when new data issues, feature importance results, residual patterns, and user expectations are discovered. In this project, evidence of iteration appears in multiple model artifacts with different timestamps, archived outputs, retraining scripts, dashboard variants, README files, and evolving generated images. Each iteration improves a portion of the system: data cleaning, model selection, forecast horizon, visualization, and operational planning. Agile-style increments therefore fit the project because HR users can review dashboards early while developers continue improving model accuracy and usability.

---

## 04 PROJECT PLAN

### 4.1 Project Cost Estimation

Project cost estimation includes computation cost, development effort, storage cost, software cost, testing effort, and maintenance effort. Since the project uses open-source Python libraries, direct licensing costs are low. The main cost is developer time for data cleaning, model development, dashboard building, report preparation, and validation. Computational cost is moderate because model training uses CPU-based XGBoost, Gradient Boosting, and Random Forest models on structured tabular data. Storage cost is also moderate, but the project contains large CSV files, Excel files, processed outputs, notebook files, images, and artifacts, so SSD space must be planned. Operational cost depends on deployment: local academic use is inexpensive, while a production HR dashboard would require server hosting, secure backups, scheduled data ingestion, access control, and periodic model retraining.

### 4.1.1 Computational Costs

Processing power requirements are moderate. A multi-core CPU is sufficient for pandas preprocessing, feature generation, model training, and dashboard use. GPU cost is not necessary for the implemented XGBoost model, although TensorFlow is listed as a dependency and could support future deep learning work. Memory usage is a more important cost factor because expanded leave datasets can become large; 16 GB RAM is recommended to avoid slowdowns. Storage requirements include raw files, combined files, processed CSV outputs, model artifacts, archived model versions, visual images, and notebooks. SSD storage improves read/write speed for large CSV files. Network latency and bandwidth costs are minimal in local use because the system reads local files, but they would increase if data is fetched from cloud storage, HRMS APIs, or hosted dashboards.

### 4.1.2 Software Performance Costs

Software performance cost is influenced by algorithm complexity, data volume, feature engineering operations, and dashboard rendering. Pandas operations over large CSV files can be memory intensive, especially when leave date ranges are expanded into daily rows. Model complexity is manageable because tree-based ensemble models train on tabular features and output daily aggregate predictions. XGBoost training has higher computational cost than a naive baseline but gives much better accuracy, as seen from the latest model metrics. Database query performance is currently handled through file-based reads and DuckDB queries, but production use would benefit from indexed database tables or columnar files such as Parquet. Cloud service performance cost would include CPU time, memory allocation, load balancing, and data transfer, but the current local design avoids those recurring costs.

### 4.2 Sustainability Assessment

The sustainability assessment considers environmental, economic, and social impact. This project uses existing employee data and open-source software, reducing the need for expensive proprietary tools. It can improve resource planning by helping organizations avoid overstaffing, understaffing, and emergency scheduling. From an environmental point of view, the system should avoid unnecessary repeated training and should reuse saved model artifacts when possible. From an economic point of view, accurate forecasting supports better shift planning, lower overtime pressure, and more efficient workforce utilization. From a social point of view, the project should be used ethically, with privacy safeguards and fair interpretation. Sustainability therefore depends not only on technical optimization but also on responsible organizational adoption.

### 4.2.1 Environmental Sustainability

Environmental sustainability focuses on reducing unnecessary computing and hardware waste. The system uses CPU-based machine learning and local artifacts, which is energy-efficient compared with repeatedly training large deep learning models. Energy consumption can be controlled by separating heavy retraining tasks from daily dashboard viewing, caching data in Streamlit, and only retraining when new data significantly changes. The carbon footprint can be reduced by using efficient algorithms, avoiding needless cloud compute, and archiving only necessary model versions. E-waste management is indirectly relevant because the project can run on existing laptops or desktops instead of requiring specialized hardware. Sustainable computing practices include saving reusable artifacts, pruning duplicate outputs, using optimized file formats, and scheduling retraining during low-demand periods.

### 4.2.2 Economic Sustainability

Economic sustainability is achieved by building the system with open-source tools and reusable project components. The project avoids expensive commercial BI or forecasting software by using Python, Streamlit, Flask, DuckDB, Plotly, scikit-learn, and XGBoost. Cost efficiency improves because HR teams can use the dashboard to anticipate staffing shortages and plan replacement coverage before problems occur. Resource utilization improves because the system identifies leave-heavy periods and department-level risks, helping managers allocate employees more rationally. Scalability can be achieved by migrating from CSV storage to a database, using scheduled pipelines, and adding server deployment only when user demand justifies it. Long-term viability depends on maintaining clean data inputs, documenting retraining procedures, and periodically checking whether model performance remains acceptable.

### 4.2.3 Social Sustainability

Social sustainability requires that the system support people rather than penalize them. Leave forecasting should help departments plan work fairly, not discourage employees from using legitimate leave. Accessibility is supported through dashboards, charts, captions, and tables that simplify complex analytics for non-technical users. Ethical considerations include employee privacy, responsible access to leave reasons, transparency about prediction uncertainty, and avoiding individual-level misuse of aggregate forecasts. Open-source contribution is possible because the system uses common Python libraries and can be improved by adding tests, documentation, and modular pipelines. Skill development is also supported because the project exposes students and team members to data engineering, machine learning, dashboard design, model evaluation, and HR analytics in one practical workflow.

### 4.3 Complexity Assessment

The project has moderate to high complexity because it combines multiple technical areas: data cleaning, time-series style feature engineering, machine learning model comparison, artifact management, dashboard development, SQL-style analytics, and academic reporting. The complexity is not caused by one single algorithm but by the end-to-end nature of the workflow. Raw data must be interpreted correctly, multi-day leave ranges must be converted to daily records, features must align between training and prediction, models must be evaluated without leakage, dashboards must remain responsive, and results must be explained to users. The system also handles several output layers: processed data, model files, metadata, test predictions, feature importance, images, and interactive views. Managing consistency across these layers is the central complexity of the project.

### 4.3.1 Computational Complexity

Computational complexity is mainly driven by dataset size, leave expansion, model training, and validation. Data cleaning is generally O(n), where n is the number of raw leave records, but expansion can increase the number of rows based on leave duration, making the effective size closer to the total number of leave days. Feature generation with rolling windows and lag variables depends on the number of daily records and selected window sizes. Model training time for XGBoost depends on the number of samples, features, trees, depth, and boosting iterations. Memory usage grows with raw frames, expanded frames, feature frames, and cached dashboard data. The latest model uses 48 engineered features and walk-forward validation, which increases computation but provides better reliability than a single random split.

### 4.3.2 Algorithmic Complexity

Algorithmic complexity includes cleaning rules, feature engineering logic, model selection, iterative forecasting, and dashboard-specific calculations. The preprocessing algorithm must parse dates, remove invalid rows, normalize columns, expand ranges, and aggregate daily counts. The forecasting algorithm must build calendar features, holiday indicators, lag features, rolling statistics, department features, leave type features, and workforce features. The training algorithm compares candidate models and ranks them using metrics such as WAPE, MAE, validation stability, and test performance. The dashboard algorithms calculate risk summaries, leave intelligence summaries, cost centre ranking, planned versus unplanned splits, staffing shortage estimates, and prediction explanations. Each algorithm is individually understandable, but their combination creates a layered system where data consistency and feature alignment are crucial.

### 4.3.3 Implementation Complexity

Implementation complexity is visible in the number and size of project files. The main Streamlit application is large and contains many functions for paths, feature creation, model loading, evaluation, explanation, forecasting, caching, visualization, and dashboard sections. The retraining script contains configuration handling, candidate model construction, walk-forward splits, model fitting, metric generation, JSON-safe model card creation, artifact saving, and forecast generation. The project also includes tests under `employee-leave-forecasting-system/tests`, a Flask dashboard, a SQL visualization dashboard, generated images, model artifacts, and multiple documentation files. Lines of code are distributed across scripts rather than a formal package structure, which is practical for a project prototype but increases maintenance complexity. Future improvement should modularize common functions into reusable packages.

### 4.3.4 Resource Complexity

Resource complexity includes CPU, RAM, storage, and deployment resources. CPU demand is moderate during dashboard use and higher during retraining. RAM demand can be significant because the project stores raw, clean, expanded, full expanded, feature, and model dataframes. Storage demand is clear from the large data and output files, including `raw_frame.csv`, `clean_frame.csv`, `expanded_frame.csv`, `full_expanded_frame.csv`, and many model artifact versions. Cloud infrastructure is not currently required, but production deployment would need a secure server, database, backup mechanism, monitoring, and user authentication. Scalability depends on replacing repeated CSV reads with optimized storage, separating training from serving, and using incremental updates rather than rebuilding all data from scratch for every refresh.

### 4.3 Risk Management

Risk management is necessary because forecasting systems can fail due to poor data quality, changing leave policies, model drift, security gaps, or wrong interpretation. In this project, the technical risks include missing source files, malformed dates, duplicate records, large memory usage, unavailable dependencies, corrupted model artifacts, and mismatch between training feature columns and prediction feature columns. Business risks include overreliance on forecasts, ignoring prediction intervals, using aggregate predictions for individual employee judgment, or failing to update the model after organizational changes. Academic risks include incomplete documentation, missing screenshots, missing plagiarism report, or lack of certificate attachments. The project reduces some risks through model metadata, test metrics, saved artifacts, model checking, and visual validation, but production readiness would require stronger governance.

### 4.3.1 Risk Identification

The major risks identified for the Leave Management and Forecasting System are data quality risk, model performance risk, model drift risk, privacy risk, dashboard usability risk, dependency risk, storage risk, and deployment risk. Data quality risk occurs when leave dates, employee identifiers, cost centres, or leave types are inconsistent. Model performance risk occurs when the forecasting model performs well historically but fails on future changes. Model drift risk occurs when workforce size, policies, holidays, production cycles, or leave behavior changes. Privacy risk exists because employee leave data is sensitive. Dashboard usability risk occurs if users misread visualizations or ignore confidence bounds. Dependency risk occurs if required Python libraries are missing. Storage risk occurs due to large generated CSV files. Deployment risk appears when moving from local use to multi-user production.

### 4.3.2 Risk Analysis

Data quality risk has high probability and high impact because messy HR exports are common and directly affect model reliability. Model drift has medium probability and high impact because leave patterns can change after policy updates, hiring waves, attrition, or unusual events. Privacy risk has high impact even if probability is controlled, because unauthorized exposure of employee leave information can damage trust and violate policy. Dashboard usability risk has medium impact because misleading interpretation may lead to poor staffing decisions. Dependency and environment risk has medium probability in academic settings because Python versions and package installations can vary. Storage risk has medium probability because generated output files can grow large. The existing project mitigates some risks with validation metrics, model cards, test files, and cached functions, but additional production controls are recommended.

### 4.3.3 Overview of Risk Mitigation, Monitoring, Management

Risk mitigation should include schema validation, duplicate detection, date validation, missing value handling, model metric monitoring, access control, and documentation. Data quality should be monitored whenever new leave files are added. Model performance should be checked after retraining using MAE, RMSE, R2, WAPE, MAPE, and SMAPE, and results should be compared with previous artifacts. Drift monitoring should compare recent actual leave counts against forecast intervals. Privacy should be managed through restricted file permissions, removal of unnecessary personal fields, and role-based dashboard access in production. Dashboard risks should be reduced with clear labels, confidence intervals, explanations, and warnings. Artifact management should keep model cards, feature importance, and test predictions for auditability. Regular backups and version control should protect scripts and documentation.

### 4.4 Project Schedule

The project schedule can be organized into phases: requirement study, data collection, data consolidation, preprocessing, exploratory analysis, model development, model validation, dashboard development, testing, documentation, and final review. The current repository suggests these phases were implemented iteratively because it contains raw data folders, combined data files, output datasets, visual images, model artifact versions, multiple dashboards, tests, and documentation notes. A realistic academic schedule is eight to ten weeks. Early weeks focus on understanding HR leave data and cleaning problems. Middle weeks focus on feature engineering and model experimentation. Later weeks focus on dashboard integration, testing, screenshots, black book preparation, and presentation material. If deployed in an organization, additional weeks should be added for security, user training, production hosting, and live data integration.

### 4.4.1 Project Task Set

The project task set includes collecting leave files, collecting employee master data, combining leave records, cleaning records, expanding date ranges, engineering calendar features, generating workforce features, aggregating department and leave type features, training candidate models, evaluating validation and test performance, saving the best model, generating forecasts, creating feature importance reports, building dashboards, adding staffing planning logic, testing core functions, preparing visual screenshots, and writing final documentation. Each task produces an artifact or decision. For example, data cleaning produces `clean_frame.csv`, expansion produces `expanded_frame.csv`, modeling produces `.pkl` files and metric CSVs, visualization produces images and dashboards, and documentation produces README files and this black book. This task set ensures that the project is traceable from raw data to final results.

### 4.4.2 Timeline Chart

| Week | Planned Work | Deliverable |
|---|---|---|
| Week 1 | Requirement study and data understanding | Problem statement and data inventory |
| Week 2 | Data collection and consolidation | Combined leave datasets |
| Week 3 | Data cleaning and leave expansion | Clean and expanded frames |
| Week 4 | Exploratory analysis and visual charts | Trend, department, leave type, and holiday plots |
| Week 5 | Feature engineering and baseline models | Model-ready dataset and baseline metrics |
| Week 6 | XGBoost, Random Forest, and Gradient Boosting comparison | Best model and validation results |
| Week 7 | Dashboard integration and forecast display | Streamlit and Flask dashboards |
| Week 8 | Testing, screenshots, and documentation | Test results, black book, and final report |

The timeline can be extended if production features such as authentication, database migration, automated ingestion, and deployment monitoring are added.

### 4.5 Team Organization (Structure)

The team organization can follow a small academic software project structure. A project coordinator manages milestones, documentation, and final submission. A data engineer handles leave file consolidation, cleaning, and feature preparation. A machine learning developer handles model training, validation, forecast generation, and artifact management. A dashboard developer handles Streamlit, Flask, Plotly charts, filters, and user interface flow. A tester validates data processing, forecasting behavior, dashboard functionality, and edge cases. A documentation lead prepares the black book, references, screenshots, appendices, and presentation content. In a small student team, one person may perform multiple roles, but the responsibilities should still be clearly assigned. For this repository, the implementation already reflects multiple roles because it includes data processing, modeling, dashboards, tests, generated visuals, and academic documentation.

---

## 05 SYSTEM DESIGN

### 5.1 Proposed System Architecture / Block Diagram

The proposed system architecture follows a layered pipeline. The first layer is the data source layer, containing raw CSV and Excel files from leave exports and employee master data. The second layer is preprocessing, where raw records are cleaned, normalized, expanded into daily leave facts, clipped to an as-of date, and saved as processed outputs. The third layer is feature engineering, where calendar, holiday, weekend, long weekend, monthly cycle, workforce headcount, department, leave type, lag, and rolling statistics are created. The fourth layer is model training and evaluation, where XGBoost, Gradient Boosting, Random Forest, and a naive baseline are compared. The fifth layer is artifact storage, including model pickle files, metadata, feature importance, model cards, metrics, and forecasts. The sixth layer is dashboard presentation through Streamlit, DuckDB analytics, and Flask views.

```text
Raw Leave Files + Employee Master
        |
        v
Data Cleaning and Leave Expansion
        |
        v
Feature Engineering and Aggregation
        |
        v
Model Training, Validation, and Selection
        |
        v
Saved Artifacts, Metrics, Forecast CSVs
        |
        v
Streamlit / SQL Analytics / Flask Dashboard
        |
        v
HR Planning, Risk Analysis, Staffing Decisions
```

### 5.2 Dataset / Database Design

The dataset design is currently file-based. Raw leave files are stored in `Employee_Leave_Data` and `relistofemployee`, consolidated data is stored in `Data/Combined_All_Leave_Data.csv`, and processed data is stored in `output`. The processed design includes raw frame, clean frame, expanded frame, full expanded frame, model dataframe, feature dataframe, department daily features, leave type daily features, leave type monthly features, department encoded features, and master workforce features. A production database design would include `Employee`, `Department`, `CostCentre`, `LeaveType`, `LeaveApplication`, `DailyLeaveFact`, `ForecastRun`, `ForecastOutput`, `ModelMetric`, and `ModelArtifact` entities. Relationships would connect employees to departments and cost centres, leave applications to leave types, daily facts to application dates, and forecasts to model runs. This logical database design is implemented conceptually through generated CSV files and can be migrated to SQL later.

### 5.3 Mathematical Model

The mathematical model estimates daily leave count as a supervised regression problem. Let `y_t` represent the actual leave count on date `t`, and let `X_t` represent engineered features for that date. The model learns a function `f` such that `y_hat_t = f(X_t)`, where `y_hat_t` is the predicted leave count. The feature vector includes calendar variables, holiday variables, workforce headcount, department indicators, leave type daily counts, lag values such as previous day and previous week leave, and rolling statistics such as seven-day and thirty-day averages. The selected XGBoost model combines many boosted decision trees, where each new tree corrects residual errors from previous trees. Model quality is measured using MAE, RMSE, R2, WAPE, MAPE, and SMAPE. Prediction intervals are derived from residual percentiles and absolute error analysis.

### 5.4 Entity Relationship Diagrams (If Applicable)

The conceptual ER model contains an `Employee` entity with employee ID, name or anonymized identifier, joining date, exit date, employee type, and active status. The `Department` entity stores department names, and the `CostCentre` entity stores cost centre codes or labels. The `LeaveType` entity stores leave categories such as casual leave, sick leave, earned privilege leave, special leave, and comp-off. The `LeaveApplication` entity records employee, leave type, start date, end date, planned or unplanned status, and leave reason. The `DailyLeaveFact` entity expands each application into daily rows for analytics. The `ForecastRun` entity stores model name, training period, test period, timestamp, and feature list. The `ForecastOutput` entity stores forecast date, predicted count, lower bound, and upper bound. These entities map naturally to the current file outputs.

### 5.5 UML Diagrams

The system can be represented using use case, activity, sequence, and component UML diagrams. In the use case diagram, actors include HR Analyst, Manager, System Admin, and ML Developer. HR Analyst views dashboards and exports summaries. Manager checks staffing risks and forecast confidence. System Admin manages files, deployment, and access. ML Developer retrains models and validates metrics. In the activity diagram, the flow begins with data loading, moves to cleaning, expansion, feature engineering, model prediction, dashboard rendering, and decision support. In the sequence diagram, the user opens the dashboard, the application loads artifacts, the model predicts or reads forecasts, charts are built, and results are returned. In the component diagram, data files, preprocessing functions, model artifacts, dashboard pages, and test scripts are separate components connected through file and function interfaces.

---

## 06 PROJECT IMPLEMENTATION

### 6.1 Overview of Project Modules

The implementation consists of several modules and folders. `streamlit_app.py` is the main dashboard and forecasting application. It contains functions for path management, holiday calendar generation, data cleaning, employee master preparation, leave expansion, feature alignment, model loading, metric calculation, forecast generation, explainability, staffing planning, risk summaries, and multiple dashboard views. `retrain_model.py` retrains models, compares candidates, applies walk-forward validation, creates model cards, saves versioned artifacts, and updates latest model files. `check_model.py` validates that model metadata exists and prints the latest model summary. `streamlit_sql_visualization.py` provides DuckDB-powered dashboard analysis over CSV data. `web_dashboard.py` provides a Flask and Plotly dashboard. The `artifacts` folder stores model files and metrics. The `output` folder stores generated processed datasets. The `Images` folder stores visual evidence for results and reporting.

### 6.2 Tools and Technologies Used

The project uses Python as the primary programming language. Pandas and NumPy are used for data manipulation and numerical processing. Scikit-learn provides model evaluation tools and candidate models such as Random Forest and Gradient Boosting. XGBoost provides the best-performing forecasting model in the current artifacts. Plotly, Matplotlib, and Seaborn support visualization. Streamlit provides the main interactive dashboard interface, while Flask provides an additional web dashboard option. DuckDB enables SQL-style analytics over CSV data. Joblib stores and loads trained model artifacts. Openpyxl supports Excel file reading. Holidays supports calendar feature creation. SHAP is used for model explanation when available, and TensorFlow is listed for possible deep learning experiments. Git supports project version control, and Markdown supports documentation such as this black book.

### 6.3 Algorithm Details

The project uses multiple algorithms working together rather than a single isolated method. The first algorithm cleans and expands leave records into daily facts. The second algorithm creates features that describe calendar behavior, workforce size, department history, leave type behavior, and previous leave counts. The third algorithm trains and compares regression models for daily leave forecasting. The fourth algorithm applies the saved model to generate future predictions and prediction intervals. The fifth algorithm transforms forecasts into dashboard insights, risk summaries, and staffing scenarios. This layered design is important because accurate forecasting depends on both data quality and model selection. The latest saved model selects XGBoost as the best model, but the system also compares Gradient Boosting, Random Forest, and a naive lag baseline to prove the selected model is not arbitrary.

### 6.3.1 Algorithm 1: Data Cleaning and Feature Engineering

Algorithm 1 begins by reading raw leave and employee master data. It standardizes column names, parses date fields, removes invalid records, handles missing values, and clips records according to the selected as-of date. It then expands multi-day leave ranges into one record per employee per leave date. After expansion, it aggregates daily counts and creates engineered features. Calendar features include day of week, month, day of month, week of year, quarter, weekend flag, month start and end flags, holiday flag, long weekend flag, and sine/cosine month encodings. History features include lag values and rolling means. Workforce features include active employee count, team member count, indirect count, local count, workforce share, joining count, exit count, and selected department headcounts. The output is a model-ready dataframe with aligned feature columns.

### 6.3.2 Algorithm 2: Model Training and Forecast Generation

Algorithm 2 trains and selects the forecasting model. The retraining process builds candidate models such as Random Forest, Gradient Boosting, XGBoost, and a naive lag baseline. It performs chronological splitting to avoid future data leakage and uses walk-forward validation for more reliable evaluation. Each model is evaluated using MAE, RMSE, MAPE, R2, WAPE, and SMAPE. The best model is chosen based on validation performance and stability. The selected model is saved with joblib, and metadata is saved with feature columns, training period, test period, forecast horizon, metrics, prediction interval values, model balance, and forecast results. For forecasting, the system generates future dates, builds the same engineered features, aligns columns to the trained model, predicts leave count, rounds outputs to practical integers, and adds lower and upper bounds from residual analysis.

### 6.3.3 Algorithm 3: Dashboard Intelligence and Staffing Planner

Algorithm 3 converts model outputs and historical facts into operational intelligence. It loads the latest model bundle and processed data, evaluates saved predictions where needed, and constructs dashboard tables and charts. It calculates daily leave summaries, department breakdowns, leave type spikes, cost centre risk scores, planned and unplanned distributions, special leave and comp-off patterns, and leave reason summaries. The staffing planner takes predicted leave, total workforce, required present workforce, known absent employees, and safety buffer values as inputs. It then estimates available staff, shortage or surplus, and planning status. This algorithm turns technical predictions into decisions that managers can act on. Instead of only saying a model predicts 36 employees on leave, the system helps decide whether that count creates a staffing risk for a selected workforce requirement.

---

## 07 SOFTWARE TESTING

### 7.1 Type of Testing

The project supports several testing types. Unit testing is represented by test files under `employee-leave-forecasting-system/tests`, including tests for data processing, forecasting, and models. Functional testing checks whether the dashboard loads data, filters date ranges, displays charts, and handles empty selections gracefully. Integration testing checks whether raw data can flow through cleaning, feature generation, model loading, prediction, and dashboard visualization. Regression testing is needed whenever data cleaning logic or model features change, because a small feature mismatch can break prediction. Model validation testing checks accuracy metrics on validation and test periods. Artifact testing is implemented by `check_model.py`, which loads the metadata file and verifies expected keys such as best model name, training end date, test metrics, and feature columns. User acceptance testing should involve HR users reviewing dashboard clarity and planning usefulness.

### 7.2 Test Cases and Test Results

The following table summarizes the practical testing condition of the current repository. A model artifact check was executed while preparing this report, and it successfully loaded the latest metadata. The result confirmed that the latest model is XGBoost, the training period is 2022-02-18 to 2025-08-26, the test period is 2025-08-27 to 2026-04-10, the model uses 48 engineered features, and the generated forecast horizon is 60 days. The repository also contains automated test files that should be run as part of final verification before submission. The project is strongest in model artifact validation and dashboard logic, while production-grade tests for authentication and database security remain future work.

| Test Case ID | Test Scenario | Expected Result | Current Result |
|---|---|---|---|
| TC-01 | Load latest metadata using `check_model.py` | Metadata loads and prints model summary | Passed; latest model is XGBoost |
| TC-02 | Validate required metadata keys | Required keys exist or warnings are shown | Passed; key summary printed successfully |
| TC-03 | Check model performance metrics | Metrics are available for test period | Passed; WAPE about 3.77%, R2 about 0.9981 |
| TC-04 | Load forecast CSV | Future predictions are available | Passed; 60-day forecast file exists |
| TC-05 | Dashboard data filtering | Date filters should return matching summaries | Implemented in Streamlit and SQL dashboards |
| TC-06 | Empty data handling | Dashboard should show warning or info message | Implemented through Streamlit warnings and info messages |
| TC-07 | Feature alignment | Prediction features match trained model columns | Implemented using alignment helper functions |

Full production testing should additionally include authentication tests, database integration tests, stress tests with larger data, privacy tests, browser compatibility checks, deployment monitoring checks, and user acceptance sessions with actual HR stakeholders.

---

## 08 RESULTS

### 8.1 Outcomes

The project successfully produces an end-to-end leave forecasting and analytics system. It consolidates historical leave data, generates cleaned and expanded datasets, creates model-ready features, trains and evaluates forecasting models, saves model artifacts, produces future forecasts, and displays insights through multiple dashboards. The latest model artifact identifies XGBoost as the best model and uses 48 engineered features. The generated 60-day forecast begins on 2026-04-11 and provides predicted leave count with lower and upper bounds. The dashboard supports operational analysis of daily leave, peak days, cost centre risk, leave type patterns, special leave, comp-off, planned versus unplanned absence, leave reasons, and staffing scenarios. The project also creates academic result images such as trends, distributions, feature importance, SHAP summary, actual versus predicted charts, and residual diagnostics.

### 8.2 Result Analysis and Validations

The latest model metrics indicate strong predictive performance. The test period from 2025-08-27 to 2026-04-10 produced MAE of approximately 7.85, RMSE of approximately 14.55, R2 of approximately 0.9981, WAPE of approximately 3.77%, MAPE of approximately 21.28%, and SMAPE of approximately 13.60%. WAPE is especially useful because it measures weighted absolute percentage error over total actual leave volume, making it more stable for operational forecasting. The model comparison in the metadata shows XGBoost outperforming Gradient Boosting, Random Forest, and the naive lag baseline. Feature importance shows department and leave type variables as important contributors, especially `dept_car_production_pune`, `leave_type_daily_special_leave_not_call_on_duty`, `dept_quality_management`, and `department_avg_leave`. Residual and actual-versus-predicted visualizations in the `Images` folder support validation by showing how closely predictions match historical test outcomes.

### 8.2 Screen Shots

The project includes generated screenshot-ready result images in the `Images` folder. These images should be inserted into the final formatted black book or presentation as visual evidence. Available visuals include `01_daily_leave_trend.png`, `02_monthly_leave_distribution.png`, `03_average_leave_by_weekday.png`, `04_top_department_leave_contribution.png`, `05_leave_type_distribution.png`, `06_festival_holiday_leave_spikes.png`, `09_xgboost_dnn_forecast_comparison.png`, `10_feature_importance.png`, `11_shap_summary.png`, `12_actual_vs_predicted_holdout.png`, `13_residual_distribution.png`, `14_residuals_vs_predicted.png`, `15_residuals_over_time.png`, and `correlation_heatmap.png`. The dashboard screenshots should also include the Streamlit forecast confidence page, daily leave dashboard, cost centre analysis page, planned versus unplanned page, special leave page, leave reason page, and staffing planner. These screenshots demonstrate that the system is not limited to backend modeling but includes usable visual analytics.

---

## 09 CONCLUSIONS

### 9.1 Conclusions

The Leave Management and Forecasting System demonstrates how HR leave data can be transformed into practical forecasting and planning intelligence. The project begins with raw CSV and Excel leave records, cleans and expands them into daily facts, engineers meaningful calendar and workforce features, trains multiple forecasting models, selects XGBoost as the best model, validates the model using standard metrics, and presents results through dashboards. The latest model shows strong test performance with WAPE around 3.77% and R2 around 0.9981, indicating that the model captures historical leave patterns effectively. The dashboards make the output useful for HR and operations teams by showing daily leave forecasts, confidence intervals, leave type analysis, cost centre risk, and staffing scenarios. Overall, the project satisfies the academic goal of building an end-to-end machine learning system with implementation, testing, visualization, and documentation.

### 9.2 Future Scope

Future scope includes database integration, secure login, role-based access, automated data ingestion from HRMS, scheduled retraining, model drift monitoring, notification alerts, cloud deployment, API integration, and improved modular code structure. The current file-based approach works well for a project prototype, but production use should migrate core data to a relational database or data warehouse. The dashboard should include user authentication and permissions to protect employee information. Forecast alerts can be sent to managers when predicted leave exceeds a staffing threshold. Drift monitoring can compare actual leave against prediction intervals and trigger retraining when accuracy declines. Additional models such as Prophet, LightGBM, temporal deep learning, or hybrid statistical-machine learning models can be evaluated. The system can also add individual anonymization, audit logs, and automated report generation in PDF format.

### 9.3 Applications

The system has applications in workforce planning, HR analytics, production staffing, leave policy analysis, department capacity management, risk identification, and management reporting. HR teams can use it to understand when leave demand is likely to rise and which leave types contribute most. Managers can use it to prepare staffing alternatives before high-absence days. Operations teams can use cost centre risk analysis to identify departments that may need backup support. The organization can use leave pattern analysis to evaluate policy impact, seasonal trends, holiday effects, and planned versus unplanned leave behavior. Academic users can use the project as an example of an end-to-end ML lifecycle, including data preparation, feature engineering, model comparison, artifact management, visualization, and documentation. With security enhancements, the same design can be adapted for real enterprise HR decision support.

---

## REFERENCES

[1] T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," in Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, San Francisco, CA, USA, 2016, pp. 785-794.  

[2] L. Breiman, "Random Forests," Machine Learning, vol. 45, no. 1, pp. 5-32, 2001.  

[3] J. H. Friedman, "Greedy Function Approximation: A Gradient Boosting Machine," Annals of Statistics, vol. 29, no. 5, pp. 1189-1232, 2001.  

[4] S. M. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," in Advances in Neural Information Processing Systems, Long Beach, CA, USA, 2017, pp. 4765-4774.  

[5] W. McKinney, "Data Structures for Statistical Computing in Python," in Proceedings of the 9th Python in Science Conference, Austin, TX, USA, 2010, pp. 56-61.  

[6] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," Journal of Machine Learning Research, vol. 12, pp. 2825-2830, 2011.  

[7] Plotly Technologies Inc., "Collaborative Data Science," Plotly, Montreal, QC, Canada, 2015.  

[8] Streamlit Inc., "Streamlit Documentation: Build Data Apps in Python," Streamlit, 2026.  

[9] DuckDB Foundation, "DuckDB: An In-Process SQL OLAP Database Management System," DuckDB Documentation, 2026.  

[10] Project Repository, "Leave Management and Forecasting System Source Files, Datasets, Artifacts, Tests, and Generated Visualizations," Local academic project workspace, 2026.

---

## APPENDIX A: DETAILS OF PAPER PUBLICATION, CONFERENCE / JOURNAL, REVIEWER COMMENTS, CERTIFICATE, AND PAPER

At the time of preparing this black book, no final conference paper, journal publication certificate, reviewer comments, patent filing document, or project event participation certificate was found as an implemented repository artifact. Therefore, this appendix is prepared as a structured placeholder to be completed after academic submission activities. If a paper is drafted, it should include title, abstract, keywords, introduction, related work, methodology, implementation, results, conclusion, references, and plagiarism declaration. If submitted to a conference or journal, this appendix should attach the name of the event, submission ID, acceptance or review status, reviewer comments, presentation certificate, participation certificate, and online publication link if available. If no publication is completed, the draft paper and internal project presentation certificate may be attached instead. This satisfies the required appendix structure while clearly identifying pending external documents.

## APPENDIX B: PLAGIARISM REPORT OF PROJECT REPORT

The plagiarism report is an external academic compliance document and is not generated by the current software system. It should be produced using the institute-approved plagiarism checking tool after the final report is formatted, screenshots are inserted, and all appendices are attached. The final plagiarism report should include similarity percentage, source match summary, excluded bibliography setting if applicable, student details, project title, guide approval, and date of checking. This appendix should attach the official plagiarism report PDF or screenshot. Until the official report is generated, this section acts as a placeholder and reminder. The content of this `Black_Book.md` is written specifically for the current Leave Management and Forecasting System repository and references actual project files, scripts, outputs, and model artifacts to reduce generic copied content.

