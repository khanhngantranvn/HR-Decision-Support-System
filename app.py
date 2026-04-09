import streamlit as st

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px


from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier

from lifelines import KaplanMeierFitter, CoxPHFitter

from scipy.optimize import linprog

from imblearn.combine import SMOTEENN

from imblearn.under_sampling import EditedNearestNeighbours


st.set_page_config(page_title="HR SYSTEM", layout="wide")


st.markdown("""

    <style>

    .main { background-color: #F8FAFC; }

    .main-title { font-size:28px !important; font-weight: 700 !important; color: #1E3A8A; text-align: center; margin-bottom: 20px; }

    .section-header { font-size:20px !important; font-weight: 600 !important; color: #334155; border-bottom: 2px solid #E2E8F0; padding-bottom: 8px; margin-top: 25px; }

    .stMetric { background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }

    </style>

    """, unsafe_allow_html=True)



def feature_engineering(df_input):

    df = df_input.copy()

    if 'Attrition' in df.columns and df['Attrition'].dtype == 'object':

        df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)


   

    if 'Years_Since_Last_Promotion' in df.columns and 'Years_at_Company' in df.columns:

        df['Stagnation_Rate'] = df['Years_Since_Last_Promotion'] / (df['Years_at_Company'] + 1)

   

    features_to_drop = ['Years_in_Current_Role', 'Absenteeism']

    df = df.drop(columns=[c for c in features_to_drop if c in df.columns], errors='ignore')

    return df


@st.cache_data

def load_data():

    df_raw = pd.read_csv("employee_attrition_dataset_10000 (1).csv")

    df_processed = feature_engineering(df_raw)

    return df_processed


@st.cache_resource

def train_models(df):

    df_train = df.copy()


    if df_train['Attrition'].dtype == 'object' or df_train['Attrition'].dtype == 'string':

        df_train['Attrition'] = df_train['Attrition'].map({'Yes': 1, 'No': 0})

   

    X = df_train.drop(columns=['Attrition', 'Employee_ID'], errors='ignore')

    y = df_train['Attrition'].fillna(0).astype(int)

   

    X_encoded = pd.get_dummies(X, drop_first=True)

    feature_names = X_encoded.columns.tolist()

    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()

   

    smnn = SMOTEENN(random_state=42, enn=EditedNearestNeighbours(n_neighbors=5))

    X_resampled, y_resampled = smnn.fit_resample(X_encoded, y)

   

    scaler = MinMaxScaler()

    X_resampled_scaled = X_resampled.copy()

    X_resampled_scaled[numeric_cols] = scaler.fit_transform(X_resampled[numeric_cols])

   

    xgb = XGBClassifier(

        n_estimators=200,

        max_depth=5,

        learning_rate=0.05,

        subsample=0.8,

        colsample_bytree=0.8,

        random_state=42,

        eval_metric='logloss'

    )

    xgb.fit(X_resampled_scaled, y_resampled)

   

    survival_df = X_encoded.copy()

    survival_df['Years_at_Company'] = df['Years_at_Company']

    survival_df['Attrition'] = y

    cph = CoxPHFitter(penalizer=0.1)

    cph.fit(survival_df, duration_col='Years_at_Company', event_col='Attrition')

   

    return xgb, cph, scaler, feature_names, numeric_cols


df = load_data()

xgb_model, cph_model, data_scaler, TRAIN_FEATURES, NUMERIC_COLS = train_models(df)


if 'accumulated_employees' not in st.session_state:

    st.session_state.accumulated_employees = []


st.sidebar.title("HR Analytics Panel")

page = st.sidebar.radio("Điều hướng", ["Business Overview", "Survival Analysis", "Budget Optimization"])


if page == "Business Overview":

    st.markdown('<p class="main-title">Tổng quan doanh nghiệp</p>', unsafe_allow_html=True)

   

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Employees", f"{len(df):,}")

    m2.metric("Attrition Rate", f"{df['Attrition'].mean():.1%}")

    m3.metric("Average Monthly Income", f"{df['Monthly_Income'].mean():,.0f} VNĐ")

    m4.metric("Average Working Years", f"{df['Years_at_Company'].mean():.1f} years")


    col1, col2 = st.columns([6, 4])

   

    with col1:

        st.markdown('<p class="section-header">Kaplan Meier by Department</p>', unsafe_allow_html=True)

        fig_km, ax = plt.subplots(figsize=(10, 6))

        kmf = KaplanMeierFitter()

        for dept in df['Department'].unique():

            mask = df['Department'] == dept

            kmf.fit(df.loc[mask, 'Years_at_Company'], df.loc[mask, 'Attrition'], label=dept)

            kmf.plot_survival_function(ax=ax)

        plt.grid(axis='y', alpha=0.3)

        st.pyplot(fig_km)


    with col2:

        st.markdown('<p class="section-header">Churn Factors by CoxPH</p>', unsafe_allow_html=True)

        summary_df = cph_model.params_.to_frame()

        summary_df.columns = ['Coefficient']

        top_factors = pd.concat([summary_df.sort_values(by='Coefficient').head(5), summary_df.sort_values(by='Coefficient').tail(5)])

        fig_cox = px.bar(top_factors, x='Coefficient', y=top_factors.index, orientation='h',

                         color='Coefficient', color_continuous_scale='RdYlGn_r')

        st.plotly_chart(fig_cox, use_container_width=True)


    st.markdown('<p class="section-header">Feature Importance by XGBoost</p>', unsafe_allow_html=True)

    feat_imp = pd.DataFrame({'Feature': TRAIN_FEATURES, 'Importance': xgb_model.feature_importances_}).sort_values('Importance', ascending=False).head(10)

    fig_imp = px.bar(feat_imp, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Viridis')

    st.plotly_chart(fig_imp, use_container_width=True)


elif page == "Survival Analysis":

    st.markdown('<p class="main-title">Survival Analysis</p>', unsafe_allow_html=True)


    with st.expander("Employee Information", expanded=True):

        c1, c2, c3 = st.columns(3)

        with c1:

            age = st.number_input("Age", 18, 65, 30)

            gender = st.selectbox("Gender", ["Male", "Female"])

            marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

            dept = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources", "IT", "Marketing"])

            dist = st.number_input("Working Distance(km)", 1, 100, 10)

        with c2:

            role = st.selectbox("Job Role", df['Job_Role'].unique() if 'Job_Role' in df.columns else ["Manager", "Developer", "Sales Executive"])

            level = st.slider("Job Level (1-5)", 1, 5, 1)

            income = st.number_input("Monthly Income", 1000, 100000, 15000)

            overtime = st.selectbox("Overtime", ["Yes", "No"])

            job_inv = st.slider("Job Involvement (1-5)", 1, 5, 3)

        with c3:

            years_at_co = st.number_input("Years at Company", 0, 40, 3)

            years_promo = st.number_input("Years Since Last Promotion", 0, 20, 1)

            satisfaction = st.slider("Job Satisfaction (1-5)", 1, 5, 3)

            perf = st.slider("Performance Rating (1-5)", 1, 5, 3)

            training = st.number_input("Training Hours Last Year", 0, 100, 20)


    if st.button("Predict"):

        input_raw = pd.DataFrame([{

            'Age': age, 'Gender': gender, 'Marital_Status': marital, 'Department': dept,

            'Job_Role': role, 'Job_Level': level, 'Monthly_Income': income, 'Overtime': overtime,

            'Years_at_Company': years_at_co, 'Years_Since_Last_Promotion': years_promo,

            'Job_Satisfaction': satisfaction, 'Performance_Rating': perf,

            'Training_Hours_Last_Year': training, 'Distance_From_Home': dist, 'Job_Involvement': job_inv

        }])

       

        input_fe = feature_engineering(input_raw)

        input_enc = pd.get_dummies(input_fe).reindex(columns=TRAIN_FEATURES, fill_value=0)

       

        input_scaled = input_enc.copy()

        existing_numeric = [c for c in NUMERIC_COLS if c in input_scaled.columns]

        input_scaled[existing_numeric] = data_scaler.transform(input_enc[existing_numeric])

       

        prob_xgb = xgb_model.predict_proba(input_scaled)[0, 1]

        hazard_score = cph_model.predict_partial_hazard(input_enc).values[0]

       

        st.session_state.accumulated_employees.append(input_raw.to_dict('records')[0])

        st.success(f"Employee added (Total {len(st.session_state.accumulated_employees)} new employees).")


        st.markdown("---")

        res1, res2 = st.columns(2)

       

        with res1:

            st.subheader("Risk Score")

            st.metric("Attrition Rate", f"{prob_xgb:.1%}")

            if prob_xgb > 0.7: st.error("High risk of attrition")

            elif prob_xgb > 0.3: st.warning("Medium risk of attrition")

            else: st.success("Low risk of attrition")

           

            feat_imp_local = pd.DataFrame({'feature': TRAIN_FEATURES, 'weight': xgb_model.feature_importances_}).sort_values('weight', ascending=False).head(5)

            fig_local = px.bar(feat_imp_local, x='weight', y='feature', orientation='h', title="Features")

            st.plotly_chart(fig_local, use_container_width=True)


        with res2:

            st.subheader("Attrition Prediction")

            surv_func = cph_model.predict_survival_function(input_enc)

            fig_s, ax_s = plt.subplots(figsize=(8, 5))

            ax_s.plot(surv_func.index, surv_func.values, color='#1E3A8A', lw=3)

            ax_s.set_xlabel("Years")

            ax_s.set_ylabel("Risk")

            st.pyplot(fig_s)


elif page == "Budget Optimization":

    st.markdown('<p class="main-title">Optimization task</p>', unsafe_allow_html=True)

   

    opt_df = df.copy()

    if st.session_state.accumulated_employees:

        new_data_df = pd.DataFrame(st.session_state.accumulated_employees)

        new_data_processed = feature_engineering(new_data_df)

        opt_df = pd.concat([opt_df, new_data_processed], ignore_index=True)

        st.info(f"Analyzing...")

   

    X_opt = pd.get_dummies(opt_df.drop(columns=['Attrition', 'Employee_ID'], errors='ignore')).reindex(columns=TRAIN_FEATURES, fill_value=0)

    X_opt_scaled = X_opt.copy()

    X_opt_scaled[NUMERIC_COLS] = data_scaler.transform(X_opt[NUMERIC_COLS])


    if st.button("Optimize"):

        # 1. Tính Risk Score lai (probs * (1 + hazard_norm))

        probs = xgb_model.predict_proba(X_opt_scaled)[:, 1]

        hazards = cph_model.predict_partial_hazard(X_opt)

        h_min, h_max = hazards.min(), hazards.max()

        hazards_norm = (hazards - h_min) / (h_max - h_min)

        opt_df['Risk_Score'] = probs * (1 + hazards_norm)


        def map_replacement(level):

            if level <= 1: return 3

            if level == 2: return 4

            if level == 3: return 5

            return 6


        RETENTION_COST_FACTOR = 0.20

        opt_df['Replacement_Months'] = opt_df['Job_Level'].apply(map_replacement)

        opt_df['Replacement_Cost'] = opt_df['Monthly_Income'] * opt_df['Replacement_Months']

        opt_df['Retention_Cost'] = opt_df['Monthly_Income'] * 12 * RETENTION_COST_FACTOR

       

        opt_df['Expected_Gain'] = (opt_df['Risk_Score'] * opt_df['Replacement_Cost']) - opt_df['Retention_Cost']


        candidates = opt_df[opt_df['Expected_Gain'] > 0].copy()


        if not candidates.empty:

            c = -candidates['Expected_Gain'].values

            A_ub = [candidates['Retention_Cost'].values, np.ones(len(candidates))]

           

            total_budget = opt_df['Retention_Cost'].sum() * 0.20

            b_ub = [total_budget, 100]


            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(0, 1), method='highs')


            if res.success:

                candidates['Decision'] = (res.x > 0.5).astype(int)

                final_list = candidates[candidates['Decision'] == 1].sort_values('Expected_Gain', ascending=False)

               

                st.success(f"Number of employees: {len(final_list)}")

               

                c1, c2, c3 = st.columns(3)

                total_loss_avoided = (final_list['Risk_Score'] * final_list['Replacement_Cost']).sum()

                investment = final_list['Retention_Cost'].sum()

               

                c1.metric("Expected Loss", f"{total_loss_avoided:,.0f} VNĐ", help = "Ước tính tổng chi phí thay thế đã tránh được nhờ giữ chân những nhân sự này.")

                c2.metric("Retention Cost", f"{investment:,.0f} VNĐ", help = "Tổng chi phí giữ chân dự kiến cho những nhân sự được đề xuất.")

                c3.metric("ROI", f"{total_loss_avoided/investment:.2f}", help = "Tỷ lệ lợi ích kỳ vọng trên chi phí đầu tư.")


                st.dataframe(

                    final_list[['Department', 'Job_Role', 'Risk_Score', 'Expected_Gain', 'Retention_Cost']],

                    column_config={

                        "Risk_Score": st.column_config.NumberColumn("risk score", format="%.2f"),

                        "Expected_Gain": st.column_config.ProgressColumn("gain", format="%d VNĐ", min_value=0, max_value=float(candidates['Expected_Gain'].max()))

                    },

                    use_container_width=True

                )

            else:

                st.error("cannot optimize")

        else:

            st.warning("no employee to retain") 