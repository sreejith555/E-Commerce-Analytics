import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Smart E-Commerce Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1a1a2e;
        border-left: 4px solid #e94560;
        padding-left: 12px;
        margin: 20px 0 12px 0;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    customers    = pd.read_csv('data/Customers.csv')
    products     = pd.read_csv('data/Products.csv')
    transactions = pd.read_csv('data/Transactions.csv')
    transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])
    return customers, products, transactions

@st.cache_resource
def load_models():
    models = {}
    model_files = {
        'kmeans':         'models/kmeans_model.pkl',
        'scaler_cluster': 'models/scaler_cluster.pkl',
        'random_forest':  'models/random_forest.pkl',
        'scaler_churn':   'models/scaler_churn.pkl',
        'label_encoders': 'models/label_encoders.pkl',
        'churn_features': 'models/churn_features.pkl',
        'product_pop':    'models/product_popularity.pkl',
        'user_sim':       'models/user_similarity.pkl',
        'item_sim':       'models/item_similarity.pkl',
    }
    for key, path in model_files.items():
        if os.path.exists(path):
            with open(path, 'rb') as f:
                models[key] = pickle.load(f)
    return models

customers, products, transactions = load_data()
models = load_models()

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def fmt_inr(value):
    if value >= 1e7:
        return f"₹{value/1e7:.2f}Cr"
    elif value >= 1e5:
        return f"₹{value/1e5:.1f}L"
    return f"₹{value:,.0f}"

def compute_rfm_scores(df):
    """Compute R/F/M quintile scores from raw columns."""
    out = df.copy()
    out['R_score'] = pd.qcut(out['recency'], q=5, labels=[5,4,3,2,1]).astype(int)
    out['F_score'] = pd.qcut(out['frequency'].rank(method='first'), q=5, labels=[1,2,3,4,5]).astype(int)
    out['M_score'] = pd.qcut(out['monetary_value'].rank(method='first'), q=5, labels=[1,2,3,4,5]).astype(int)
    out['RFM_Score'] = out['R_score'] + out['F_score'] + out['M_score']
    return out

def rfm_segment(score):
    if score >= 13: return 'Champions'
    elif score >= 10: return 'Loyal Customers'
    elif score >= 7:  return 'Potential Loyalists'
    elif score >= 5:  return 'At Risk'
    return 'Lost'

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/000000/shopping-cart.png", width=60)
st.sidebar.title("🛒 E-Commerce Analytics")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate to", [
    "📊 Overview Dashboard",
    "🎯 Customer Segmentation",
    "⚠️ Churn Prediction",
    "📈 Sales Forecasting",
    "💡 Product Recommendations",
])

st.sidebar.markdown("---")
st.sidebar.caption("Smart E-Commerce Capstone Project")
st.sidebar.caption(f"👥 {len(customers):,} customers  |  🛍️ {len(transactions):,} transactions")

# ═══════════════════════════════════════════════════════════
# PAGE 1 — Overview Dashboard
# ═══════════════════════════════════════════════════════════
if page == "📊 Overview Dashboard":
    st.title("📊 Smart E-Commerce Analytics — Overview")
    st.markdown("**Business Intelligence Dashboard** | Insights from customer & transaction data")
    st.markdown("---")

    # KPIs
    total_revenue = transactions['total_amount'].sum()
    avg_order     = transactions['total_amount'].mean()
    churn_rate    = customers['churn'].mean() * 100
    prime_pct     = (customers['loyalty_tier'] == 'Prime').mean() * 100
    avg_nps       = customers['net_promoter_score'].mean()
    total_orders  = len(transactions)

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("💰 Total Revenue",  fmt_inr(total_revenue))
    c2.metric("🛍️ Total Orders",   f"{total_orders:,}")
    c3.metric("📦 Avg Order Value", fmt_inr(avg_order))
    c4.metric("⚠️ Churn Rate",     f"{churn_rate:.1f}%")
    c5.metric("⭐ Prime Members",  f"{prime_pct:.0f}%")
    c6.metric("😊 Avg NPS",        f"{avg_nps:.1f}")
    st.markdown("---")

    # Monthly Revenue Trend
    st.markdown('<div class="section-header">Monthly Revenue Trend</div>', unsafe_allow_html=True)
    monthly = (transactions
        .groupby(transactions['transaction_date'].dt.to_period('M'))['total_amount']
        .sum().reset_index())
    monthly.columns = ['month','revenue']
    monthly['month_dt'] = monthly['month'].dt.to_timestamp()

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(monthly['month_dt'], monthly['revenue'], color='royalblue', linewidth=1.5)
    ax.fill_between(monthly['month_dt'], monthly['revenue'], alpha=0.12, color='royalblue')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'₹{x/1e5:.0f}L'))
    ax.set_title('Monthly Revenue (2023–2025)', fontweight='bold')
    ax.set_xlabel('Month'); ax.set_ylabel('Revenue')
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # Row 2
    ca, cb, cc = st.columns(3)
    with ca:
        st.markdown('<div class="section-header">Revenue by Category</div>', unsafe_allow_html=True)
        cat_rev = transactions.groupby('category')['total_amount'].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(5,4))
        ax.bar(cat_rev.index, cat_rev.values, color=sns.color_palette('husl', len(cat_rev)))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'₹{x/1e5:.0f}L'))
        ax.tick_params(axis='x', rotation=15)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with cb:
        st.markdown('<div class="section-header">Payment Methods</div>', unsafe_allow_html=True)
        pay = transactions['payment_method'].value_counts()
        fig, ax = plt.subplots(figsize=(5,4))
        ax.pie(pay, labels=pay.index, autopct='%1.1f%%',
               colors=sns.color_palette('pastel'), startangle=90)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with cc:
        st.markdown('<div class="section-header">Top 8 States</div>', unsafe_allow_html=True)
        sc = customers['state'].value_counts().head(8)
        fig, ax = plt.subplots(figsize=(5,4))
        ax.barh(sc.index[::-1], sc.values[::-1], color='steelblue')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Row 3 — Demographics
    st.markdown('<div class="section-header">Customer Demographics</div>', unsafe_allow_html=True)
    cd, ce, cf = st.columns(3)
    with cd:
        fig, ax = plt.subplots(figsize=(5,3))
        ax.hist(customers['age'], bins=20, color='mediumpurple', edgecolor='white')
        ax.set_title('Age Distribution')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with ce:
        top_cat = customers['top_category'].value_counts()
        fig, ax = plt.subplots(figsize=(5,3))
        ax.barh(top_cat.index, top_cat.values, color=sns.color_palette('Set2', len(top_cat)))
        ax.set_title('Preferred Category')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with cf:
        churn_data = customers['churn'].map({0:'Active',1:'Churned'}).value_counts()
        fig, ax = plt.subplots(figsize=(5,3))
        ax.pie(churn_data, labels=churn_data.index, autopct='%1.1f%%',
               colors=['#66bb66','#ee6666'], explode=[0,0.05], startangle=90)
        ax.set_title('Churn vs Active')
        plt.tight_layout(); st.pyplot(fig); plt.close()


# ═══════════════════════════════════════════════════════════
# PAGE 2 — Customer Segmentation
# ═══════════════════════════════════════════════════════════
elif page == "🎯 Customer Segmentation":
    st.title("🎯 Customer Segmentation")
    st.markdown("RFM Score Computation + K-Means Clustering")
    st.markdown("---")

    # Compute RFM from raw columns
    rfm = compute_rfm_scores(customers[['customer_id','recency','frequency','monetary_value']])
    rfm['RFM_Segment'] = rfm['RFM_Score'].apply(rfm_segment)

    seg_counts = rfm['RFM_Segment'].value_counts()
    emoji_map = {'Champions':'🏆','Loyal Customers':'💛','Potential Loyalists':'💚','At Risk':'⚠️','Lost':'❌'}

    cols = st.columns(len(seg_counts))
    for col, (seg, cnt) in zip(cols, seg_counts.items()):
        col.metric(f"{emoji_map.get(seg,'')} {seg}", f"{cnt:,}", f"{cnt/len(rfm)*100:.1f}%")
    st.markdown("---")

    cl, cr = st.columns(2)
    with cl:
        st.markdown("#### RFM Segment Distribution")
        colors_seg = ['#2ecc71','#3498db','#9b59b6','#e67e22','#e74c3c']
        fig, ax = plt.subplots(figsize=(7,4))
        ax.bar(seg_counts.index, seg_counts.values,
               color=colors_seg[:len(seg_counts)], edgecolor='white')
        for i,(idx,val) in enumerate(seg_counts.items()):
            ax.text(i, val+20, f'{val:,}\n({val/len(rfm)*100:.1f}%)', ha='center', fontsize=9)
        ax.set_ylabel('Customers')
        ax.tick_params(axis='x', rotation=15)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with cr:
        st.markdown("#### RFM Score Distribution")
        fig, ax = plt.subplots(figsize=(7,4))
        rfm['RFM_Score'].hist(bins=12, ax=ax, color='steelblue', edgecolor='white')
        ax.set_xlabel('Total RFM Score'); ax.set_ylabel('Customers')
        ax.set_title('Distribution of RFM Scores')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("#### Segment Profiles — Average RFM Values")
    profile = rfm.groupby('RFM_Segment')[['recency','frequency','monetary_value','RFM_Score']].mean().round(1)
    profile.columns = ['Avg Recency (days)','Avg Frequency','Avg Monetary (₹)','Avg RFM Score']
    st.dataframe(profile.sort_values('Avg RFM Score', ascending=False), use_container_width=True)

    # K-Means
    if 'kmeans' in models and 'scaler_cluster' in models:
        st.markdown("---")
        st.markdown("#### K-Means Cluster Visualisation")
        cluster_features = ['recency','frequency','monetary_value']
        X_scaled = models['scaler_cluster'].transform(rfm[cluster_features])
        rfm['Cluster'] = models['kmeans'].predict(X_scaled)

        palette = sns.color_palette('husl', rfm['Cluster'].nunique())
        fig, axes = plt.subplots(1, 2, figsize=(14,5))
        for cluster in sorted(rfm['Cluster'].unique()):
            mask = rfm['Cluster'] == cluster
            axes[0].scatter(rfm.loc[mask,'recency'], rfm.loc[mask,'monetary_value'],
                            alpha=0.4, s=15, label=f'Cluster {cluster}', color=palette[cluster])
            axes[1].scatter(rfm.loc[mask,'frequency'], rfm.loc[mask,'monetary_value'],
                            alpha=0.4, s=15, label=f'Cluster {cluster}', color=palette[cluster])
        for ax, xl, t in [(axes[0],'Recency (days)','Recency vs Monetary'),
                          (axes[1],'Frequency','Frequency vs Monetary')]:
            ax.set_xlabel(xl); ax.set_ylabel('Monetary Value (₹)'); ax.set_title(t); ax.legend()
        plt.tight_layout(); st.pyplot(fig); plt.close()
    else:
        st.info("ℹ️ Run **02_Segmentation.ipynb** to generate K-Means model files.")


# ═══════════════════════════════════════════════════════════
# PAGE 3 — Churn Prediction
# ═══════════════════════════════════════════════════════════
elif page == "⚠️ Churn Prediction":
    st.title("⚠️ Churn Prediction")
    st.markdown("Predict customer churn probability using Random Forest")
    st.markdown("---")

    churned = customers['churn'].sum()
    active  = len(customers) - churned
    churn_rate = customers['churn'].mean() * 100

    c1,c2,c3 = st.columns(3)
    c1.metric("Total Customers",   f"{len(customers):,}")
    c2.metric("Active Customers",  f"{active:,}")
    c3.metric("Churned Customers", f"{churned:,}", f"{churn_rate:.1f}% of total", delta_color="inverse")
    st.markdown("---")

    cl, cr = st.columns(2)
    with cl:
        st.markdown("#### Churn vs Recency")
        fig, ax = plt.subplots(figsize=(6,4))
        for label, color in [('Active','#66bb66'),('Churned','#ee6666')]:
            mask = customers['churn'] == (0 if label=='Active' else 1)
            ax.hist(customers.loc[mask,'recency'].clip(upper=700), bins=25,
                    alpha=0.65, label=label, color=color, density=True)
        ax.set_xlabel('Recency (days)'); ax.set_ylabel('Density'); ax.legend()
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with cr:
        st.markdown("#### Churn vs NPS")
        fig, ax = plt.subplots(figsize=(6,4))
        for label, color in [('Active','#66bb66'),('Churned','#ee6666')]:
            mask = customers['churn'] == (0 if label=='Active' else 1)
            ax.hist(customers.loc[mask,'net_promoter_score'], bins=20,
                    alpha=0.65, label=label, color=color, density=True)
        ax.set_xlabel('Net Promoter Score'); ax.set_ylabel('Density'); ax.legend()
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown("#### 🔍 Individual Customer Churn Risk Predictor")

    if all(k in models for k in ['random_forest','scaler_churn','label_encoders','churn_features']):
        ca, cb, cc = st.columns(3)
        with ca:
            age           = st.slider("Age", 18, 70, 35)
            session_freq  = st.slider("Session Frequency (monthly)", 1, 30, 10)
            avg_session   = st.slider("Avg Session Duration (min)", 1.0, 30.0, 8.0)
            cart_abandon  = st.slider("Cart Abandonment Rate", 0.0, 1.0, 0.3)
            returns_ratio = st.slider("Returns Ratio", 0.0, 0.5, 0.1)
        with cb:
            avg_order_val  = st.number_input("Avg Order Value (₹)", 200, 50000, 2000)
            frequency      = st.slider("Purchase Frequency", 1, 20, 5)
            monetary       = st.number_input("Monetary Value (₹)", 500, 500000, 20000)
            recency        = st.slider("Recency (days)", 1, 730, 60)
            time_between   = st.slider("Avg Days Between Purchases", 1, 365, 45)
        with cc:
            nps      = st.slider("Net Promoter Score", 0, 100, 60)
            seasonal = st.slider("Seasonal Spike Factor", 0.8, 2.0, 1.1)

        cd, ce, cf = st.columns(3)
        with cd:
            gender  = st.selectbox("Gender", ["Female","Male"])
            income  = st.selectbox("Income Level", ["Low","Medium","High"])
        with ce:
            loyalty = st.selectbox("Loyalty Tier", ["Non-Prime","Prime"])
            device  = st.selectbox("Device Type", ["Mobile","Desktop","Tablet"])
        with cf:
            discount = st.selectbox("Discount Dependency", ["Low","Medium","High"])
            channel  = st.selectbox("Acquisition Channel",
                                    ["Organic Search","Influencer","Paid Ads","Referral","Social Media"])

        # Top category selector
        top_cat = st.selectbox("Top Category", ["Electronics","Fashion","Beauty","Books","Sports"])

        if st.button("🔮 Predict Churn Risk", use_container_width=True):
            try:
                # Compute RFM scores consistently with training
                r_score = int(np.clip(5 - round(recency / 146), 1, 5))
                f_score = int(np.clip(round(frequency / 4), 1, 5))
                m_score = int(np.clip(round(monetary / 100000 * 5), 1, 5))

                input_dict = {
                    'age': age, 'session_frequency': session_freq,
                    'avg_session_duration': avg_session,
                    'cart_abandonment_rate': cart_abandon,
                    'returns_ratio': returns_ratio,
                    'avg_order_value': avg_order_val,
                    'frequency': frequency,
                    'seasonal_spike_factor': seasonal,
                    'monetary_value': monetary,
                    'time_between_purchases': time_between,
                    'recency': recency,
                    'net_promoter_score': nps,
                    'R_score': r_score, 'F_score': f_score, 'M_score': m_score,
                    'RFM_Score': r_score + f_score + m_score,
                    'gender': gender, 'income_level': income,
                    'loyalty_tier': loyalty, 'device_type': device,
                    'discount_dependency': discount,
                    'acquisition_channel': channel,
                    'top_category': top_cat,
                }
                input_df = pd.DataFrame([input_dict])

                for col_name, le in models['label_encoders'].items():
                    if col_name in input_df.columns:
                        try:
                            input_df[col_name] = le.transform(input_df[col_name])
                        except:
                            input_df[col_name] = 0

                feature_cols = models['churn_features']
                input_final = input_df[feature_cols]
                prob = models['random_forest'].predict_proba(input_final)[0][1]

                st.markdown("---")
                if prob >= 0.7:
                    st.error(f"🔴 **HIGH CHURN RISK: {prob*100:.1f}%**")
                    st.markdown("**Action:** Immediate retention campaign — personalised discount + outreach")
                elif prob >= 0.4:
                    st.warning(f"🟡 **MEDIUM CHURN RISK: {prob*100:.1f}%**")
                    st.markdown("**Action:** Send re-engagement email + loyalty rewards")
                else:
                    st.success(f"🟢 **LOW CHURN RISK: {prob*100:.1f}%**")
                    st.markdown("**Action:** Continue standard engagement — customer is satisfied")

                fig, ax = plt.subplots(figsize=(8, 1.5))
                bar_color = '#ee6666' if prob >= 0.7 else '#f39c12' if prob >= 0.4 else '#2ecc71'
                ax.barh(0, prob, color=bar_color, height=0.5)
                ax.barh(0, 1, color='#e0e0e0', height=0.5, zorder=0)
                ax.set_xlim(0, 1); ax.set_yticks([])
                ax.set_xticks([0,0.25,0.5,0.75,1.0])
                ax.set_xticklabels(['0%','25%','50%','75%','100%'])
                ax.set_title(f'Churn Probability: {prob*100:.1f}%', fontweight='bold')
                ax.axvline(0.4, color='#f39c12', linestyle='--', alpha=0.7)
                ax.axvline(0.7, color='#ee6666', linestyle='--', alpha=0.7)
                plt.tight_layout(); st.pyplot(fig); plt.close()

            except Exception as e:
                st.error(f"Prediction error: {e}")
    else:
        st.info("ℹ️ Run **03_Churn_Prediction.ipynb** to generate model files.")


# ═══════════════════════════════════════════════════════════
# PAGE 4 — Sales Forecasting (Interactive)
# ═══════════════════════════════════════════════════════════
elif page == "📈 Sales Forecasting":
    st.title("📈 Sales Forecasting")
    st.markdown("LSTM Deep Learning model — Fully interactive forecast explorer")
    st.markdown("---")

    # ── Build full monthly data (all categories) ──────────────────
    transactions['year'] = transactions['transaction_date'].dt.year
    transactions['month_dt'] = transactions['transaction_date'].dt.to_period('M').dt.to_timestamp()

    all_monthly = (transactions
        .groupby('month_dt')['total_amount']
        .sum().reset_index())
    all_monthly.columns = ['month_dt','revenue']
    all_monthly = all_monthly.sort_values('month_dt').reset_index(drop=True)

    # ── SIDEBAR-STYLE CONTROLS (in-page columns) ──────────────────
    st.markdown("### ⚙️ Forecast Controls")
    ctrl1, ctrl2, ctrl3 = st.columns(3)

    with ctrl1:
        st.markdown("**📅 Historical View**")
        all_years  = sorted(transactions['year'].unique())
        year_range = st.select_slider(
            "Year range to display",
            options=all_years,
            value=(all_years[0], all_years[-1])
        )
        hist_window = st.radio(
            "Historical chart window",
            ["All time", "Last 3 years", "Last 2 years", "Last 1 year"],
            index=1, horizontal=True
        )

    with ctrl2:
        st.markdown("**🔮 Forecast Settings**")
        forecast_months = st.slider("Forecast horizon (months)", 1, 24, 6)
        growth_scenario = st.selectbox(
            "Growth scenario",
            ["Baseline (LSTM)", "Optimistic (+10%)", "Pessimistic (-10%)",
             "High Growth (+25%)", "Recession (-25%)"]
        )
        confidence_band = st.slider("Confidence band width (%)", 0, 30, 10)

    with ctrl3:
        st.markdown("**📦 Category Filter**")
        categories = ["All"] + sorted(transactions['category'].unique().tolist())
        selected_category = st.selectbox("Filter by category", categories)
        st.markdown("**📊 Chart Style**")
        show_markers  = st.checkbox("Show data points", value=True)
        show_trend    = st.checkbox("Show trend line", value=False)
        show_yoy_ann  = st.checkbox("Annotate YoY growth", value=True)

    st.markdown("---")

    # ── Filter data by category ───────────────────────────────────
    if selected_category == "All":
        filtered_txn = transactions
    else:
        filtered_txn = transactions[transactions['category'] == selected_category]

    monthly = (filtered_txn
        .groupby('month_dt')['total_amount']
        .sum().reset_index())
    monthly.columns = ['month_dt','revenue']
    monthly = monthly.sort_values('month_dt').reset_index(drop=True)

    # ── Filter by year range ──────────────────────────────────────
    monthly = monthly[
        (monthly['month_dt'].dt.year >= year_range[0]) &
        (monthly['month_dt'].dt.year <= year_range[1])
    ].reset_index(drop=True)

    # ── KPI row ───────────────────────────────────────────────────
    total_rev  = monthly['revenue'].sum()
    avg_rev    = monthly['revenue'].mean()
    peak_rev   = monthly['revenue'].max()
    peak_month = monthly.loc[monthly['revenue'].idxmax(), 'month_dt'].strftime('%b %Y')
    latest_rev = monthly['revenue'].iloc[-1]
    mom_change = (monthly['revenue'].iloc[-1] / monthly['revenue'].iloc[-2] - 1) * 100 if len(monthly) > 1 else 0

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("💰 Total Revenue",    fmt_inr(total_rev),
              f"{'All' if selected_category=='All' else selected_category}")
    c2.metric("📊 Avg Monthly",      fmt_inr(avg_rev))
    c3.metric("📈 Peak Month",       fmt_inr(peak_rev), peak_month)
    c4.metric("🗓️ Latest Month",     fmt_inr(latest_rev), f"{mom_change:+.1f}% MoM")
    c5.metric("📅 Months of Data",   f"{len(monthly)}")
    st.markdown("---")

    # ── Historical chart window filter ───────────────────────────
    window_map = {
        "All time": len(monthly),
        "Last 3 years": 36,
        "Last 2 years": 24,
        "Last 1 year":  12,
    }
    display_n = min(window_map[hist_window], len(monthly))
    monthly_display = monthly.tail(display_n).reset_index(drop=True)

    # ── Generate forecast ─────────────────────────────────────────
    scenario_mult = {
        "Baseline (LSTM)":    1.00,
        "Optimistic (+10%)":  1.10,
        "Pessimistic (-10%)": 0.90,
        "High Growth (+25%)": 1.25,
        "Recession (-25%)":   0.75,
    }
    mult = scenario_mult[growth_scenario]

    # Load LSTM model if available, else use exponential smoothing
    lstm_model_path = 'models/lstm_best.keras'
    scaler_path     = 'models/scaler_lstm.pkl'

    forecast_dates = pd.date_range(
        start = monthly['month_dt'].max() + pd.offsets.MonthBegin(1),
        periods = forecast_months, freq='MS'
    )

    if os.path.exists(lstm_model_path) and os.path.exists(scaler_path):
        try:
            from tensorflow.keras.models import load_model as keras_load
            from sklearn.preprocessing import MinMaxScaler

            lstm_mdl  = keras_load(lstm_model_path)
            with open(scaler_path, 'rb') as f:
                scaler_lstm = pickle.load(f)

            LOOK_BACK = 12
            all_rev_scaled = scaler_lstm.transform(
                all_monthly.set_index('month_dt').reindex(monthly['month_dt'])['revenue']
                .fillna(method='ffill').values.reshape(-1,1)
            )
            last_seq = all_rev_scaled[-LOOK_BACK:].copy()
            future_preds = []
            for _ in range(forecast_months):
                seq = last_seq[-LOOK_BACK:].reshape(1, LOOK_BACK, 1)
                pred = lstm_mdl.predict(seq, verbose=0)[0, 0]
                future_preds.append(pred)
                last_seq = np.append(last_seq, [[pred]], axis=0)
            base_forecast = scaler_lstm.inverse_transform(
                np.array(future_preds).reshape(-1,1)).flatten()
            forecast_source = "LSTM model"
        except Exception:
            base_forecast = None
            forecast_source = "smoothing"
    else:
        base_forecast = None
        forecast_source = "smoothing"

    if base_forecast is None:
        # Exponential smoothing fallback
        alpha = 0.3
        smoothed = monthly['revenue'].ewm(alpha=alpha).mean()
        trend    = (smoothed.iloc[-1] - smoothed.iloc[-min(6, len(smoothed))]) / min(6, len(smoothed))
        base_forecast = np.array([smoothed.iloc[-1] + trend * (i+1) for i in range(forecast_months)])
        forecast_source = "exponential smoothing"

    adjusted_forecast = base_forecast * mult
    upper_band = adjusted_forecast * (1 + confidence_band/100)
    lower_band = adjusted_forecast * (1 - confidence_band/100)

    forecast_df = pd.DataFrame({
        'date':             forecast_dates,
        'forecast_revenue': adjusted_forecast,
        'upper':            upper_band,
        'lower':            lower_band,
    })

    # ── Main forecast chart ───────────────────────────────────────
    st.markdown(f"#### 📊 Historical Revenue + {forecast_months}-Month Forecast  "
                f"<span style='font-size:0.85rem;color:#888;'>({growth_scenario} · via {forecast_source})</span>",
                unsafe_allow_html=True)

    marker_style = 'o' if show_markers else None
    ms = 3 if show_markers else 0

    fig, ax = plt.subplots(figsize=(15, 5))

    # Historical
    ax.plot(monthly_display['month_dt'], monthly_display['revenue'],
            color='royalblue', linewidth=2, marker=marker_style, markersize=ms,
            label='Historical Revenue', zorder=3)
    ax.fill_between(monthly_display['month_dt'], monthly_display['revenue'],
                    alpha=0.08, color='royalblue')

    # Optional trend line on historical
    if show_trend and len(monthly_display) > 2:
        x_num = np.arange(len(monthly_display))
        z = np.polyfit(x_num, monthly_display['revenue'], 1)
        p = np.poly1d(z)
        ax.plot(monthly_display['month_dt'], p(x_num),
                color='navy', linewidth=1.2, linestyle=':', alpha=0.6, label='Trend')

    # Forecast
    scenario_colors = {
        "Baseline (LSTM)":    'darkorange',
        "Optimistic (+10%)":  '#2ecc71',
        "Pessimistic (-10%)": '#e74c3c',
        "High Growth (+25%)": '#27ae60',
        "Recession (-25%)":   '#c0392b',
    }
    fc_color = scenario_colors[growth_scenario]

    ax.plot(forecast_df['date'], forecast_df['forecast_revenue'],
            color=fc_color, linewidth=2.5, marker='o', markersize=6,
            linestyle='--', label=f'Forecast ({growth_scenario})', zorder=4)

    if confidence_band > 0:
        ax.fill_between(forecast_df['date'], forecast_df['lower'], forecast_df['upper'],
                        color=fc_color, alpha=0.15,
                        label=f'±{confidence_band}% Confidence Band')

    # Vertical divider at forecast start
    ax.axvline(forecast_df['date'].iloc[0], color='grey', linestyle=':', alpha=0.5)
    ax.text(forecast_df['date'].iloc[0], ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1,
            '  Forecast →', color='grey', fontsize=9, va='top')

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'₹{x/1e5:.0f}L'))
    ax.set_xlabel('Month')
    ax.set_ylabel('Revenue')
    title_cat = f" — {selected_category}" if selected_category != "All" else ""
    ax.set_title(f'Revenue Forecast{title_cat}  |  {forecast_months}-Month Horizon', fontweight='bold')
    ax.legend(loc='upper left')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Forecast table ────────────────────────────────────────────
    st.markdown("---")
    col_table, col_chart = st.columns([1, 1])

    with col_table:
        st.markdown("#### 📋 Forecast Table")
        tbl = forecast_df.copy()
        tbl['Month']              = tbl['date'].dt.strftime('%b %Y')
        tbl['Forecast Revenue']   = tbl['forecast_revenue'].map(lambda x: f'₹{x:,.0f}')
        tbl['Lower Bound']        = tbl['lower'].map(lambda x: f'₹{x:,.0f}')
        tbl['Upper Bound']        = tbl['upper'].map(lambda x: f'₹{x:,.0f}')

        # MoM change vs last actual
        prev = monthly['revenue'].iloc[-1]
        mom_list = []
        for val in tbl['forecast_revenue']:
            pct = (val / prev - 1) * 100
            mom_list.append(f"{'↑' if pct>=0 else '↓'} {abs(pct):.1f}%")
            prev = val
        tbl['MoM Change'] = mom_list

        st.dataframe(
            tbl[['Month','Forecast Revenue','Lower Bound','Upper Bound','MoM Change']],
            use_container_width=True, hide_index=True
        )
        total_fc = forecast_df['forecast_revenue'].sum()
        st.success(f"**Total Forecasted Revenue ({forecast_months}M): {fmt_inr(total_fc)}**")

    with col_chart:
        st.markdown("#### 📊 Monthly Forecast Bar")
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        bar_colors = [fc_color] * len(forecast_df)
        bars = ax2.bar(forecast_df['date'].dt.strftime('%b\n%Y'),
                       forecast_df['forecast_revenue'],
                       color=bar_colors, alpha=0.85, edgecolor='white', linewidth=0.8)
        if confidence_band > 0:
            ax2.errorbar(
                x=range(len(forecast_df)),
                y=forecast_df['forecast_revenue'],
                yerr=[forecast_df['forecast_revenue'] - forecast_df['lower'],
                      forecast_df['upper'] - forecast_df['forecast_revenue']],
                fmt='none', color='grey', capsize=5, linewidth=1.2, alpha=0.7
            )
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'₹{x/1e5:.0f}L'))
        ax2.set_title(f'{growth_scenario}', fontweight='bold', fontsize=10)
        ax2.set_ylabel('Revenue')
        for bar, val in zip(bars, forecast_df['forecast_revenue']):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.01,
                     fmt_inr(val), ha='center', fontsize=7.5, fontweight='bold', rotation=30)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    # ── Scenario Comparison ───────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🔀 Scenario Comparison")

    scenario_list = ["Baseline (LSTM)", "Optimistic (+10%)", "Pessimistic (-10%)",
                     "High Growth (+25%)", "Recession (-25%)"]
    sc_colors     = ['royalblue','#2ecc71','#e74c3c','#27ae60','#c0392b']

    fig3, ax3 = plt.subplots(figsize=(13, 5))
    ax3.plot(monthly_display['month_dt'].tail(12), monthly_display['revenue'].tail(12),
             color='steelblue', linewidth=2, label='Historical (last 12M)', zorder=5)

    for sc, color in zip(scenario_list, sc_colors):
        sc_fc = base_forecast * scenario_mult[sc]
        ax3.plot(forecast_df['date'], sc_fc,
                 marker='o', markersize=4, linewidth=1.8, linestyle='--',
                 color=color, label=sc, alpha=0.85)

    ax3.axvline(forecast_df['date'].iloc[0], color='grey', linestyle=':', alpha=0.5)
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'₹{x/1e5:.0f}L'))
    ax3.set_title('All Scenarios — Side by Side Comparison', fontweight='bold')
    ax3.set_xlabel('Month')
    ax3.set_ylabel('Revenue')
    ax3.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

    # ── YoY Analysis ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📅 Year-over-Year Revenue")

    yoy_data = monthly.copy()
    yoy_data['year'] = yoy_data['month_dt'].dt.year
    yoy = yoy_data.groupby('year')['revenue'].sum()

    fig4, ax4 = plt.subplots(figsize=(11, 4))
    bar_palette = sns.color_palette('Blues_d', len(yoy))
    bars4 = ax4.bar(yoy.index, yoy.values, color=bar_palette, edgecolor='white')
    ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f'₹{x/1e7:.1f}Cr'))
    ax4.set_title('Annual Revenue (Year-over-Year)', fontweight='bold')

    prev_val = None
    for bar, val in zip(bars4, yoy.values):
        label = fmt_inr(val)
        if show_yoy_ann and prev_val:
            pct = (val/prev_val - 1)*100
            label += f"\n({'↑' if pct>=0 else '↓'}{abs(pct):.1f}%)"
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.01,
                 label, ha='center', fontsize=8, fontweight='bold')
        prev_val = val

    plt.tight_layout()
    st.pyplot(fig4)
    plt.close()


# ═══════════════════════════════════════════════════════════
# PAGE 5 — Product Recommendations
# ═══════════════════════════════════════════════════════════
elif page == "💡 Product Recommendations":
    st.title("💡 Product Recommendation System")
    st.markdown("Popularity-based & Collaborative Filtering")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["🔥 Popular Products", "👤 Customer Recommendations", "🔗 Similar Products"])

    # ─── Tab 1: Popular Products ───
    with tab1:
        st.markdown("#### Top Products by Popularity Score")

        product_stats = transactions.groupby('product_id').agg(
            total_orders  = ('transaction_id','count'),
            total_revenue = ('total_amount','sum'),
        ).reset_index()

        # Use mrp (not selling_price — column doesn't exist)
        product_stats = product_stats.merge(
            products[['product_id','product_name','product_category','mrp','rating']],
            on='product_id', how='left'
        )

        def norm(s): return (s - s.min()) / (s.max() - s.min() + 1e-9)
        product_stats['popularity_score'] = (
            0.40 * norm(product_stats['total_orders']) +
            0.35 * norm(product_stats['total_revenue']) +
            0.25 * norm(product_stats['rating'])
        )
        top_products = product_stats.sort_values('popularity_score', ascending=False)

        col1, col2 = st.columns([1,3])
        with col1:
            cat_filter = st.selectbox("Filter by Category",
                ['All'] + sorted(products['product_category'].unique().tolist()))
            top_n = st.slider("Number of products", 5, 20, 10)

        filtered = top_products if cat_filter == 'All' else \
                   top_products[top_products['product_category'] == cat_filter]
        display = filtered.head(top_n)[['product_name','product_category','total_orders','rating','popularity_score']].copy()
        display.columns = ['Product','Category','Orders','Rating','Popularity Score']
        display['Popularity Score'] = display['Popularity Score'].round(4)

        with col2:
            st.dataframe(display, use_container_width=True, hide_index=True)

        top10 = top_products.head(10)
        fig, ax = plt.subplots(figsize=(10,5))
        ax.barh(top10['product_name'].str[:35][::-1], top10['popularity_score'][::-1],
                color=plt.cm.RdYlGn(np.linspace(0.3, 0.9, 10)))
        ax.set_xlabel('Popularity Score')
        ax.set_title('Top 10 Most Popular Products', fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # ─── Tab 2: Customer Recommendations ───
    with tab2:
        st.markdown("#### 👤 Customer Recommendation Engine")
        st.markdown("Choose an **existing customer** or **build a custom profile** to get personalised recommendations.")

        mode = st.radio(
            "Customer source",
            ["🔍 Existing Customer", "✏️ Custom Profile"],
            horizontal=True, key="rec_mode"
        )
        st.markdown("---")

        if 'user_sim' in models:
            user_sim_df  = models['user_sim']
            all_cust_ids = list(user_sim_df.index)

            if 'user_item_matrix' in models:
                user_item = models['user_item_matrix']
            else:
                txn_active = transactions[transactions['customer_id'].isin(all_cust_ids)]
                user_item  = txn_active.groupby(['customer_id','product_id'])['quantity'].sum().unstack(fill_value=0)

            # ── MODE 1: Existing Customer ────────────────────────
            if mode == "🔍 Existing Customer":
                col_sel, col_prof = st.columns([1, 2])

                with col_sel:
                    customer_id = st.selectbox("Select Customer ID", all_cust_ids[:200], key="exist_cust")
                    top_n_recs  = st.slider("No. of recommendations", 3, 15, 6, key="exist_n")
                    get_btn     = st.button("🎯 Get Recommendations", use_container_width=True, key="exist_btn")

                with col_prof:
                    cust_row = customers[customers['customer_id'] == customer_id]
                    if not cust_row.empty:
                        r = cust_row.iloc[0]
                        st.markdown("**Customer Profile**")
                        p1, p2, p3 = st.columns(3)
                        p1.metric("Age",         r['age'])
                        p2.metric("Gender",      r['gender'])
                        p3.metric("Loyalty",     r['loyalty_tier'])
                        p4, p5, p6 = st.columns(3)
                        p4.metric("Top Category", r['top_category'])
                        p5.metric("NPS",          r['net_promoter_score'])
                        p6.metric("Churn Risk",   "⚠️ Yes" if r['churn'] == 1 else "✅ No")

                if get_btn:
                    if customer_id in user_item.index:
                        similar_users  = user_sim_df[customer_id].drop(customer_id).nlargest(10).index
                        sim_purchases  = user_item.loc[similar_users].sum(axis=0)
                        already_bought = user_item.loc[customer_id]
                        not_bought_ids = already_bought[already_bought == 0].index
                        recs = sim_purchases[not_bought_ids].nlargest(top_n_recs).reset_index()
                        recs.columns = ['product_id', 'score']
                        recs = recs.merge(
                            products[['product_id','product_name','product_category','mrp','rating','number_of_reviews']],
                            on='product_id', how='left')

                        st.success(f"✅ Showing {len(recs)} recommendations for **{customer_id}**")

                        cols_per_row = 3
                        for row_start in range(0, len(recs), cols_per_row):
                            row_recs  = recs.iloc[row_start:row_start+cols_per_row]
                            card_cols = st.columns(cols_per_row)
                            for col, (_, rec) in zip(card_cols, row_recs.iterrows()):
                                with col:
                                    stars = "⭐" * int(round(rec['rating']))
                                    st.markdown(f"""
<div style="background:#fff;border-radius:10px;padding:14px;
            box-shadow:0 2px 8px rgba(0,0,0,0.08);margin-bottom:10px;">
  <div style="font-weight:700;font-size:0.9rem;color:#1a1a2e;margin-bottom:4px;">
    {rec['product_name'][:38]}
  </div>
  <div style="color:#888;font-size:0.78rem;margin-bottom:6px;">
    {rec['product_category']}
  </div>
  <div style="font-size:0.82rem;">{stars} {rec['rating']:.1f}
    &nbsp;·&nbsp; {int(rec['number_of_reviews']):,} reviews
  </div>
  <div style="font-weight:700;color:#e94560;font-size:1rem;margin-top:6px;">
    ₹{rec['mrp']:,}
  </div>
</div>""", unsafe_allow_html=True)

                        fig, ax = plt.subplots(figsize=(9, 3.5))
                        ax.barh(recs['product_name'].str[:32][::-1],
                                recs['score'][::-1],
                                color=plt.cm.Blues(np.linspace(0.4, 0.85, len(recs))))
                        ax.set_xlabel('Collaborative Filtering Score')
                        ax.set_title('Recommendation Strength', fontweight='bold', fontsize=10)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                    else:
                        st.info("Customer not in model matrix — showing top popular products instead.")
                        fb = top_products.head(top_n_recs)[['product_name','product_category','mrp','rating']]
                        st.dataframe(fb, use_container_width=True, hide_index=True)

            # ── MODE 2: Custom Profile ───────────────────────────
            else:
                st.markdown("##### 🧑 Define Your Customer Profile")
                st.caption("Only key demographics needed — recommendations are driven by purchase behaviour of similar customers.")

                ca, cb, cc = st.columns(3)
                with ca:
                    c_age     = st.slider("Age", 18, 70, 30, key="c_age")
                    c_gender  = st.selectbox("Gender", ["Female", "Male"], key="c_gender")
                    c_loyalty = st.selectbox("Loyalty Tier", ["Non-Prime", "Prime"], key="c_loyalty")
                with cb:
                    c_top_cat = st.selectbox("Preferred Category",
                                             ["Electronics","Fashion","Beauty","Books","Sports"], key="c_cat")
                    c_income  = st.selectbox("Income Level", ["Low","Medium","High"], key="c_income")
                    c_device  = st.selectbox("Device", ["Mobile","Desktop","Tablet"], key="c_device")
                with cc:
                    c_freq    = st.slider("Purchase Frequency", 1, 20, 5, key="c_freq")
                    c_recency = st.slider("Days Since Last Purchase", 1, 365, 45, key="c_recency")
                    c_top_n   = st.slider("No. of recommendations", 3, 15, 6, key="c_topn")

                st.markdown("---")
                custom_btn = st.button("🎯 Get Recommendations for Custom Profile",
                                       use_container_width=True, key="custom_btn")

                if custom_btn:
                    pool = customers.copy()

                    pool['cat_match']     = (pool['top_category'] == c_top_cat).astype(int)
                    pool['loyalty_match'] = (pool['loyalty_tier'] == c_loyalty).astype(int)
                    pool['income_match']  = (pool['income_level'] == c_income).astype(int)
                    pool['device_match']  = (pool['device_type']  == c_device).astype(int)
                    pool['gender_match']  = (pool['gender']       == c_gender).astype(int)

                    def prox(col, val):
                        rng = pool[col].max() - pool[col].min() + 1e-9
                        return 1 - (pool[col] - val).abs() / rng

                    pool['age_prox']     = prox('age',       c_age)
                    pool['freq_prox']    = prox('frequency', c_freq)
                    pool['recency_prox'] = prox('recency',   c_recency)

                    pool['similarity_score'] = (
                        0.25 * pool['cat_match'] +
                        0.15 * pool['loyalty_match'] +
                        0.10 * pool['income_match'] +
                        0.10 * pool['device_match'] +
                        0.05 * pool['gender_match'] +
                        0.15 * pool['age_prox'] +
                        0.10 * pool['freq_prox'] +
                        0.10 * pool['recency_prox']
                    )

                    pool_in_matrix = pool[pool['customer_id'].isin(user_item.index)]
                    top_similar    = pool_in_matrix.nlargest(15, 'similarity_score')['customer_id'].tolist()

                    with st.expander("👥 Top matched customer profiles used to generate recommendations"):
                        show_cols = ['customer_id','age','gender','loyalty_tier',
                                     'top_category','income_level','frequency','recency','similarity_score']
                        match_display = pool_in_matrix[pool_in_matrix['customer_id'].isin(top_similar)][show_cols]
                        match_display = match_display.sort_values('similarity_score', ascending=False).copy()
                        match_display['similarity_score'] = match_display['similarity_score'].round(3)
                        st.dataframe(match_display, use_container_width=True, hide_index=True)

                    if top_similar:
                        sim_purchases = user_item.loc[top_similar].sum(axis=0)
                        cat_products  = set(products.loc[products['product_category'] == c_top_cat, 'product_id'])

                        recs_all = sim_purchases.nlargest(c_top_n * 3).reset_index()
                        recs_all.columns = ['product_id', 'score']
                        recs_all = recs_all.merge(
                            products[['product_id','product_name','product_category',
                                      'mrp','rating','number_of_reviews','is_prime_eligible']],
                            on='product_id', how='left')

                        recs_all['final_score'] = recs_all.apply(
                            lambda row: row['score'] * 1.4 if row['product_id'] in cat_products else row['score'], axis=1)

                        if c_loyalty == "Prime":
                            recs_all.loc[recs_all['is_prime_eligible'] == 'Yes', 'final_score'] *= 1.15

                        recs = recs_all.sort_values('final_score', ascending=False).head(c_top_n)

                        st.info(
                            f"🧑 **Custom Profile** — Age {c_age} | {c_gender} | {c_loyalty} | "
                            f"Prefers: **{c_top_cat}** | {c_income} income | {c_device} | "
                            f"Buys every ~{c_recency} days"
                        )
                        st.success(f"✅ {len(recs)} recommendations based on {len(top_similar)} similar customers")

                        cols_per_row = 3
                        for row_start in range(0, len(recs), cols_per_row):
                            row_recs  = recs.iloc[row_start:row_start+cols_per_row]
                            card_cols = st.columns(cols_per_row)
                            for col, (_, rec) in zip(card_cols, row_recs.iterrows()):
                                with col:
                                    stars      = "⭐" * int(round(rec['rating']))
                                    cat_badge  = "🎯 Preferred" if rec['product_category'] == c_top_cat else ""
                                    prime_badge = "⭐ Prime" if rec['is_prime_eligible'] == 'Yes' else ""
                                    st.markdown(f"""
<div style="background:#fff;border-radius:10px;padding:14px;
            box-shadow:0 2px 8px rgba(0,0,0,0.08);margin-bottom:10px;
            border-left:4px solid {'#e94560' if rec['product_category']==c_top_cat else '#ddd'};">
  <div style="font-weight:700;font-size:0.9rem;color:#1a1a2e;margin-bottom:4px;">
    {rec['product_name'][:38]}
  </div>
  <div style="color:#888;font-size:0.78rem;margin-bottom:4px;">
    {rec['product_category']} &nbsp;
    <span style="color:#e94560;font-weight:600;">{cat_badge}</span>
    <span style="color:#f39c12;">{prime_badge}</span>
  </div>
  <div style="font-size:0.82rem;">{stars} {rec['rating']:.1f}
    &nbsp;·&nbsp; {int(rec['number_of_reviews']):,} reviews
  </div>
  <div style="font-weight:700;color:#e94560;font-size:1rem;margin-top:6px;">
    ₹{rec['mrp']:,}
  </div>
</div>""", unsafe_allow_html=True)

                        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
                        axes[0].barh(
                            recs['product_name'].str[:30][::-1],
                            recs['final_score'][::-1],
                            color=['#e94560' if cat == c_top_cat else 'steelblue'
                                   for cat in recs['product_category'][::-1]]
                        )
                        axes[0].set_xlabel('Recommendation Score')
                        axes[0].set_title('Score (🔴 = preferred category)', fontweight='bold', fontsize=9)

                        cat_counts = recs['product_category'].value_counts()
                        axes[1].pie(cat_counts, labels=cat_counts.index, autopct='%1.0f%%',
                                    colors=sns.color_palette('husl', len(cat_counts)), startangle=90)
                        axes[1].set_title('Recommended Category Mix', fontweight='bold', fontsize=9)

                        plt.suptitle(f'Recommendations for Custom Profile — {c_top_cat} Preferred',
                                     fontweight='bold', fontsize=11)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                    else:
                        st.warning("No matching customers found in model. Try adjusting the profile.")

        else:
            st.info("ℹ️ Run **05_Product_Recommendation.ipynb** to generate model files.")

    # ─── Tab 3: Similar Products ───
    with tab3:
        st.markdown("#### 🔗 Find Similar Products")

        if 'item_sim' in models:
            item_sim_df = models['item_sim']
            prod_in_model = products[products['product_id'].isin(item_sim_df.index)].copy()

            # ── Search controls ──────────────────────────────────
            s1, s2, s3 = st.columns([2, 1, 1])
            with s1:
                search_query = st.text_input("🔍 Search product by name", placeholder="e.g. iPhone, Nike, Lipstick...")
            with s2:
                cat_filter_sim = st.selectbox("Filter by category",
                    ["All"] + sorted(prod_in_model['product_category'].unique().tolist()),
                    key="sim_cat_filter")
            with s3:
                top_n_sim = st.slider("Similar products to show", 3, 12, 5, key="item_sim_n")

            # ── Apply filters ────────────────────────────────────
            filtered_prods = prod_in_model.copy()
            if cat_filter_sim != "All":
                filtered_prods = filtered_prods[filtered_prods['product_category'] == cat_filter_sim]
            if search_query.strip():
                filtered_prods = filtered_prods[
                    filtered_prods['product_name'].str.contains(search_query.strip(), case=False, na=False)
                ]

            if filtered_prods.empty:
                st.warning("No products match your search. Try a different name or clear the category filter.")
            else:
                # ── Product selector from filtered results ────────
                st.markdown(f"**{len(filtered_prods)} product(s) found**")
                selected = st.selectbox(
                    "Select a product",
                    filtered_prods['product_id'].tolist(),
                    format_func=lambda pid: (
                        f"{products.loc[products['product_id']==pid,'product_name'].values[0]}"
                        f"  —  {products.loc[products['product_id']==pid,'product_category'].values[0]}"
                        f"  |  ₹{products.loc[products['product_id']==pid,'mrp'].values[0]:,}"
                    ),
                    key="sim_product_select"
                )

                # ── Selected product details card ─────────────────
                sel_row = products[products['product_id'] == selected].iloc[0]
                st.markdown("---")
                sc1, sc2, sc3, sc4, sc5 = st.columns(5)
                sc1.metric("Category",  sel_row['product_category'])
                sc2.metric("Brand",     sel_row['brand'])
                sc3.metric("MRP",       f"₹{sel_row['mrp']:,}")
                sc4.metric("Rating",    f"{sel_row['rating']} ⭐")
                sc5.metric("Reviews",   f"{int(sel_row['number_of_reviews']):,}")
                st.markdown("---")

                # ── Similarity options ────────────────────────────
                oc1, oc2 = st.columns([2, 1])
                with oc1:
                    sim_category_filter = st.radio(
                        "Show similar products from",
                        ["All categories", "Same category only", "Different categories only"],
                        horizontal=True, key="sim_scope"
                    )
                with oc2:
                    min_rating = st.slider("Minimum rating", 1.0, 5.0, 3.0, 0.5, key="sim_min_rating")

                find_btn = st.button("🔗 Find Similar Products", use_container_width=True, key="find_sim_btn")

                if find_btn:
                    sim_scores = item_sim_df[selected].drop(selected)

                    # Apply category scope filter
                    sel_cat = sel_row['product_category']
                    if sim_category_filter == "Same category only":
                        same_cat_ids = products.loc[products['product_category'] == sel_cat, 'product_id']
                        sim_scores = sim_scores[sim_scores.index.isin(same_cat_ids)]
                    elif sim_category_filter == "Different categories only":
                        diff_cat_ids = products.loc[products['product_category'] != sel_cat, 'product_id']
                        sim_scores = sim_scores[sim_scores.index.isin(diff_cat_ids)]

                    sim_df = sim_scores.nlargest(top_n_sim * 4).reset_index()
                    sim_df.columns = ['product_id', 'similarity']
                    sim_df = sim_df.merge(
                        products[['product_id','product_name','product_category',
                                  'brand','mrp','rating','number_of_reviews','is_prime_eligible']],
                        on='product_id', how='left')

                    # Apply minimum rating filter
                    sim_df = sim_df[sim_df['rating'] >= min_rating].head(top_n_sim)

                    if sim_df.empty:
                        st.warning("No similar products match the filters. Try lowering the minimum rating.")
                    else:
                        sel_name = sel_row['product_name']
                        st.success(f"✅ {len(sim_df)} products similar to **{sel_name}**")

                        # ── Product cards ─────────────────────────
                        cols_per_row = 3
                        for row_start in range(0, len(sim_df), cols_per_row):
                            row_items = sim_df.iloc[row_start:row_start+cols_per_row]
                            card_cols = st.columns(cols_per_row)
                            for col, (_, item) in zip(card_cols, row_items.iterrows()):
                                with col:
                                    stars       = "⭐" * int(round(item['rating']))
                                    same_cat    = item['product_category'] == sel_cat
                                    prime_badge = "⭐ Prime" if item['is_prime_eligible'] == 'Yes' else ""
                                    sim_pct     = item['similarity'] * 100
                                    bar_fill    = int(sim_pct)
                                    st.markdown(f"""
<div style="background:#fff;border-radius:10px;padding:14px;
            box-shadow:0 2px 8px rgba(0,0,0,0.08);margin-bottom:10px;
            border-left:4px solid {'#3498db' if same_cat else '#9b59b6'};">
  <div style="font-weight:700;font-size:0.88rem;color:#1a1a2e;margin-bottom:3px;">
    {item['product_name'][:38]}
  </div>
  <div style="color:#888;font-size:0.76rem;margin-bottom:4px;">
    {item['brand']} &nbsp;·&nbsp; {item['product_category']}
    &nbsp;<span style="color:#f39c12;">{prime_badge}</span>
  </div>
  <div style="font-size:0.8rem;margin-bottom:4px;">
    {stars} {item['rating']:.1f} &nbsp;·&nbsp; {int(item['number_of_reviews']):,} reviews
  </div>
  <div style="font-weight:700;color:#e94560;font-size:0.95rem;margin-bottom:6px;">
    ₹{item['mrp']:,}
  </div>
  <div style="font-size:0.75rem;color:#555;margin-bottom:3px;">Similarity: {sim_pct:.1f}%</div>
  <div style="background:#eee;border-radius:4px;height:5px;">
    <div style="background:{'#3498db' if same_cat else '#9b59b6'};
                width:{bar_fill}%;height:5px;border-radius:4px;"></div>
  </div>
</div>""", unsafe_allow_html=True)

                        st.markdown("---")

                        # ── Charts ────────────────────────────────
                        ch1, ch2 = st.columns(2)

                        with ch1:
                            fig, ax = plt.subplots(figsize=(7, 4))
                            bar_colors = ['#3498db' if cat == sel_cat else '#9b59b6'
                                          for cat in sim_df['product_category'][::-1]]
                            ax.barh(sim_df['product_name'].str[:28][::-1],
                                    sim_df['similarity'][::-1],
                                    color=bar_colors)
                            ax.set_xlabel('Cosine Similarity')
                            ax.set_title(
                                f'Similarity Scores\n(🔵 same cat · 🟣 different cat)',
                                fontweight='bold', fontsize=9)
                            ax.set_xlim(0, 1)
                            for i, (_, row) in enumerate(sim_df[::-1].iterrows()):
                                ax.text(row['similarity'] + 0.01, i,
                                        f"{row['similarity']*100:.1f}%", va='center', fontsize=8)
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()

                        with ch2:
                            fig, ax = plt.subplots(figsize=(7, 4))
                            scatter = ax.scatter(
                                sim_df['mrp'],
                                sim_df['rating'],
                                s=sim_df['similarity'] * 800,
                                c=sim_df['similarity'],
                                cmap='Blues', alpha=0.8, edgecolors='white', linewidths=0.8
                            )
                            # Highlight selected product
                            ax.scatter(sel_row['mrp'], sel_row['rating'],
                                       s=300, color='#e94560', zorder=5,
                                       marker='*', label='Selected product')
                            plt.colorbar(scatter, ax=ax, label='Similarity')
                            ax.set_xlabel('MRP (₹)')
                            ax.set_ylabel('Rating')
                            ax.set_title('Price vs Rating\n(bubble size = similarity)',
                                         fontweight='bold', fontsize=9)
                            ax.xaxis.set_major_formatter(
                                mticker.FuncFormatter(lambda x,_: f'₹{x/1e3:.0f}K'))
                            ax.legend(fontsize=8)
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()

                        # ── Summary table ─────────────────────────
                        st.markdown("#### 📋 Comparison Table")
                        tbl = sim_df[['product_name','brand','product_category',
                                      'mrp','rating','similarity']].copy()
                        tbl.columns = ['Product','Brand','Category','MRP (₹)','Rating','Similarity']
                        tbl['MRP (₹)']    = tbl['MRP (₹)'].map(lambda x: f'₹{x:,}')
                        tbl['Similarity'] = tbl['Similarity'].map(lambda x: f'{x*100:.1f}%')
                        st.dataframe(tbl, use_container_width=True, hide_index=True)

        else:
            st.info("ℹ️ Run **05_Product_Recommendation.ipynb** to generate model files.")