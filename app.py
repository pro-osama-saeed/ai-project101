import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="AI House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Enhanced Custom CSS ---
st.markdown("""
<style>
    /* Main background gradient */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Custom container styling */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Card styling */
    .custom-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }
    
    /* Title styling */
    .title-text {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    /* Subtitle styling */
    .subtitle-text {
        text-align: center;
        color: #4a5568;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        color: #667eea;
        font-weight: 700;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        color: white;
    }
    
    /* Info box styling */
    .info-box {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    /* Feature card */
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- Load and Train Model ---
@st.cache_resource
def train_model():
    try:
        df = pd.read_csv('house_data.csv')
        X = df[['Size', 'Rooms', 'Age']]
        y = df['Price']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        return model, accuracy, mae, rmse, df, X_test, y_test, y_pred
    except FileNotFoundError:
        return None, None, None, None, None, None, None, None

model, accuracy, mae, rmse, df, X_test, y_test, y_pred = train_model()

# --- Sidebar ---
with st.sidebar:
    st.markdown("### 🏡 Configure Your House")
    st.markdown("---")
    
    size = st.slider(
        "🏗️ Size (Square Feet)", 
        min_value=500, 
        max_value=5000, 
        value=1500, 
        step=50,
        help="Total living area in square feet"
    )
    
    rooms = st.slider(
        "🛏️ Number of Bedrooms", 
        min_value=1, 
        max_value=10, 
        value=3, 
        step=1,
        help="Total number of bedrooms"
    )
    
    age = st.slider(
        "📅 Age of House (Years)", 
        min_value=0, 
        max_value=100, 
        value=10, 
        step=1,
        help="Years since the house was built"
    )
    
    st.markdown("---")
    predict_button = st.button("🔮 Predict Price", type="primary")
    
    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    if df is not None:
        st.metric("Total Houses", len(df))
        st.metric("Avg Price", f"${df['Price'].mean():,.0f}")
        st.metric("Price Range", f"${df['Price'].min():,.0f} - ${df['Price'].max():,.0f}")

# --- Main Content ---
st.markdown('<h1 class="title-text">🏠 AI Real Estate Valuator</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Predict house prices instantly using advanced Machine Learning</p>', unsafe_allow_html=True)

if model is None:
    st.error("❌ Error: 'house_data.csv' not found. Please upload it to your GitHub repo.")
    st.stop()

# --- Prediction Section ---
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Prediction", "📈 Model Performance", "🔍 Data Explorer", "ℹ️ About"])

with tab1:
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### Your House Details")
        
        detail_col1, detail_col2, detail_col3 = st.columns(3)
        with detail_col1:
            st.markdown(f"""
            <div class="feature-card">
                <h3 style="color: #667eea; margin-bottom: 0.5rem;">🏗️ Size</h3>
                <p style="font-size: 1.5rem; font-weight: 600; margin: 0;">{size:,} sqft</p>
            </div>
            """, unsafe_allow_html=True)
        
        with detail_col2:
            st.markdown(f"""
            <div class="feature-card">
                <h3 style="color: #667eea; margin-bottom: 0.5rem;">🛏️ Bedrooms</h3>
                <p style="font-size: 1.5rem; font-weight: 600; margin: 0;">{rooms}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with detail_col3:
            st.markdown(f"""
            <div class="feature-card">
                <h3 style="color: #667eea; margin-bottom: 0.5rem;">📅 Age</h3>
                <p style="font-size: 1.5rem; font-weight: 600; margin: 0;">{age} years</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Make Prediction
        input_data = [[size, rooms, age]]
        prediction = model.predict(input_data)[0]
        
        # Display prediction with animation effect
        st.markdown("### 💰 Estimated Price")
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; border-radius: 15px; text-align: center;">
            <h1 style="color: white; font-size: 3rem; margin: 0;">${prediction:,.2f}</h1>
            <p style="color: rgba(255,255,255,0.9); margin-top: 0.5rem; font-size: 1.1rem;">
                Predicted Market Value
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Price breakdown
        st.markdown("### 📊 Price Breakdown")
        
        # Calculate contribution of each feature (simplified visualization)
        coef = model.coef_
        intercept = model.intercept_
        
        size_contribution = coef[0] * size
        rooms_contribution = coef[1] * rooms
        age_contribution = coef[2] * age
        
        contribution_df = pd.DataFrame({
            'Feature': ['Base Price', 'Size', 'Bedrooms', 'Age'],
            'Contribution': [intercept, size_contribution, rooms_contribution, age_contribution]
        })
        
        fig = px.bar(
            contribution_df, 
            x='Feature', 
            y='Contribution',
            color='Contribution',
            color_continuous_scale='Purples',
            title='Feature Contribution to Price'
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Comparison with dataset
        st.markdown("### 📍 Where Does This Price Stand?")
        percentile = (df['Price'] < prediction).sum() / len(df) * 100
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Lower than", f"{percentile:.1f}% of houses")
        with col_b:
            st.metric("Higher than", f"{100-percentile:.1f}% of houses")
        with col_c:
            avg_diff = ((prediction - df['Price'].mean()) / df['Price'].mean()) * 100
            st.metric("vs Average", f"{avg_diff:+.1f}%")

with tab2:
    st.markdown("### 🎯 Model Performance Metrics")
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric(
            "R² Score", 
            f"{accuracy:.2%}",
            help="Coefficient of determination - how well the model fits the data"
        )
    
    with metric_col2:
        st.metric(
            "MAE", 
            f"${mae:,.0f}",
            help="Mean Absolute Error - average prediction error"
        )
    
    with metric_col3:
        st.metric(
            "RMSE", 
            f"${rmse:,.0f}",
            help="Root Mean Squared Error - standard deviation of prediction errors"
        )
    
    with metric_col4:
        accuracy_percentage = accuracy * 100
        st.metric(
            "Accuracy", 
            f"{accuracy_percentage:.1f}%"
        )
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Actual vs Predicted
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=y_test, 
            y=y_pred,
            mode='markers',
            marker=dict(
                size=10,
                color=y_test,
                colorscale='Purples',
                showscale=True,
                colorbar=dict(title="Actual Price")
            ),
            name='Predictions'
        ))
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        fig1.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Perfect Prediction'
        ))
        
        fig1.update_layout(
            title='Actual vs Predicted Prices',
            xaxis_title='Actual Price ($)',
            yaxis_title='Predicted Price ($)',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Residuals
        residuals = y_test - y_pred
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=y_pred,
            y=residuals,
            mode='markers',
            marker=dict(
                size=10,
                color=residuals,
                colorscale='RdBu',
                showscale=True,
                colorbar=dict(title="Residual")
            )
        ))
        
        fig2.add_hline(y=0, line_dash="dash", line_color="red")
        
        fig2.update_layout(
            title='Residual Plot',
            xaxis_title='Predicted Price ($)',
            yaxis_title='Residual ($)',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Feature importance
    st.markdown("### 🔑 Feature Importance")
    feature_importance = pd.DataFrame({
        'Feature': ['Size (sqft)', 'Bedrooms', 'Age (years)'],
        'Coefficient': model.coef_,
        'Abs_Coefficient': np.abs(model.coef_)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    fig3 = px.bar(
        feature_importance,
        x='Feature',
        y='Coefficient',
        color='Coefficient',
        color_continuous_scale='Purples',
        title='Feature Coefficients (Impact on Price)'
    )
    fig3.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig3, use_container_width=True)

with tab3:
    st.markdown("### 🔍 Dataset Exploration")
    
    # Dataset overview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### Sample Data")
        st.dataframe(df.head(10), use_container_width=True)
    
    with col2:
        st.markdown("#### Statistical Summary")
        st.dataframe(df.describe(), use_container_width=True)
    
    st.markdown("---")
    
    # Visualizations
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        # Price distribution
        fig4 = px.histogram(
            df, 
            x='Price', 
            nbins=30,
            color_discrete_sequence=['#667eea'],
            title='Price Distribution'
        )
        fig4.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig4, use_container_width=True)
    
    with viz_col2:
        # Size vs Price
        fig5 = px.scatter(
            df, 
            x='Size', 
            y='Price',
            color='Rooms',
            size='Age',
            color_continuous_scale='Purples',
            title='Size vs Price (colored by Rooms, sized by Age)'
        )
        fig5.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig5, use_container_width=True)
    
    # Correlation heatmap
    st.markdown("#### Feature Correlation")
    corr_matrix = df.corr()
    fig6 = px.imshow(
        corr_matrix,
        text_auto='.2f',
        color_continuous_scale='Purples',
        aspect='auto'
    )
    fig6.update_layout(
        title='Correlation Heatmap',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig6, use_container_width=True)

with tab4:
    st.markdown("### ℹ️ About This Application")
    
    st.markdown("""
    <div class="info-box">
        <h4>🎯 Purpose</h4>
        <p>This AI-powered application uses Machine Learning to predict house prices based on key features 
        such as size, number of bedrooms, and age of the property.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h4>🤖 Technology Stack</h4>
        <ul>
            <li><strong>Streamlit</strong> - Interactive web framework</li>
            <li><strong>Scikit-learn</strong> - Machine Learning library</li>
            <li><strong>Plotly</strong> - Interactive visualizations</li>
            <li><strong>Pandas</strong> - Data manipulation</li>
            <li><strong>NumPy</strong> - Numerical computing</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h4>📊 Model Information</h4>
        <p><strong>Algorithm:</strong> Multiple Linear Regression</p>
        <p><strong>Features:</strong> Size (sqft), Number of Bedrooms, Age (years)</p>
        <p><strong>Target:</strong> House Price ($)</p>
        <p><strong>Training Split:</strong> 70% train, 30% test</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h4>🔮 How to Use</h4>
        <ol>
            <li>Adjust the sliders in the sidebar to configure your house specifications</li>
            <li>The model will automatically predict the price</li>
            <li>Explore different tabs to see model performance and data insights</li>
            <li>Use the Data Explorer to understand the underlying dataset</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit & Machine Learning")

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)