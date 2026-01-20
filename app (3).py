import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Page setup
st.set_page_config(page_title="Purchase Intent Prediction", layout="wide")
st.title("🛒 Customer Purchase Intent Analysis & Prediction")
st.write("### Using Random Forest to Classify User Behavior and Suggest Discounts")

# Load dataset from Hugging Face URL
@st.cache_data
def load_data(url):
    try:
        df = pd.read_csv(url, on_bad_lines='skip', encoding_errors='ignore')
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return None

csv_url = "https://huggingface.co/spaces/Saidee156/ConsumerElectronicsSales/resolve/main/consumer_electronics_sales_data.csv"
df = load_data(csv_url)

if df is None:
    st.stop()

# Column standardization
column_map = {
    'customer_age': 'CustomerAge',
    'age': 'CustomerAge',
    'product_category': 'ProductCategory',
    'category': 'ProductCategory',
    'product_brand': 'ProductBrand',
    'brand': 'ProductBrand',
    'customer_gender': 'CustomerGender',
    'gender': 'CustomerGender',
    'product_id': 'ProductID',
    'productid': 'ProductID',
    'intent': 'PurchaseIntent',
    'purchase_intent': 'PurchaseIntent',
    'product_price': 'ProductPrice'
}
df.rename(columns={col: column_map.get(col.lower(), col) for col in df.columns}, inplace=True)

# Required columns check
required_columns = ['CustomerAge', 'ProductCategory', 'ProductBrand', 'CustomerGender', 'ProductID', 'PurchaseIntent']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    st.error(f"Missing essential columns from dataset: {missing_columns}")
    st.stop()
    
# Success message and data type conversion
st.success("Dataset loaded and columns standardized!")
df['CustomerAge'] = df['CustomerAge'].astype(int)
df['CustomerGender'] = df['CustomerGender'].astype(int)
df['PurchaseIntent'] = df['PurchaseIntent'].astype(int)

# --- Feature Engineering & Preprocessing ---
# Store original columns for display in EDA
original_category_col = df['ProductCategory'].copy()
original_brand_col = df['ProductBrand'].copy()

le_category = LabelEncoder()
df['ProductCategory'] = le_category.fit_transform(df['ProductCategory'])
le_brand = LabelEncoder()
df['ProductBrand'] = le_brand.fit_transform(df['ProductBrand'])

# --- Model Training ---
X = df.drop(columns=['ProductID', 'PurchaseIntent'])
y = df['PurchaseIntent']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# RandomForest with GridSearchCV (Cached to avoid re-running)
@st.cache_resource
def train_model(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]}
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_

best_model, best_params = train_model(X_train, y_train)
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
# ---------------------------------------------

# --- EDA Functions ---
def run_univariate_analysis(df, price_col):
    st.subheader("📊 Univariate Analysis: Single Feature Distribution")

    # Age Distribution
    col_age, col_price = st.columns(2)
    with col_age:
        st.markdown("##### Customer Age Distribution")
        fig_age, ax_age = plt.subplots(figsize=(6, 3))
        sns.histplot(df['CustomerAge'], bins=20, kde=True, ax=ax_age)
        ax_age.set_title("Age Distribution")
        st.pyplot(fig_age)

    if price_col in df.columns:
        # Price Distribution
        with col_price:
            st.markdown("##### Product Price Distribution")
            fig_price, ax_price = plt.subplots(figsize=(6, 3))
            sns.histplot(df[price_col], bins=30, kde=True, ax=ax_price)
            ax_price.set_title("Product Price Distribution")
            st.pyplot(fig_price)
    
    col_intent, col_gender = st.columns(2)
    # Intent Distribution
    with col_intent:
        st.markdown("##### Purchase Intent Distribution")
        fig_intent, ax_intent = plt.subplots(figsize=(6, 3))
        sns.countplot(x='PurchaseIntent', data=df, ax=ax_intent)
        ax_intent.set_title("Intent Counts (0:Low, 1:Medium, 2:High)")
        st.pyplot(fig_intent)
    
    # Gender Distribution
    with col_gender:
        st.markdown("##### Customer Gender Distribution")
        fig_gender, ax_gender = plt.subplots(figsize=(6, 3))
        sns.countplot(x='CustomerGender', data=df, ax=ax_gender)
        ax_gender.set_title("Gender Counts (0:Female, 1:Male - Example)")
        st.pyplot(fig_gender)
    
def run_bivariate_analysis(df_original, price_col):
    st.subheader("🔗 Bivariate Analysis: Feature Relationships")

    col_age_intent, col_price_intent = st.columns(2)
    
    # Age vs Intent (Categorical vs Numerical)
    with col_age_intent:
        st.markdown("##### Purchase Intent by Customer Age")
        fig_age_intent, ax_age_intent = plt.subplots(figsize=(6, 4))
        sns.boxplot(x='PurchaseIntent', y='CustomerAge', data=df_original, ax=ax_age_intent)
        ax_age_intent.set_title("Age vs. Intent")
        st.pyplot(fig_age_intent)

    if price_col in df_original.columns:
        # Price vs Intent (Numerical vs Categorical)
        with col_price_intent:
            st.markdown("##### Product Price vs Purchase Intent")
            fig_price_intent, ax_price_intent = plt.subplots(figsize=(6, 4))
            sns.boxplot(x='PurchaseIntent', y=price_col, data=df_original, ax=ax_price_intent)
            ax_price_intent.set_title("Price vs. Intent")
            st.pyplot(fig_price_intent)
        
    # Category vs Intent (Categorical vs Categorical)
    st.markdown("##### Purchase Intent by Product Category")
    category_df = df_original.copy()
    category_df['ProductCategory'] = original_category_col # Use original string labels
    
    fig_cat_intent, ax_cat_intent = plt.subplots(figsize=(10, 6))
    sns.countplot(x='ProductCategory', hue='PurchaseIntent', data=category_df, ax=ax_cat_intent, palette='viridis')
    ax_cat_intent.set_title("Purchase Intent Counts per Product Category")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Intent', labels=['Low (0)', 'Medium (1)', 'High (2)'])
    plt.tight_layout()
    st.pyplot(fig_cat_intent)
        

# --- Streamlit Tabs for Organized Output ---
tab1, tab2, tab3 = st.tabs(["📊 Data Exploration (EDA)", "📈 Model Evaluation", "🚀 Prediction Interface"])

# --- TAB 1: Data Exploration ---
with tab1:
    st.header("1. Data Exploration and Visualization")
    
    st.subheader("Dataset Snapshot")
    st.dataframe(df.head())
    st.write(f"**Dataset Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
    
    st.markdown("---")
    
    # Run EDA functions
    run_univariate_analysis(df, 'ProductPrice')
    st.markdown("---")
    run_bivariate_analysis(df, 'ProductPrice')
    st.markdown("---")
    
    st.subheader("Correlation Heatmap (All Numerical Features)")
    numeric_df = df.select_dtypes(include=['number'])
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    st.pyplot(fig)
    
    # Product Price Filter
    with st.expander("🔍 Filter Products by Price Range"):
        if 'ProductPrice' in df.columns:
            min_price = float(df['ProductPrice'].min())
            max_price = float(df['ProductPrice'].max())
            price_range = st.slider("Select a price range (₹)", min_value=min_price, max_value=max_price, value=(min_price, max_price))
            filtered_df = df[(df['ProductPrice'] >= price_range[0]) & (df['ProductPrice'] <= price_range[1])].copy()
            filtered_df['DecodedBrand'] = le_brand.inverse_transform(filtered_df['ProductBrand'])
            st.write(f"**Total Products:** {len(filtered_df)} priced between ₹{price_range[0]:.2f} and ₹{price_range[1]:.2f}")
            st.dataframe(filtered_df[['ProductID', 'DecodedBrand', 'ProductCategory', 'ProductPrice']].reset_index(drop=True).head(10))
        else:
            st.warning("The dataset does not contain a 'ProductPrice' column for filtering.")

# --- TAB 2: Model Evaluation ---
with tab2:
    st.header("2. Model Evaluation Results")

    col_acc, col_params = st.columns(2)
    with col_acc:
        st.metric(label="Model Accuracy on Test Set", value=f"{accuracy:.2f}", delta="Random Forest")
    with col_params:
        st.subheader("Best Hyperparameters (Grid Search)")
        st.json(best_params)
        
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

    col_cm, col_fi = st.columns(2)
    
    with col_cm:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        cm = confusion_matrix(y_test, y_pred)
        
        # FIX FOR ValueError: Use all unique labels from the full target 'y'
        unique_labels = sorted(y.unique()) 
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
        disp.plot(ax=ax, cmap=plt.cm.Blues)
        st.pyplot(fig)
    
    with col_fi:
        st.subheader("Feature Importance")
        # Feature Importance Plot
        importances = best_model.feature_importances_
        feature_names = X.columns
        feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)
        
        fig_fi, ax_fi = plt.subplots(figsize=(8, 5))
        sns.barplot(x=feature_importances, y=feature_importances.index, ax=ax_fi, palette="viridis")
        ax_fi.set_title("Feature Importance from Random Forest")
        ax_fi.set_xlabel("Importance Score")
        st.pyplot(fig_fi)
        
# --- TAB 3: Prediction Interface ---
with tab3:
    st.header("3. Predict Purchase Intent & Suggest Discount")
    
    # Define columns for input layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Changed min/max/value to float to be consistent with step=1.0
        age_input = st.number_input("Customer Age", min_value=float(df['CustomerAge'].min()), max_value=float(df['CustomerAge'].max()), value=float(df['CustomerAge'].mean()), step=1.0)
        category_input = st.selectbox("Product Category (Original)", list(le_category.classes_))
        category_encoded = le_category.transform([category_input])[0]
        
    with col2:
        brand_input = st.selectbox("Product Brand (Original)", list(le_brand.classes_))
        brand_encoded = le_brand.transform([brand_input])[0]
        # Include all other required numerical inputs
        other_inputs = {}
        for col in X.columns:
            if col not in ['CustomerAge', 'ProductCategory', 'ProductBrand']:
                default_val = df[col].mean()
                
                # FIX IMPLEMENTED HERE: Check data type to ensure value and step are consistent
                if df[col].dtype in ['int64', 'int32'] and df[col].nunique() < 10:
                    # Treat low-cardinality integers (like gender) as discrete steps
                    input_step = 1
                    input_value = int(default_val)
                    input_format = "%d"
                else:
                    # Treat continuous or high-cardinality values (like price) as floats
                    input_step = 1.0 
                    input_value = default_val 
                    input_format = "%.2f"
                    
                other_inputs[col] = st.number_input(
                    f"{col} (Encoded)", 
                    value=input_value, 
                    step=input_step, 
                    format=input_format,
                    help=f"Encoded or Numerical value for {col}"
                )

    # Prepare input for prediction
    user_inputs_list = [age_input, category_encoded, brand_encoded]
    for col in X.columns:
        if col not in ['CustomerAge', 'ProductCategory', 'ProductBrand']:
            user_inputs_list.append(other_inputs[col])

    # Ensure input list order matches X.columns order exactly
    input_df = pd.DataFrame([user_inputs_list], columns=X.columns)
    user_inputs_scaled = scaler.transform(input_df)

    if st.button("🔮 **Predict Purchase Intent**", type="primary"):
        prediction = best_model.predict(user_inputs_scaled)
        intent_mapping = {2: "High", 1: "Medium", 0: "Low"}
        
        predicted_label = intent_mapping.get(prediction[0], "Unknown")
        
        # Discount logic: Aggressive offers
        discount_suggestions = {
            2: {"markdown": "### 🎉 **10% off - Limited Offer**", "emoji": "💰", "color": "green", "text": "This customer is highly likely to purchase. A moderate, limited-time discount should close the sale quickly."},
            1: {"markdown": "### 🎁 **15% off - Valued Customer Deal**", "emoji": "🛍️", "color": "orange", "text": "The purchase intent is moderate. A substantial discount will significantly increase conversion probability."},
            0: {"markdown": "### 🔥 **30% off - Exclusive Flash Sale**", "emoji": "🚨", "color": "red", "text": "The intent is low. A much larger, exclusive discount is required to overcome resistance and convert the user."}
        }
        
        result = discount_suggestions[prediction[0]]

        st.markdown("---")
        st.subheader("Prediction Result")
        st.metric(label="Predicted Purchase Intent", value=f"{predicted_label}", delta=f"Class {prediction[0]}")
        
        st.markdown(f"**Suggested Strategy:** {result['emoji']} {result['markdown']}")
        st.info(result['text'])