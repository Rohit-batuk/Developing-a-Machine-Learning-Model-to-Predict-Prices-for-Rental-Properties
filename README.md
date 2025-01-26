# Developing-a-Machine-Learning-Model-to-Predict-Prices-for-Rental-Properties

## Introduction
The objective of this project is to predict the rental property prices using machine learning. More specifically, the goal is to develop a model that accurately forecasts the price of rental properties based on various features such as location, amenities, property size, and other factors.

Dataset: The dataset includes variables such as accommodates, bedrooms, review_scores_rating, room_type, and instant_bookable.


## Project Structure
data/: Contains raw and processed data files.
notebooks/: Jupyter notebooks for data analysis, model training, and evaluation.
scripts/: Python scripts for data preprocessing, model training, and deployment.
models/: Saved machine learning models.
reports/: Generated analysis reports and visualizations.
README.md: Project overview and instructions.


## Data Collection
The data for this project is collected from various sources, including:

Real estate listings
Property management databases
Market analysis reports
Key features collected:

Property location
Property size (square footage)
Number of bedrooms and bathrooms
Amenities
Age of the property
Historical price trends
Data Preprocessing


## Data Description:

   Initial Data:

   Rows: df.shape[0] (original count before preprocessing).

   Columns: df.shape[1] (e.g., log_price, room_type, cancellation_policy, etc.).


   Key steps:

   Dropped duplicates and the irrelevant id column.

   Encoded categorical variables (room_type, cancellation_policy, instant_bookable) using one-hot encoding.

  
  Missing Values:

  Three imputation strategies tested:

   Mean imputation: Filled missing values with column means.

   Median imputation: Filled missing values with column medians.

   KNN imputation: Used KNNImputer (neighbors=5) for multivariate imputation.

 
  Outliers Treatment:

  Treated using IQR method (capping values beyond 1.5×IQR).


## Methodology
   Tools: Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn).

   Preprocessing:

   Encoded categorical variables into dummy variables.

   Split data into training (70%) and testing (30%) sets.

   Model: Linear Regression.

   Evaluation Metric: R² (coefficient of determination).
    
 
     
## Analysis and Results
  
  Model Performance Across Strategies        
   ![Imputaion methods](https://github.com/user-attachments/assets/8d958447-885c-4e92-9440-f4d0c4db1561)



## Key Insights
 
  1  KNN imputation performed best, likely due to its ability to preserve relationships in the data.

  2  Outlier treatment reduced model performance, suggesting outliers (e.g., luxury properties) may reflect pricing trends.

  3  Feature Impact:

   Positive coefficients: accommodates (+0.062), bedrooms (+0.043), room_type_Private room (+0.076).

   Negative coefficients: instant_bookable_True (-0.021).


## Visualizations
    
  Heatmap: Highlighted strong correlations between log_price and features like accommodates and bedrooms.
  ![corelation](https://github.com/user-attachments/assets/b247f8f6-cf35-47af-b3a4-fdad556bd07b)


  

  Boxplots: Showed reduced variance after outlier treatment.
  ![with outlier](https://github.com/user-attachments/assets/36c19732-5827-40ac-848d-5a16222d48b8)

  ![without outlier](https://github.com/user-attachments/assets/224bc966-3e29-4fcd-a7c4-894eb5c27735)


       
## Conclusion.   

   Interpretation:

   Larger properties (accommodates, bedrooms) and private rooms command higher prices.

   Listings with instant_bookable enabled were slightly cheaper, possibly due to host incentives.

   Limitations:

   Linear Regression assumes linearity, which may not capture complex relationships.

   Data imputation methods may introduce bias.


##  Recommendations

   Optimize Pricing: Hosts with larger properties can justify higher prices.


## Appendix

   Code: Provided in the Jupyter notebook (Machine-Learning-Model-to-Predict-Prices-for-Rental-Properties.py).

   Data: Processed dataset derived from Air_BNB.xlsx.

