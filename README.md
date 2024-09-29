# Prediction_COVID-19_R
Data analysis and predictive modeling project based on WHO COVID-19 data. It includes data cleaning, handling missing values, and visualizing case trends. Regression models such as Linear Regression, Ridge Regression, and XGBoost were applied to predict cumulative deaths, with a performance comparison between the models.

# COVID-19 Data Analysis and Predictive Modeling

This repository contains an analysis of global COVID-19 data, along with various visualizations and predictive models. The data was obtained from the **World Health Organization (WHO)** and combined with geographical information to explore and predict COVID-19 cases and deaths across different countries.

## Table of Contents
- [Project Overview](#project-overview)
- [Data Sources](#data-sources)
- [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Visualization](#visualization)
- [Predictive Modeling](#predictive-modeling)
- [Model Performance](#model-performance)
- [Conclusion](#conclusion)
- [How to Run the Project](#how-to-run-the-project)

## Project Overview
The purpose of this project is to analyze global COVID-19 data, clean and preprocess it for use in machine learning models, and then build models to predict cumulative deaths using various regression techniques. We also visualize the trends of new cases and deaths over time across different countries.

## Data Sources
- **WHO COVID-19 Dataset:** Daily reported cases, cumulative cases, deaths, and cumulative deaths from the World Health Organization.
- **Geographical Dataset:** Coordinates (latitude, longitude) for each country, used for mapping and geographical analysis.

## Data Cleaning and Preprocessing
### Key steps:
1. **Handling Missing Values:** Rows with missing `Country_code` or `WHO_region` were removed. Missing values in `New_cases` and `New_deaths` were interpolated using linear interpolation.
2. **Filtering Data:** Negative values for `New_cases` and `New_deaths` were removed.
3. **Merging Datasets:** The cleaned WHO COVID-19 data was merged with geographical data for additional analysis.
4. **Feature Engineering:** New date-based features (`Date_year`, `Date_month`, `Date_day`) were created for better time-series analysis.

```r
# Data cleaning and merging
cases_cleaned <- cases_with_coords %>%
  mutate(Date_reported = as.Date(Date_reported),
         Date_year = year(Date_reported),
         Date_month = month(Date_reported),
         Date_day = day(Date_reported)) %>%
  select(-Country_code, -Date_reported)
```

## Exploratory Data Analysis (EDA)
### Basic Data Exploration
We conducted a basic exploration of the dataset to understand the structure, dimensions, and summary statistics. A correlation matrix was also calculated to identify relationships between numeric variables.
```r
# Summary of the cleaned dataset
summary(cases_cleaned)
```
### Correlation Matrix
```r
# Correlation matrix
correlation_matrix <- cor(cases_cleaned)
```
![Correlation Matrix](https://github.com/DrFrancescoRomano/Prediction_COVID-19_R/blob/main/images/CorrCOVID19.png)


## Visualization
Several visualizations were created to explore the trends in the data.

### 1. New COVID-19 Cases Over Time
A line plot showing the trend of new COVID-19 cases over time with a smoothed trendline.
```r
p <- ggplot(cases_cleaned, aes(x = as.Date(paste(Date_year, Date_month, Date_day, sep = "-")), y = New_cases)) +
  geom_line(color = "steelblue", size = 1.2) +       
  geom_smooth(method = "loess", se = FALSE, color = "red", size = 1, linetype = "dashed") +
  labs(title = "New COVID-19 Cases Over Time", x = "Date", y = "New Cases") +
  theme_minimal() +
  scale_x_date(date_labels = "%b %Y", date_breaks = "1 month") +
  scale_y_continuous(labels = scales::comma) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
```
!


### 2. COVID-19 Cases by Location (World Map)
A world map showing the geographical distribution of new COVID-19 cases, with color and size representing the number of new cases.
```r
p2 <- ggplot() +
  geom_polygon(data = world, aes(x = long, y = lat, group = group), fill = "lightblue", color = "white") +
  geom_point(data = cases_cleaned, aes(x = longitude, y = latitude, color = New_cases, size = New_cases), alpha = 0.7) +
  scale_color_gradient(low = "yellow", high = "red", name = "New Cases", labels = scales::comma) +
  labs(title = "New COVID-19 Cases by Location", x = "Longitude", y = "Latitude") +
  theme_minimal()
```
### 3. Total Cumulative Cases by Country
A bar chart showing the total cumulative COVID-19 cases for each country, sorted by the number of cases.
```r
p3 <- ggplot(cases_cleaned, aes(x = reorder(Country, Cumulative_cases), y = Cumulative_cases, fill = Continent)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  scale_y_log10() +
  labs(title = "Total Cumulative Cases by Country", x = "Country", y = "Cumulative Cases") +
  theme(axis.text.y = element_text(size = 3))
```

## Predictive Modeling
Three models were trained to predict cumulative deaths:

1. Linear Regression
2. Ridge Regression
3. XGBoost

## Model Performance
The performance of each model was evaluated using Mean Squared Error (MSE) and R-squared (R²) for both the training and test datasets.
```r
# Results summary
results <- data.frame(
  Model = c("Linear Regression", "Ridge Regression", "XGBoost"),
  Train_MSE = c(mse_lm, mse_ridge, mse_xgb),
  Test_MSE = c(test_mse_lm, test_mse_ridge, test_mse_xgb),
  Train_R2 = c(train_r2_lm, train_r2_ridge, train_r2_xgb),
  Test_R2 = c(test_r2_lm, test_r2_ridge, test_r2_xgb)
)
```

## Model Results
| Model              | Train MSE | Test MSE | Train R² | Test R² |
|--------------------|-----------|----------|----------|---------|
| Linear Regression  | 0.3671    | 0.4114   | 0.5851   | 0.5852  |
| Ridge Regression   | 0.3698    | 0.4235   | 0.5730   | 0.5730  |
| XGBoost            | 0.0308    | 0.0345   | 0.9693   | 0.9652  |

## Conclusion
In this analysis, we cleaned and explored the WHO COVID-19 dataset, visualized key trends, and developed predictive models to estimate cumulative deaths. The XGBoost model provided the best performance with high accuracy on both the training and test sets.



## Contact

If you have any questions or suggestions, feel free to contact me:

- **Name**: Francesco Romano
- **LinkedIn**: [Francesco Romano](https://www.linkedin.com/in/francescoromano03/)
