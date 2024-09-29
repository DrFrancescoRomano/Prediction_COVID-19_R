# Install necessary packages
# These libraries are required for data manipulation, modeling, and visualization
install.packages(c("dplyr", "caret", "readr", "yardstick", "pROC", "boot", "zoo", 
                   "ggplot2", "lubridate", "glmnet", "nnet", "xgboost", "corrplot", "Hmisc"))

# Load libraries
library(dplyr)
library(caret)
library(readr)
library(yardstick)
library(pROC)
library(corrplot)
library(Hmisc)  # P-value correlation
library(boot)
library(zoo)
library(ggplot2)
library(lubridate)
library(xgboost)
library(glmnet)  # Lasso and Ridge regression
library(nnet)    # Neural Network
library(scales)   
library(plotly)
library(ggthemes) 

# --------------------- Data Loading & Exploration ---------------------

# Load datasets
cases <- read_csv("WHO COVID-19 cases.csv", show_col_types = FALSE)
countries <- read_csv("countries.csv", show_col_types = FALSE)

# Basic data exploration
str(cases)        # View the structure of the dataset
dim(cases)        # Get the dimensions of the dataset
summary(cases)    # Summary statistics

# There are negative values in New_cases and New_Death
cases <- cases %>% filter(New_cases >=0 & New_deaths >= 0)

# Check for missing values in each column
missing_values <- colSums(is.na(cases))
print(missing_values)

# Remove rows with missing Country_code or WHO_region
cases <- cases %>%
  filter(!is.na(Country_code) & !is.na(WHO_region))

# --------------------- Data Cleaning & Transformation ---------------------

# Interpolation for New_cases and New_deaths
cases <- cases %>%
  arrange(Country_code, Date_reported) %>%
  group_by(Country_code) %>%
  mutate(New_cases = na.approx(New_cases, na.rm = FALSE),
         New_deaths = na.approx(New_deaths, na.rm = FALSE)) %>%
  ungroup() %>%
  mutate(New_cases = ifelse(is.na(New_cases), 0, New_cases),
         New_deaths = ifelse(is.na(New_deaths), 0, New_deaths))

# Recheck for missing values after cleaning
missing_values_after <- colSums(is.na(cases))
print(missing_values_after)

# Merge the 'cases' dataset with 'countries' for geographical coordinates
cases_with_coords <- cases %>%
  left_join(countries, by = c("Country_code" = "country")) %>%
  na.omit()

# Summary statistics after merge
summary(cases_with_coords)

# Create new date-based features
cases_cleaned <- cases_with_coords %>%
  mutate(Date_reported = as.Date(Date_reported),
         Date_year = year(Date_reported),
         Date_month = month(Date_reported),
         Date_day = day(Date_reported)) %>%
  select(-Country_code, -Date_reported)  # Remove unnecessary columns

# --------------------- Data Preparation for Modeling ---------------------

# Remove columns with too many categories (e.g., 'name' and 'WHO_region')
cases_1 <- cases_cleaned %>% select(-c(name, WHO_region))

# Correlation matrix for numeric columns
case_1Numeric <- cases_1 %>% select(where(is.numeric))
cases_1Corr <- cor(case_1Numeric)
print(cases_1Corr)

# Correlation matrix with p-values
rcorr_matrix_pvalue <- rcorr(as.matrix(case_1Numeric))
print(rcorr_matrix_pvalue)

# Plot the correlation matrix
corrplot(cases_1Corr)

# Final cleaned dataset
cases_cleaned <- cases_1 %>%
  select(-Country)  # Remove 'Country' column

# ---------------------- Charts ---------------------------------------

# Chart 1
p <- ggplot(cases_1, aes(x = as.Date(paste(Date_year, Date_month, Date_day, sep = "-")), y = New_cases)) +
  geom_line(color = "steelblue", size = 1.2) +       
  geom_smooth(method = "loess", se = FALSE, color = "red", size = 1, linetype = "dashed") +  
  labs(title = "New COVID-19 Cases Over Time",
       x = "Date", y = "New Cases") +    
  theme_minimal() +                      
  scale_x_date(date_labels = "%b %Y", date_breaks = "1 month") +  
  scale_y_continuous(labels = scales::comma) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))         

p_interactive <- ggplotly(p)
p_interactive


# Chart 2
library(maps)
world <- map_data("world")
p2 <- ggplot() +
  geom_polygon(data = world, aes(x = long, y = lat, group = group), fill = "lightblue", color = "white") +  # Mappa del mondo
  geom_point(data = cases_1, aes(x = longitude, y = latitude, color = New_cases, size = New_cases), alpha = 0.7) +  # Punti per i casi COVID-19 con colore e dimensione basati su New_cases
  scale_color_gradient(low = "yellow", high = "red", name = "New Cases", labels = scales::comma) +  # Scala colori dal giallo al rosso con numeri normali
  scale_size_continuous(range = c(1, 10), guide = "none") +  
  scale_y_continuous(labels = comma) +  
  labs(title = "New COVID-19 Cases by Location",
       x = "Longitude", y = "Latitude") +  
  theme_minimal() +  
  theme(plot.title = element_text(face = "bold", size = 16),
        axis.title.x = element_text(face = "bold"),
        axis.title.y = element_text(face = "bold"))

p2

# Chart 3
p3 <- ggplot(cases_1, aes(x = reorder(Country, Cumulative_cases), y = Cumulative_cases, fill = Continent)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  scale_y_log10() +  
  labs(title = "Total Cumulative Cases by Country", x = "Country", y = "Cumulative Cases") +
  theme(axis.text.y = element_text(size = 3))  
p3

# Chart 4
p4 <- ggplot(cases_1, aes(x = as.Date(paste(Date_year, Date_month, Date_day, sep = "-")), y = New_cases)) +
  geom_line() +
  facet_wrap(~ Country, scales = "free_y") +
  labs(title = "New COVID-19 Cases Over Time by Country", x = "Date", y = "New Cases") +
  theme(
    axis.text.y = element_text(size = 5),
    strip.text = element_text(size = 6)
  )

p4


# --------------------- Feature Encoding & Scaling ---------------------

# One-Hot Encoding
dummies <- dummyVars(~ ., data = cases_cleaned)
cases_cleaned <- predict(dummies, newdata = cases_cleaned)
cases_cleaned <- as.data.frame(cases_cleaned)

# Prepare the feature matrix (X) and target variable (y)
x <- cases_cleaned %>% select(-Cumulative_deaths)  # Features
y <- cases_cleaned$Cumulative_deaths  # Target

# Ensure that X is numeric
x_numeric <- x %>% select(where(is.numeric))
x_scaled <- as.data.frame(scale(x_numeric))

# Scale the target variable
y_scaled <- scale(y)

# Split the dataset into training and testing sets
set.seed(42)  # For reproducibility
train_index <- createDataPartition(y_scaled, p = 0.8, list = FALSE)
x_train <- x_scaled[train_index, ]
x_test <- x_scaled[-train_index, ]
y_train <- y_scaled[train_index]
y_test <- y_scaled[-train_index]

# --------------------- Model Training & Evaluation ---------------------

# Cross-validation setup
train_control <- trainControl(method = "cv", number = 5)

# --- Linear Regression ---
model_lm <- train(x_train, y_train, method = "lm", trControl = train_control)
predictions_lm <- predict(model_lm, newdata = x_test)
mse_lm <- mean((predictions_lm - y_test) ^ 2)
r_squared_lm <- 1 - sum((predictions_lm - y_test) ^ 2) / sum((y_test - mean(y_test)) ^ 2)

# --- Ridge Regression ---
grid_ridge <- expand.grid(alpha = 0, lambda = seq(0, 10, by = 0.1))
model_ridge <- train(x_train, y_train, method = "glmnet", trControl = train_control, tuneGrid = grid_ridge)
predictions_ridge <- predict(model_ridge, newdata = x_test)
mse_ridge <- mean((predictions_ridge - y_test) ^ 2)
r_squared_ridge <- 1 - mse_ridge / var(y_test)

# --- XGBoost Model ---
x_train_matrix <- as.matrix(x_train)
y_train_matrix <- as.matrix(y_train)
x_test_matrix <- as.matrix(x_test)
y_test_matrix <- as.matrix(y_test)

# Train the XGBoost model
xgb_model <- xgboost(
  data = x_train_matrix, 
  label = y_train_matrix, 
  nrounds = 500,                 
  max_depth = 3,                 
  eta = 0.01,             
  subsample = 0.8,               
  colsample_bytree = 0.8,        
  min_child_weight = 9,          
  lambda = 1,                    
  alpha = 0.1,                   
  objective = "reg:squarederror",
  eval_metric = "rmse",          
  early_stopping_rounds = 10,    
  verbose = 0
)
  
  # Predictions and RMSE calculation
  preds <- predict(xgb_model, x_test_matrix)
  rmse <- sqrt(mean((preds - y_test_matrix)^2))
  print(paste("RMSE:", rmse))
  
  # --------------------- Results & Comparison ---------------------
  
  # Create a results summary table
  results <- data.frame(
    Model = c("Linear Regression", "Ridge Regression", "XGBoost"),
    Train_MSE = c(mean((predict(model_lm, newdata = x_train) - y_train) ^ 2),
                  mean((predict(model_ridge, newdata = x_train) - y_train) ^ 2),
                  mean((predict(xgb_model, newdata = x_train_matrix) - y_train_matrix) ^ 2)),
    Test_MSE = c(mse_lm, mse_ridge, mean((preds - y_test_matrix)^2)),
    Train_R2 = c(r_squared_lm, r_squared_ridge, 1 - mean((predict(xgb_model, newdata = x_train_matrix) - y_train_matrix)^2) / var(y_train_matrix)),
    Test_R2 = c(1 - mse_lm / var(y_test), 
                1 - mse_ridge / var(y_test), 
                1 - mean((preds - y_test_matrix)^2) / var(y_test_matrix))
  )
  
  # Print the results
  cat("Results Summary:\n")
  print(results)
  
  # Evaluate overfitting by comparing train and test performance
  train_results <- data.frame(
    Model = c("Linear Regression", "Ridge Regression", "XGBoost"),
    Train_MSE = c(mean((predict(model_lm, newdata = x_train) - y_train) ^ 2),
                  mean((predict(model_ridge, newdata = x_train) - y_train) ^ 2),
                  mean((predict(xgb_model, newdata = x_train_matrix) - y_train_matrix) ^ 2)),
    Test_MSE = c(mse_lm, mse_ridge, mean((preds - y_test_matrix)^2))
  )
  
  # Print the train results to evaluate overfitting
  print(train_results)
  
  