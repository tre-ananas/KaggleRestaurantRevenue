############### Kaggle: Restaurant Revenue Prediction
############### Ryan Wolff
############### 17 November 2023
############### Competition: https://www.kaggle.com/competitions/restaurant-revenue-prediction/overview
############### Other Data: https://www.kaggle.com/datasets/nyahmet/turkey-cities-populationincomeeducation-dataset/data





########## Load Data and Packages

# Load Packages
library(vroom) # Loading Data
library(tidymodels) # Modeling and cross validation
library(tidyverse) # Everything, really
library(DataExplorer) # EDA
library(patchwork) # Plots
library(GGally) # EDA
library(naivebayes) # Naive Bayes
library(discrim) # PCR
library(lubridate) # Dates
library(embed) # Extra recipe steps
library(workflows) # Workflows
library(bonsai)
library(lightgbm)

# Load Data
train <- vroom('train.csv')
test <- vroom('test.csv')
turkey <- vroom('turkey.csv') # This data set does not come from the competition but may provide useful features





########## Clean and Merge

# Clean Up Turkish Cities Data
turkey <- turkey %>%
  # Remove geographic features
  select(-latitude, -longitude, -`plate code`) %>%
  # Change feature name 'city' to 'City'
  rename(City = city) %>%
  # Remove dots in 'population' feature; convert to numeric
  mutate(population = as.numeric(gsub("\\.", "", population))) %>%
  # Rename 'per capita annual income' feature so it doesn't include a line break
  rename(`per capita annual income` = "
per capita annual income") %>%
  # Remove dots and 'TL' from 'per capita annual income' feature; convert to numeric
  mutate(`per capita annual income` = as.numeric(gsub(" TL", "", gsub("\\.", "", `per capita annual income`)))) %>%
  # Shorten name of 'number of people with higher education and above' feature
  rename(`percentage w higher ed or more` = "number of people with higher education and above") %>%
  # Remove % signs; convert to numeric
  mutate(`percentage w higher ed or more` = as.numeric(gsub("%", "", `percentage w higher ed or more`)))

# Merge Turkey Dataset with Training and Testing Data on 'City'
full_train <- merge(train, turkey, by = "City", all.x = TRUE)
full_test <- merge(test, turkey, by = "City", all.x = TRUE)





########## EDA

##### Variable Types

# Create Datasets for EDA
eda_train <- full_train
eda_test <- full_test

# Check Types
glimpse(eda_train)
glimpse(eda_test)

### Findings:
  # `Open Date` > date
  # City > factor
  # `City Group` > factor
  # Type > factor
  # Many of the Ps > factor

# Change Types for Non-P Features
eda_train$`Open Date` <- as.Date(eda_train$`Open Date`, format = "%m/%d/%Y")
eda_train$City <- as.factor(eda_train$City)
eda_train$`City Group` <- as.factor(eda_train$`City Group`)
eda_train$Type <- as.factor(eda_train$Type)

eda_test$`Open Date` <- as.Date(eda_test$`Open Date`, format = "%m/%d/%Y")
eda_test$City <- as.factor(eda_test$City)
eda_test$`City Group` <- as.factor(eda_test$`City Group`)
eda_test$Type <- as.factor(eda_test$Type)

# Visualize Types
plot_intro(eda_train)
plot_intro(eda_test)

### Findings:
  # No missing observations in full_train
  # Some missing observations in full_test

##### Data Summaries

# Check Data Summaries
skimr::skim(eda_train)
skimr::skim(eda_test)

### Findings:
  # 298 missing values for full_test in population, per capita annual income, and percentage w higher ed or more

##### Examine Missing Values

# Subset 'full_test' to include only rows with missing data in some column
incomplete_full_test <- full_test[!complete.cases(full_test), ]
incomplete_full_test

### Findings:
  # All rows with City = "Tanımsız" which means "undefined" are missing population, 
  # per capita annual income, and percentage w higher ed or more in full_test

##### Frequency and Density Charts
plot_bar(eda)
plot_density(eda)

### Findings:
  # revenue is right-skewed

##### Correlations
plot_correlation(eda)

### Findings:
  # chart is too cluttered and we don't know what
  # Ps represent

##### Factor Level Imbalances

# Types of Establishment
table(full_train$Type)
table(full_test$Type)

### Findings:
  # Mobile appears in testing but not training.
  # Only 1 drive through in training but a 2244 in testing.





########## Modeling: Linear Regression 1

##### Recipe

# Create Recipe
rec <- recipe(revenue ~ ., data = full_train) %>%
  # Remove ID; Remove Type because EDA showed some problems
  step_rm(Id, Type) %>%
  # Create a date column with proper format
  step_mutate(
    Open_Date = as.Date(`Open Date`, format = "%m/%d/%Y")
  ) %>%
  # Remove old date column
  step_rm(`Open Date`) %>%
  # Break down date into more specific features
  step_date(Open_Date, features = c('month', 'quarter')) %>%
  # Extract Season
  step_mutate(season = factor(case_when(
    between(month(Open_Date), 3, 5) ~ "Spring",
    between(month(Open_Date), 6, 8) ~ "Summer",
    between(month(Open_Date), 9, 11) ~ "Fall",
    TRUE ~ "Winter"
  )))  %>%
  # Remove Open_Date now that we have better features
  step_rm(Open_Date) %>%
  # Remove city names because there are new cities in the test data
  step_rm(City) %>%
  # Integers to factors
  step_mutate_at(all_integer_predictors(), fn = factor) %>%
  # Ignoring P-variables, turn appropriate variables into factors
  step_mutate_at(c('City Group'), fn = factor) %>%
  # Dummy factors
  step_dummy(all_nominal_predictors()) %>%
  # Median imputation for missing quantitative values
  step_impute_median(population) %>%
  step_impute_median(`per capita annual income`) %>%
  step_impute_median(`percentage w higher ed or more`) %>%
  # Normalize numeric features
  step_normalize(all_numeric_predictors()) %>%
  # Remove zero-variance variables
  step_zv()%>%
  # Remove features correlated more than .85
  step_corr(all_numeric_predictors(), threshold = 0.85) %>%
  # PCR Threshold = .85
  step_pca(all_numeric_predictors(), threshold = .85)
                 
# Prep, Bake, and View Recipe
prepped <- prep(rec)
baked <- bake(prepped, full_train) %>%
  slice(1:10)
baked

##### Modeling

# Set up model
linear_regression <- linear_reg() %>% # Type of model
  set_engine("lm") # Engine = What R function to use--linear model here

# Workflow
wf <- workflow() %>% 
  add_recipe(rec) %>%
  add_model(linear_regression) %>%
  fit(data = full_train) # Fit the workflow

# Look at fitted LM model
extract_fit_engine(wf) %>%
  tidy()
extract_fit_engine(wf) %>%
  summary

##### Predictions

# Predict Sales for Each ID in full_test
predictions <- bind_cols(full_test$Id, 
                              predict(wf, new_data = full_test)) %>% # Bind predictions to corresponding IDs
    rename("Id" = "...1", "Prediction" = ".pred")# Rename columns

# Write Predictions to .csv
# vroom_write(x=predictions, file="linear_regression_1_predictions.csv", delim = ",")





########## Modeling: Penalized Linear Regression

##### Recipe

# Create Recipe
rec <- recipe(revenue ~ ., data = full_train) %>%
  # Remove ID; Remove Type because EDA showed some problems
  step_rm(Id, Type) %>%
  # Create a date column with proper format
  step_mutate(
    Open_Date = as.Date(`Open Date`, format = "%m/%d/%Y")
  ) %>%
  # Remove old date column
  step_rm(`Open Date`) %>%
  # Break down date into more specific features
  step_date(Open_Date, features = c('month', 'quarter')) %>%
  # Extract Season
  step_mutate(season = factor(case_when(
    between(month(Open_Date), 3, 5) ~ "Spring",
    between(month(Open_Date), 6, 8) ~ "Summer",
    between(month(Open_Date), 9, 11) ~ "Fall",
    TRUE ~ "Winter"
  )))  %>%
  # Remove Open_Date now that we have better features
  step_rm(Open_Date) %>%
  # Remove city names because there are new cities in the test data
  step_rm(City) %>%
  # Integers to factors
  step_mutate_at(all_integer_predictors(), fn = factor) %>%
  # Ignoring P-variables, turn appropriate variables into factors
  step_mutate_at(c('City Group'), fn = factor) %>%
  # Dummy factors
  step_dummy(all_nominal_predictors()) %>%
  # Median imputation for missing quantitative values
  step_impute_median(population) %>%
  step_impute_median(`per capita annual income`) %>%
  step_impute_median(`percentage w higher ed or more`) %>%
  # Normalize numeric features
  step_normalize(all_numeric_predictors()) %>%
  # Remove zero-variance variables
  step_zv()%>%
  # Remove correlated features (> .85)
  step_corr(all_numeric_predictors(), threshold = 0.85) %>%
  # PCR Threshold = .85
  step_pca(all_numeric_predictors(), threshold = .85)
                 
# Prep, Bake, and View Recipe
prepped <- prep(rec)
baked <- bake(prepped, full_train) %>%
  slice(1:10)
baked

##### Modeling

# Set up model
penalized_linear_regression <- linear_reg(penalty = tune(), # Tune penalty and mixture
                                          mixture = tune()) %>%
  set_engine("glmnet") # use the glmnet function

# Set Up Workflow
wf <- workflow() %>% 
  add_recipe(rec) %>%
  add_model(penalized_linear_regression) %>%
  fit(data = full_train) # Fit the workflow

# Grid of values to tune over
tg <- grid_regular(penalty(),
                         mixture(),
                         levels = 20) # 20 levels to be really precise in tuning

# Split data for cross-validation (CV)
folds <- vfold_cv(full_train, v = 5, repeats = 1) # 5 folds

# Run cross-validation
cv_results <- wf %>%
  tune_grid(resamples = folds,
            grid = tg,
            metrics = metric_set(rmse))

# Find best tuning parameters to minimize RMSE
best_tune <- cv_results %>%
  select_best("rmse")

# Finalize workflow and fit it
final_wf <- wf %>%
  finalize_workflow(best_tune) %>%
  fit(data = full_train)

# Look at fitted LM model
extract_fit_engine(final_wf) %>%
  tidy()
extract_fit_engine(final_wf) %>%
  summary

##### Predictions

# Predict Sales for Each ID in full_test
predictions <- bind_cols(full_test$Id, 
                              predict(final_wf, new_data = full_test)) %>% # Bind predictions to corresponding IDs
    rename("Id" = "...1", "Prediction" = ".pred")# Rename columns

# Write Predictions to .csv
# vroom_write(x=predictions, file="penalized_linear_regression_1_predictions.csv", delim = ",")





########## Modeling: Random Forest 1

##### Recipe

# Create Recipe
rec <- recipe(revenue ~ ., data = full_train) %>%
  # Remove ID; Remove Type because EDA showed some problems
  step_rm(Id, Type) %>%
  # Create a date column with proper format
  step_mutate(
    Open_Date = as.Date(`Open Date`, format = "%m/%d/%Y")
  ) %>%
  # Remove old date column
  step_rm(`Open Date`) %>%
  # Break down date into more specific features
  step_date(Open_Date, features = c('month', 'quarter')) %>%
  # Extract Season
  step_mutate(season = factor(case_when(
    between(month(Open_Date), 3, 5) ~ "Spring",
    between(month(Open_Date), 6, 8) ~ "Summer",
    between(month(Open_Date), 9, 11) ~ "Fall",
    TRUE ~ "Winter"
  ))) %>%
  # Remove Open_Date now that we have better features
  step_rm(Open_Date) %>%
  # Remove city names because there are new cities in the test data
  step_rm(City) %>%
  # Integers to factors
  step_mutate_at(all_integer_predictors(), fn = factor) %>%
  # Ignoring P-variables, turn appropriate variables into factors
  step_mutate_at(c('City Group'), fn = factor) %>%
  # Dummy factors
  step_dummy(all_nominal_predictors()) %>%
  # Median imputation for missing quantitative values
  step_impute_median(population) %>%
  step_impute_median(`per capita annual income`) %>%
  step_impute_median(`percentage w higher ed or more`) %>%
  # Normalize numeric features
  step_normalize(all_numeric_predictors()) %>%
  # Remove zero-variance variables
  step_zv()%>%
  # Remove features correlated above .85
  step_corr(all_numeric_predictors(), threshold = 0.85) %>%
  # PCR Threshold = .85
  step_pca(all_numeric_predictors(), threshold = .85)
                 
# Prep, Bake, and View Recipe
prepped <- prep(rec)
baked <- bake(prepped, full_train)
head(baked, 3)

##### Modeling

# Set up model
random_forest <- rand_forest(mtry = tune(), # Tune mtry and min_n; keep trees at 1000
                              min_n = tune(),
                              trees = 1000) %>%
  set_engine("ranger") %>% # Use ranger function
  set_mode("regression") # Regression bc the target variable is quantitative

# Set Up Workflow
wf <- workflow() %>% 
  add_recipe(rec) %>%
  add_model(random_forest) %>%
  fit(data = full_train) # Fit the workflow

# Grid of values to tune over
tg <- grid_regular(mtry(range = c(1, 13)),
                            min_n(),
                            levels = 5)

# Split data for cross-validation (CV)
folds <- vfold_cv(full_train, v = 5, repeats = 1) # 5 folds for CV

# Run cross-validation
cv_results <- wf %>%
  tune_grid(resamples = folds,
            grid = tg,
            metrics = metric_set(rmse))

# Find best tuning parameters to minimize RMSE
best_tune <- cv_results %>%
  select_best("rmse")

# Finalize workflow and fit it
final_wf <- wf %>%
  finalize_workflow(best_tune) %>%
  fit(data = full_train)

##### Predictions

# Predict Sales for Each ID in full_test
predictions <- bind_cols(full_test$Id, 
                              predict(final_wf, new_data = full_test)) %>% # Bind predictions to corresponding IDs
    rename("Id" = "...1", "Prediction" = ".pred")# Rename columns

# Write Predictions to .csv
# vroom_write(x=predictions, file="random_forest_1_predictions.csv", delim = ",")





########## Modeling: Boosted Trees

##### Recipe

# Create Recipe
rec <- recipe(revenue ~ ., data = full_train) %>%
  # Remove ID; Remove Type because EDA showed some problems
  step_rm(Id, Type) %>%
  # Create a date column with proper format
  step_mutate(
    Open_Date = as.Date(`Open Date`, format = "%m/%d/%Y")
  ) %>%
  # Remove old date column
  step_rm(`Open Date`) %>%
  # Break down date into more specific features
  step_date(Open_Date, features = c('month', 'quarter')) %>%
  # Extract Season
  step_mutate(season = factor(case_when(
    between(month(Open_Date), 3, 5) ~ "Spring",
    between(month(Open_Date), 6, 8) ~ "Summer",
    between(month(Open_Date), 9, 11) ~ "Fall",
    TRUE ~ "Winter"
  )))  %>%
  # Remove Open_Date now that we have better features
  step_rm(Open_Date) %>%
  # Remove city names because there are new cities in the test data
  step_rm(City) %>%
  # Integers to factors
  step_mutate_at(all_integer_predictors(), fn = factor) %>%
  # Ignoring P-variables, turn appropriate variables into factors
  step_mutate_at(c('City Group'), fn = factor) %>%
  # Dummy factors
  step_dummy(all_nominal_predictors()) %>%
  # Median imputation for missing quantitative values
  step_impute_median(population) %>%
  step_impute_median(`per capita annual income`) %>%
  step_impute_median(`percentage w higher ed or more`) %>%
  # Normalize numeric features
  step_normalize(all_numeric_predictors()) %>%
  # Remove zero-variance variables
  step_zv()%>%
  # Remove correlated features (>,85)
  step_corr(all_numeric_predictors(), threshold = 0.85) %>%
  # PCR Threshold = .85
  step_pca(all_numeric_predictors(), threshold = .85)
                 
# Prep, Bake, and View Recipe
prepped <- prep(rec)
baked <- bake(prepped, full_train) %>%
  slice(1:10)
baked

##### Modeling

# Set up model
boosted_trees <- boost_tree(trees = 1000, # 1000 trees; Tune tree_depth and learn_rate
                       tree_depth = tune(), 
                       learn_rate = tune()) %>%
  set_engine("lightgbm") %>% # Use lightgbm function
  set_mode("regression") # Regression bc target variable is quantitative

# Set up Workflow
wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(boosted_trees)

# Grid of values to tune over
tg <- grid_regular(
  tree_depth(range = c(1, 5)),
  learn_rate(range = c(0, .25), trans=NULL),
  levels = 5
)

# Split data for cross-validation (CV)
folds <- vfold_cv(full_train, v = 5, repeats = 1) # 5 splits

# Run cross-validation
cv_results <- wf %>%
  tune_grid(resamples = folds,
            grid = tg,
            metrics = metric_set(rmse))

# Find best tuning parameters to minimize RMSE
best_tune <- cv_results %>%
  select_best("rmse")

# Finalize workflow and fit it
final_wf <- wf %>%
  finalize_workflow(best_tune) %>%
  fit(data = full_train)

##### Predictions

# Predict Sales for Each ID in full_test
predictions <- bind_cols(full_test$Id, 
                              predict(final_wf, new_data = full_test)) %>% # Bind predictions to corresponding IDs
    rename("Id" = "...1", "Prediction" = ".pred")# Rename columns

# Write Predictions to .csv
# vroom_write(x=predictions, file="boosted_trees_1_predictions.csv", delim = ",")





########## Modeling: Neural Network

##### Recipe

# Create Recipe
rec <- recipe(revenue ~ ., data = full_train) %>%
  # Remove ID; Remove Type because EDA showed some problems
  step_rm(Id, Type) %>%
  # Create a date column with proper format
  step_mutate(
    Open_Date = as.Date(`Open Date`, format = "%m/%d/%Y")
  ) %>%
  # Remove old date column
  step_rm(`Open Date`) %>%
  # Break down date into more specific features
  step_date(Open_Date, features = c('month', 'quarter')) %>%
  # Extract Season
  step_mutate(season = factor(case_when(
    between(month(Open_Date), 3, 5) ~ "Spring",
    between(month(Open_Date), 6, 8) ~ "Summer",
    between(month(Open_Date), 9, 11) ~ "Fall",
    TRUE ~ "Winter"
  )))  %>%
  # Remove Open_Date now that we have better features
  step_rm(Open_Date) %>%
  # Remove city names because there are new cities in the test data
  step_rm(City) %>%
  # Integers to factors
  step_mutate_at(all_integer_predictors(), fn = factor) %>%
  # Ignoring P-variables, turn appropriate variables into factors
  step_mutate_at(c('City Group'), fn = factor) %>%
  # Dummy factors
  step_dummy(all_nominal_predictors()) %>%
  # Median imputation for missing quantitative values
  step_impute_median(population) %>%
  step_impute_median(`per capita annual income`) %>%
  step_impute_median(`percentage w higher ed or more`) %>%
  # Normalize numeric features
  step_normalize(all_numeric_predictors()) %>%
  # Remove zero-variance variables
  step_zv()%>%
  # Remove correlated features (> .85)
  step_corr(all_numeric_predictors(), threshold = 0.85) %>%
  # PCR Threshold = .85
  step_pca(all_numeric_predictors(), threshold = .85) %>%
  # Scale Xs to [0, 1]
  step_range(all_numeric_predictors(), min = 0, max = 1)
                 
# Prep, Bake, and View Recipe
prepped <- prep(rec)
baked <- bake(prepped, full_train)
head(baked)

##### Modeling

# Set up model
neural_network <- mlp(hidden_units = tune(), # Tune hidden_units and epochs
                      epochs = tune()) %>%
  set_engine("nnet") %>% # Use nnet function
  set_mode('regression') # Regression bc target variable is quantitative

# Set Up Workflow
wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(neural_network)

# Grid of values to tune over
tg <- grid_regular(hidden_units(range = c(1, 10)),
                   epochs(range = c(0, 1000)),
                   levels = 10)

# Split data for cross-validation (CV)
folds <- vfold_cv(full_train, v = 5, repeats = 1) # 5 folds for CV

# Run cross-validation
cv_results <- wf %>%
  tune_grid(resamples = folds,
            grid = tg,
            metrics = metric_set(rmse))

# Find best tuning parameters to minimize RMSE
best_tune <- cv_results %>%
  select_best("rmse")
### Best: 2 hidden_units, 444 epochs

# Finalize workflow and fit it
final_wf <- wf %>%
  finalize_workflow(best_tune) %>%
  fit(data = full_train)

##### Predictions

# Predict Sales for Each ID in full_test
predictions <- bind_cols(full_test$Id, 
                              predict(final_wf, new_data = full_test)) %>% # Bind predictions to corresponding IDs
    rename("Id" = "...1", "Prediction" = ".pred")# Rename columns

# Write Predictions to .csv
# vroom_write(x=predictions, file="neural_network_1_predictions.csv", delim = ",")





########## Modeling: Gradient Boosted Trees

##### Recipe

# Create Recipe
rec <- recipe(revenue ~ ., data = full_train) %>%
  # Remove ID; Remove Type because EDA showed some problems
  step_rm(Id, Type) %>%
  # Create a date column with proper format
  step_mutate(
    Open_Date = as.Date(`Open Date`, format = "%m/%d/%Y")
  ) %>%
  # Remove old date column
  step_rm(`Open Date`) %>%
  # Break down date into more specific features
  step_date(Open_Date, features = c('month', 'quarter')) %>%
  # Extract Season
  step_mutate(season = factor(case_when(
    between(month(Open_Date), 3, 5) ~ "Spring",
    between(month(Open_Date), 6, 8) ~ "Summer",
    between(month(Open_Date), 9, 11) ~ "Fall",
    TRUE ~ "Winter"
  )))  %>%
  # Remove Open_Date now that we have better features
  step_rm(Open_Date) %>%
  # Remove city names because there are new cities in the test data
  step_rm(City) %>%
  # Integers to factors
  step_mutate_at(all_integer_predictors(), fn = factor) %>%
  # Ignoring P-variables, turn appropriate variables into factors
  step_mutate_at(c('City Group'), fn = factor) %>%
  # Dummy factors
  step_dummy(all_nominal_predictors()) %>%
  # Median imputation for missing quantitative values
  step_impute_median(population) %>%
  step_impute_median(`per capita annual income`) %>%
  step_impute_median(`percentage w higher ed or more`) %>%
  # Normalize numeric features
  step_normalize(all_numeric_predictors()) %>%
  # Remove zero-variance variables
  step_zv()%>%
  # Remove correlated features
  step_corr(all_numeric_predictors(), threshold = 0.85) %>%
  # PCR Threshold = .90
  step_pca(all_numeric_predictors(), threshold = .85)
                 
# Prep, Bake, and View Recipe
prepped <- prep(rec)
baked <- bake(prepped, full_train)
head(baked)

##### Modeling

# Set up model
gradient_boosted <- boost_tree(trees = 1000, # Keep trees at 1000 and tune tree_depth and min_n
                               tree_depth = tune(), 
                               min_n = tune(), learn_rate = tune()) %>%
  set_engine("xgboost") %>% # Use xgboost function
  set_mode("regression") # Regression bc target variable is quantitative

# Set Up Workflow
wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(gradient_boosted)

# Grid of values to tune over
tg <- grid_regular(
  tree_depth(range = c(1, 5)),
  min_n(range = c(1, 5)),
  learn_rate(range = c(0.01, 0.2), trans = NULL),
  levels = 5
)


# Split data for cross-validation (CV)
folds <- vfold_cv(full_train, v = 5, repeats = 1) # 5 folds for CV

# Run cross-validation
cv_results <- wf %>%
  tune_grid(resamples = folds,
            grid = tg,
            metrics = metric_set(rmse))

# Find best tuning parameters to minimize RMSE
best_tune <- cv_results %>%
  select_best("rmse")
# min_n = 1, tree_depth = 1, learn_rate = .01

# Finalize workflow and fit it
final_wf <- wf %>%
  finalize_workflow(best_tune) %>%
  fit(data = full_train)

##### Predictions

# Predict Sales for Each ID in full_test
predictions <- bind_cols(full_test$Id, 
                              predict(final_wf, new_data = full_test)) %>% # Bind predictions to corresponding IDs
    rename("Id" = "...1", "Prediction" = ".pred")# Rename columns

# Write Predictions to .csv
# vroom_write(x=predictions, file="gradient_boosted_trees_1_predictions.csv", delim = ",")





########## Modeling: Random Forest 2

##### Recipe

# Create Recipe
rec <- recipe(revenue ~ ., data = full_train) %>%
  # Remove ID; Remove Type because EDA showed some problems
  step_rm(Id, Type) %>%
  # Create a date column with proper format
  step_mutate(
    Open_Date = as.Date(`Open Date`, format = "%m/%d/%Y")
  ) %>%
  # Remove old date column
  step_rm(`Open Date`) %>%
  # Break down date into more specific features
  step_date(Open_Date, features = c('month', 'quarter')) %>%
  # Extract Season
  step_mutate(season = factor(case_when(
    between(month(Open_Date), 3, 5) ~ "Spring",
    between(month(Open_Date), 6, 8) ~ "Summer",
    between(month(Open_Date), 9, 11) ~ "Fall",
    TRUE ~ "Winter"
  )))  %>%
  # Remove Open_Date now that we have better features
  step_rm(Open_Date) %>%
  # Remove city names because there are new cities in the test data
  step_rm(City) %>%
  # Integers to factors
  step_mutate_at(all_integer_predictors(), fn = factor) %>%
  # Ignoring P-variables, turn appropriate variables into factors
  step_mutate_at(c('City Group'), fn = factor) %>%
  # Dummy factors
  step_dummy(all_nominal_predictors()) %>%
  # Median imputation for missing quantitative values
  step_impute_median(population) %>%
  step_impute_median(`per capita annual income`) %>%
  step_impute_median(`percentage w higher ed or more`) %>%
  # Normalize numeric features
  step_normalize(all_numeric_predictors()) %>%
  # Remove zero-variance variables
  step_zv()%>%
  # Remove correlated features above .90
  step_corr(all_numeric_predictors(), threshold = 0.90) %>%
  # PCR Threshold = .90
  step_pca(all_numeric_predictors(), threshold = .90)
                 
# Prep, Bake, and View Recipe
prepped <- prep(rec)
baked <- bake(prepped, full_train)
head(baked)

##### Modeling

# Set up model
random_forest <- rand_forest(mtry = tune(), # Tune metry and min_n and keep trees at 1000
                              min_n = tune(),
                              trees = 1000) %>% # Type of Model
  set_engine("ranger") %>% # Use ranger function
  set_mode("regression") # Regression bc target variable is quantitative

# Set Up Workflow
wf <- workflow() %>% 
  add_recipe(rec) %>%
  add_model(random_forest) %>%
  fit(data = full_train) # Fit the workflow

# Grid of values to tune over
tg <- grid_regular(mtry(range = c(1, 13)),
                            min_n(range = c(1, 10)),
                            levels = 10)

# Split data for cross-validation (CV)
folds <- vfold_cv(full_train, v = 5, repeats = 1) # 5 folds for CV

# Run cross-validation
cv_results <- wf %>%
  tune_grid(resamples = folds,
            grid = tg,
            metrics = metric_set(rmse))

# Find best tuning parameters to minimize RMSE
best_tune <- cv_results %>%
  select_best("rmse")
# mtry = 13, min_n = 10

# Finalize workflow and fit it
final_wf <- wf %>%
  finalize_workflow(best_tune) %>%
  fit(data = full_train)

##### Predictions

# Predict Sales for Each ID in full_test
predictions <- bind_cols(full_test$Id, 
                              predict(final_wf, new_data = full_test)) %>% # Bind predictions to corresponding IDs
    rename("Id" = "...1", "Prediction" = ".pred")# Rename columns

# Write Predictions to .csv
# vroom_write(x=predictions, file="random_forest_2_predictions.csv", delim = ",")





########## Modeling: Random Forest 3

##### Recipe

# Create Recipe
rec <- recipe(revenue ~ ., data = full_train) %>%
  # Remove ID; Remove Type because EDA showed some problems
  step_rm(Id, Type) %>%
  # Create a date column with proper format
  step_mutate(
    Open_Date = as.Date(`Open Date`, format = "%m/%d/%Y")
  ) %>%
  # Remove old date column
  step_rm(`Open Date`) %>%
  # Break down date into more specific features
  step_date(Open_Date, features = c('month', 'quarter')) %>%
  # Extract Season
  step_mutate(season = factor(case_when(
    between(month(Open_Date), 3, 5) ~ "Spring",
    between(month(Open_Date), 6, 8) ~ "Summer",
    between(month(Open_Date), 9, 11) ~ "Fall",
    TRUE ~ "Winter"
  )))  %>%
  # Remove Open_Date now that we have better features
  step_rm(Open_Date) %>%
  # Remove city names because there are new cities in the test data
  step_rm(City) %>%
  # Integers to factors
  step_mutate_at(all_integer_predictors(), fn = factor) %>%
  # Ignoring P-variables, turn appropriate variables into factors
  step_mutate_at(c('City Group'), fn = factor) %>%
  # Dummy factors
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(revenue)) %>%
  # Median imputation for missing quantitative values
  step_impute_median(population) %>%
  step_impute_median(`per capita annual income`) %>%
  step_impute_median(`percentage w higher ed or more`) %>%
  # Remove zero-variance variables
  step_zv()%>%
  # Normalize numeric features
  step_normalize(all_numeric_predictors()) %>%
  # Remove correlated features
  step_corr(all_numeric_predictors(), threshold = 0.85) %>%
  # PCR Threshold = .85
  step_pca(all_numeric_predictors(), threshold = .85)
                 
# Prep, Bake, and View Recipe
prepped <- prep(rec)
baked <- bake(prepped, full_train)
head(baked)

##### Modeling

# Set up model
random_forest <- rand_forest(mtry = tune(), # Tune mtry and min_n and use 1000 trees
                              min_n = tune(),
                              trees = 1000) %>% # Type of Model
  set_engine("ranger") %>% # Use ranger function
  set_mode("regression") # Regression bc target variable is quantitative

# Set Up Workflow
wf <- workflow() %>% 
  add_recipe(rec) %>%
  add_model(random_forest) %>%
  fit(data = full_train) # Fit the workflow

# Grid of values to tune over
tg <- grid_regular(mtry(range = c(1, 10)),
                            min_n(range(10, 100)),
                            levels = 10)

# Split data for cross-validation (CV)
folds <- vfold_cv(full_train, v = 5, repeats = 1) # 5 folds for CV

# Run cross-validation
cv_results <- wf %>%
  tune_grid(resamples = folds,
            grid = tg,
            metrics = metric_set(rmse))

# Find best tuning parameters to minimize RMSE
best_tune <- cv_results %>%
  select_best("rmse")
# mtry = 4; min_n = 60

# Finalize workflow and fit it
final_wf <- wf %>%
  finalize_workflow(best_tune) %>%
  fit(data = full_train)

##### Predictions

# Predict Sales for Each ID in full_test
predictions <- bind_cols(full_test$Id, 
                              predict(final_wf, new_data = full_test)) %>% # Bind predictions to corresponding IDs
    rename("Id" = "...1", "Prediction" = ".pred")# Rename columns

# Write Predictions to .csv
# vroom_write(x=predictions, file="random_forest_3_predictions.csv", delim = ",")





########## Modeling: Random Forest 4

##### Recipe

# Create Recipe
rec <- recipe(revenue ~ ., data = full_train) %>%
  # Remove ID; Remove Type because EDA showed some problems
  step_rm(Id, Type) %>%
  # Create a date column with proper format
  step_mutate(
    Open_Date = as.Date(`Open Date`, format = "%m/%d/%Y")
  ) %>%
  # Remove old date column
  step_rm(`Open Date`) %>%
  # Break down date into more specific features
  step_date(Open_Date, features = c('month', 'quarter')) %>%
  # Extract Season
  step_mutate(season = factor(case_when(
    between(month(Open_Date), 3, 5) ~ "Spring",
    between(month(Open_Date), 6, 8) ~ "Summer",
    between(month(Open_Date), 9, 11) ~ "Fall",
    TRUE ~ "Winter"
  )))  %>%
  # Remove Open_Date now that we have better features
  step_rm(Open_Date) %>%
  # Remove city names because there are new cities in the test data
  step_rm(City) %>%
  # Integers to factors
  step_mutate_at(all_integer_predictors(), fn = factor) %>%
  # Ignoring P-variables, turn appropriate variables into factors
  step_mutate_at(c('City Group'), fn = factor) %>%
  # Dummy factors
  step_dummy(all_nominal_predictors()) %>%
  # Median imputation for missing quantitative values
  step_impute_median(population) %>%
  step_impute_median(`per capita annual income`) %>%
  step_impute_median(`percentage w higher ed or more`) %>%
  # Normalize numeric features
  step_normalize(all_numeric_predictors()) %>%
  # Remove zero-variance variables
  step_zv()%>%
  # Remove correlated features (>.85)
  step_corr(all_numeric_predictors(), threshold = 0.85) %>%
  # PCR Threshold = .85
  step_pca(all_numeric_predictors(), threshold = .85)
                 
# Prep, Bake, and View Recipe
prepped <- prep(rec)
baked <- bake(prepped, full_train)
head(baked)

##### Modeling

# Set up model
random_forest <- rand_forest(mtry = tune(), # Tune mtry and min_n; Keep trees at 1000
                              min_n = tune(),
                              trees = 1000) %>% # Type of Model
  set_engine("ranger") %>% # Use ranger function
  set_mode("regression") # Regression bc target variable is quantitative

# Set Up Workflow
wf <- workflow() %>% 
  add_recipe(rec) %>%
  add_model(random_forest) %>%
  fit(data = full_train) # Fit the workflow

# Grid of values to tune over
tg <- grid_regular(mtry(range = c(1, 13)),
                            min_n(range = c(1, 200)),
                            levels = 13)

# Split data for cross-validation (CV)
folds <- vfold_cv(full_train, v = 5, repeats = 1) # 5 Folds fo CV

# Run cross-validation
cv_results <- wf %>%
  tune_grid(resamples = folds,
            grid = tg,
            metrics = metric_set(rmse))

# Find best tuning parameters to minimize RMSE
best_tune <- cv_results %>%
  select_best("rmse")
# mtry = 3, min_n = 100

# Finalize workflow and fit it
final_wf <- wf %>%
  finalize_workflow(best_tune) %>%
  fit(data = full_train)

##### Predictions

# Predict Sales for Each ID in full_test
predictions <- bind_cols(full_test$Id, 
                              predict(final_wf, new_data = full_test)) %>% # Bind predictions to corresponding IDs
    rename("Id" = "...1", "Prediction" = ".pred")# Rename columns

# Write Predictions to .csv
# vroom_write(x=predictions, file="random_forest_4_predictions.csv", delim = ",")





########## Modeling: Random Forest 5

##### Recipe

# Create Recipe
rec <- recipe(revenue ~ ., data = full_train) %>%
  # Remove ID; Remove Type because EDA showed some problems
  step_rm(Id, Type) %>%
  # Create a date column with proper format
  step_mutate(
    Open_Date = as.Date(`Open Date`, format = "%m/%d/%Y")
  ) %>%
  # Remove old date column
  step_rm(`Open Date`) %>%
  # Break down date into more specific features
  step_date(Open_Date, features = c('month', 'quarter')) %>%
  # Extract Season
  step_mutate(season = factor(case_when(
    between(month(Open_Date), 3, 5) ~ "Spring",
    between(month(Open_Date), 6, 8) ~ "Summer",
    between(month(Open_Date), 9, 11) ~ "Fall",
    TRUE ~ "Winter"
  )))  %>%
  # Remove Open_Date now that we have better features
  step_rm(Open_Date) %>%
  # Remove city names because there are new cities in the test data
  step_rm(City) %>%
  # Integers to factors
  step_mutate_at(all_integer_predictors(), fn = factor) %>%
  # Ignoring P-variables, turn appropriate variables into factors
  step_mutate_at(c('City Group'), fn = factor) %>%
  # Dummy factors
  step_dummy(all_nominal_predictors()) %>%
  # Median imputation for missing quantitative values
  step_impute_median(population) %>%
  step_impute_median(`per capita annual income`) %>%
  step_impute_median(`percentage w higher ed or more`) %>%
  # Normalize numeric features
  step_normalize(all_numeric_predictors()) %>%
  # Remove zero-variance variables
  step_zv()
                 
# Prep, Bake, and View Recipe
prepped <- prep(rec)
baked <- bake(prepped, full_train)
head(baked)

##### Modeling

# Set up model
random_forest <- rand_forest(mtry = tune(), # Tune mtry and min_n and set trees to 1000
                              min_n = tune(),
                              trees = 1000) %>% # Type of Model
  set_engine("ranger") %>% # Use ranger function
  set_mode("regression") # Regression bc target variable is quantitative

# Set Up Workflow
wf <- workflow() %>% 
  add_recipe(rec) %>%
  add_model(random_forest) %>%
  fit(data = full_train) # Fit the workflow

# Grid of values to tune over
tg <- grid_regular(mtry(range = c(1, 20)),
                            min_n(range = c(1, 400)),
                            levels = 20)

# Split data for cross-validation (CV)
folds <- vfold_cv(full_train, v = 5, repeats = 1) # 5 folds fo CV

# Run cross-validation
cv_results <- wf %>%
  tune_grid(resamples = folds,
            grid = tg,
            metrics = metric_set(rmse))

# Find best tuning parameters to minimize RMSE
best_tune <- cv_results %>%
  select_best("rmse")
# mtry = 3, min_n = 100

# Finalize workflow and fit it
final_wf <- wf %>%
  finalize_workflow(best_tune) %>%
  fit(data = full_train)

##### Predictions

# Predict Sales for Each ID in full_test
predictions <- bind_cols(full_test$Id, 
                              predict(final_wf, new_data = full_test)) %>% # Bind predictions to corresponding IDs
    rename("Id" = "...1", "Prediction" = ".pred")# Rename columns

# Write Predictions to .csv
# vroom_write(x=predictions, file="random_forest_5_predictions.csv", delim = ",")




########## Modeling: Random Forest 6

##### Recipe

# Create Recipe
rec <- recipe(revenue ~ ., data = full_train) %>%
  # Remove ID; Remove Type because EDA showed some problems; remove city names because there are new cities in the test data;
  # remove date because that seems not as important as the other features all things considered
  step_rm(Id, Type, `Open Date`, City) %>%
  # Integers to factors
  step_mutate_at(all_integer_predictors(), fn = factor) %>%
  # Ignoring P-variables, turn appropriate variables into factors
  step_mutate_at(c('City Group'), fn = factor) %>%
  # Dummy factors
  step_dummy(all_nominal_predictors()) %>%
  # Median imputation for missing quantitative values
  step_impute_median(population) %>%
  step_impute_median(`per capita annual income`) %>%
  step_impute_median(`percentage w higher ed or more`) %>%
  # Normalize numeric features
  step_normalize(all_numeric_predictors()) %>%
  # Remove zero-variance variables
  step_zv()%>%
  # Remove features correlated above .85
  step_corr(all_numeric_predictors(), threshold = 0.85) %>%
  # PCR Threshold = .85
  step_pca(all_numeric_predictors(), threshold = .85)
                 
# Prep, Bake, and View Recipe
prepped <- prep(rec)
baked <- bake(prepped, full_train)
head(baked, 3)

##### Modeling

# Set up model
random_forest <- rand_forest(mtry = tune(), # Tune mtry and min_n; Set trees at 1000
                              min_n = tune(),
                              trees = 1000) %>% # Type of Model
  set_engine("ranger") %>% # Use ranger function
  set_mode("regression") # Regression bc target variable is quantitative

# Set Up Workflow
wf <- workflow() %>% 
  add_recipe(rec) %>%
  add_model(random_forest) %>%
  fit(data = full_train) # Fit the workflow

# Grid of values to tune over
tg <- grid_regular(mtry(range = c(1, 13)),
                            min_n(),
                            levels = 5)

# Split data for cross-validation (CV)
folds <- vfold_cv(full_train, v = 5, repeats = 1) # 5 folds for CV

# Run cross-validation
cv_results <- wf %>%
  tune_grid(resamples = folds,
            grid = tg,
            metrics = metric_set(rmse))

# Find best tuning parameters to minimize RMSE
best_tune <- cv_results %>%
  select_best("rmse")

# Finalize workflow and fit it
final_wf <- wf %>%
  finalize_workflow(best_tune) %>%
  fit(data = full_train)

##### Predictions

# Predict Sales for Each ID in full_test
predictions <- bind_cols(full_test$Id, 
                              predict(final_wf, new_data = full_test)) %>% # Bind predictions to corresponding IDs
    rename("Id" = "...1", "Prediction" = ".pred")# Rename columns

# Write Predictions to .csv
# vroom_write(x=predictions, file="random_forest_6_predictions.csv", delim = ",")


