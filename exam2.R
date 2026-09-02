training <- read.csv("https://raw.githubusercontent.com/IAA-Faculty/statistical_foundations/master/tele_churn3.csv")

write.csv(training, "tele_churn3.csv", row.names = FALSE)

table(training$churn)
str(training)


library(earth)

mars1 <- earth(
  churn ~ total.intl.calls,
  data = training,
  glm = list(family = binomial)  # tells earth to fit a logistic GAM/MARS model
)

summary(mars1)

library(caret)
library(nnet)
library(dplyr)
y_train_s = training$churn
x_train <- training %>% select(-churn)
x_train_scaled <- scale(x_train)

# Ensure outcome is a factor with two levels
y_train_s <- factor(y_train_s, levels = c(0,1))

# Tuning grid
tune_grid <- expand.grid(
  size = c(3,5,7,9),
  decay = c(0,0.1,0.5,1)
)

# Cross-validation
ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

# caret requires the outcome levels to be character strings for twoClassSummary
levels(y_train_s) <- c("zero", "one")

set.seed(123)
nn.tuned <- train(
  x = x_train,
  y = y_train_s,
  method = "nnet",
  tuneGrid = tune_grid,
  trControl = ctrl,
  linout = FALSE,       # classification (logistic output)
  trace = FALSE
)

nn.tuned$bestTune

df = training

library(caret)
library(nnet)
library(dplyr)

#--------------------------------------------------
# 1. Load data (assume your data frame is named "df")
#--------------------------------------------------

# Convert logical churn to factor with two classes
df$churn <- factor(df$churn, levels = c(FALSE, TRUE))
levels(df$churn) <- c("no", "yes")

# Convert categorical variables to factor
df$international.plan <- factor(df$international.plan)
df$voice.mail.plan    <- factor(df$voice.mail.plan)

#--------------------------------------------------
# 2. Train/Test Split
#--------------------------------------------------
set.seed(123)
train_index <- createDataPartition(df$churn, p = 0.7, list = FALSE)

train <- df[train_index, ]
test  <- df[-train_index, ]

#--------------------------------------------------
# 3. Create Predictor and Outcome Matrices
#--------------------------------------------------
x_train <- train %>% select(-churn)
y_train <- train$churn

x_test  <- test %>% select(-churn)
y_test  <- test$churn

#--------------------------------------------------
# 4. Training Control for Classification Neural Net
#--------------------------------------------------
ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = TRUE
)

#--------------------------------------------------
# 5. Neural Network Tuning Grid
#--------------------------------------------------
tune_grid <- expand.grid(
  size = c(3,5),      # number of hidden neurons
  decay = c(0, 0.1)   # weight decay (L2 regularization)
)

#--------------------------------------------------
# 6. Train the Neural Network
#--------------------------------------------------
set.seed(123)
nn_model <- train(
  x = x_train,
  y = y_train,
  method = "nnet",
  tuneGrid = tune_grid,
  trControl = ctrl,
  preProcess = c("center", "scale"),  # standardize predictors
  linout = FALSE,                     # logistic activation for classification
  trace = FALSE
)

# Show tuning results and best model
nn_model
nn_model$bestTune

#--------------------------------------------------
# 7. Predict on Test Data
#--------------------------------------------------
# Probabilities
pred_prob <- predict(nn_model, newdata = x_test, type = "prob")[, "yes"]

# Class predictions
pred_class <- predict(nn_model, newdata = x_test)

#--------------------------------------------------
# 8. Evaluate Model Performance
#--------------------------------------------------
confusionMatrix(pred_class, y_test)

# Compute ROC/AUC
library(pROC)
roc_obj <- roc(y_test, pred_prob)
auc(roc_obj)

