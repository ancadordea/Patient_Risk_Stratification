library(dplyr)
library(ggplot2)
library(ggcorrplot)
library(caTools)
library(caret)
library(tidyr)
library(nnet)

set.seed(42)

# Create synthetic dataset with 300 rows
patient_ID <- 1:300
age <- sample(18:90, 300, replace = TRUE)
gender <- sample(c("male", "female"), 300, replace = TRUE)
admission_date <- as.Date('2023/01/01') + sample(0:364, 300, replace = TRUE)
discharge_date <- admission_date + sample(1:30, 300, replace = TRUE)
diagnosis <- sample(c("COVID-19", "pneumonia", "bronchitis", "asthma"), 300, 
                    replace = TRUE)
treatment_cost <- round(runif(300, 1000, 10000), 2)
length_of_stay <- as.numeric(discharge_date - admission_date)

# Combine into data frame
patient_data <- data.frame(patient_ID, age, gender, 
                           admission_date, discharge_date, diagnosis, treatment_cost, 
                           length_of_stay)

write.csv(patient_data, "patient_data.csv", row.names = FALSE)

# Preprocessing

# Since dataset is sythetic, there should be no missing values 
sum(is.na(patient_data))
patient_data_clean <- na.omit(patient_data)

# Fix data types (on the cleaned data)
patient_data_clean <- patient_data_clean %>%
  mutate(
    gender = as.factor(gender),
    diagnosis = as.factor(diagnosis),
    admission_date = as.Date(admission_date),
    discharge_date = as.Date(discharge_date)
  )

# EDA
# Summary statistics for numerical variables
summary(patient_data)

# Summary statistics for categorical variables
table(patient_data$gender)
table(patient_data$diagnosis)

# Plot patient age
ggplot(patient_data, aes(x = age)) +
  geom_histogram(binwidth = 5, fill = "pink", color = "black") +
  labs(title = "Patient Age Distribution", x = "Age", y = "Frequency")

# Bar plot of diagnosis counts
ggplot(patient_data, aes(x = diagnosis, fill = diagnosis)) +
  geom_bar() +
  labs(title = "Diagnosis Frequency", x = "diagnosis", y = "Count") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Line chart of daily admissions
daily_admissions <- patient_data %>%
  group_by(admission_date) %>%
  summarize(Count = n())

ggplot(daily_admissions, aes(x = admission_date, y = Count)) +
  geom_line(color = "blue") +
  labs(title = "Daily Admissions Over Time", x = "Date", y = "Number of Admissions")

# Plot of Treatment Cost vs. Length of Stay
ggplot(patient_data, aes(x = length_of_stay, y = treatment_cost, color = diagnosis)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "lm", se = FALSE) +
  labs(title = "Treatment Cost vs. Length of Stay by Diagnosis",
       x = "Length of Stay (days)", y = "Treatment Cost ($)") +
  theme_minimal()

# Risk Stratification (Low / Medium / High)

# Add risk score - based on age, diagnosis and length of stay
patient_data_risk <- patient_data %>%
  mutate(
    risk_score = case_when(
      age >= 75 ~ 2,
      age >= 60 ~ 1,
      TRUE ~ 0
    ) +
      case_when(
        diagnosis %in% c("COVID-19", "pneumonia") ~ 2,
        diagnosis == "bronchitis" ~ 1,
        TRUE ~ 0
      ) +
      case_when(
        length_of_stay >= 14 ~ 2,
        length_of_stay >= 7 ~ 1,
        TRUE ~ 0
      )
  )

# Categorise into risk group
patient_data_risk <- patient_data_risk %>%
  mutate(
    risk_group = case_when(
      risk_score >= 5 ~ "High",
      risk_score >= 3 ~ "Medium",
      TRUE ~ "Low"
    )
  )

write.csv(patient_data_risk, "patient_data_with_risk_groups.csv", row.names = FALSE)

long_data <- patient_data_risk %>%
  select(age, length_of_stay, diagnosis, risk_group) %>%
  pivot_longer(cols = c(age, length_of_stay), names_to = "Feature", values_to = "Value")

# Plot features by risk group

# Numeric features boxplots 
long_data <- patient_data_risk %>%
  select(age, length_of_stay, diagnosis, risk_group) %>%
  pivot_longer(cols = c(age, length_of_stay), names_to = "Feature", values_to = "Value")

p1 <- ggplot(long_data, aes(x = risk_group, y = Value, fill = risk_group)) +
  geom_boxplot() +
  facet_wrap(~ Feature, scales = "free", ncol = 2) +
  labs(
    title = "Boxplots of Patient Features by Risk Group",
    x = "Risk Group", y = "Value", fill = "Risk Group"
  ) +
  theme_minimal()

# Diagnosis count by risk group
p2 <- patient_data_risk %>%
  ggplot(aes(x = risk_group, fill = diagnosis)) +
  geom_bar(position = "dodge") +
  labs(
    title = "Diagnosis Distribution by Risk Group",
    x = "Risk Group", y = "Count", fill = "Diagnosis"
  ) +
  theme_minimal()

# Print plots one after another
print(p1)
print(p2)

# Analysis - ML model
patient_data_risk$risk_group <- factor(patient_data_risk$risk_group, levels = c("Low", "Medium", "High"))

# Train/test split
set.seed(42)

split <- sample.split(patient_data_risk$risk_group, SplitRatio = 0.8)
train <- subset(patient_data_risk, split == TRUE)
test <- subset(patient_data_risk, split == FALSE)

# Train model
model <- multinom(risk_group ~ age + length_of_stay + diagnosis, data = train)

# Check summary
summary(model)

predictions <- predict(model, newdata = test)

# Evaluate model
confusion <- confusionMatrix(predictions, test$risk_group)
print(confusion)

# Export model data
# Extract the confusion matrix table
conf_matrix_df <- as.data.frame(confusion$table)
write.csv(conf_matrix_df, "confusion_matrix.csv", row.names = FALSE)

# Extract statistics by class
class_stats <- confusion$byClass
class_stats_df <- as.data.frame(class_stats)

# Add class names as a column (row names are class names)
class_stats_df$class <- rownames(class_stats_df)
rownames(class_stats_df) <- NULL

# Reorder columns to have class first
class_stats_df <- class_stats_df[, c(ncol(class_stats_df), 1:(ncol(class_stats_df)-1))]
write.csv(class_stats_df, "confusion_matrix_class_statistics.csv", row.names = FALSE)

# Extract overall statistics
overall_stats <- confusion$overall
overall_stats_df <- data.frame(Metric = names(overall_stats), Value = unname(overall_stats))
write.csv(overall_stats_df, "confusion_matrix_overall_statistics.csv", row.names = FALSE)
