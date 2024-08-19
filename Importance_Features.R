# Load the necessary libraries
library(readxl)
library(tidyverse)
library(dendextend)
library(RColorBrewer)

# Load the dataset
file_path <- "/Users/cynthianyongesa/Desktop/DATA/1_SPEECH/Clean_All_Data_Unique.xlsx"
df <- read_excel(file_path)

# Select only numeric columns for analysis, excluding specific columns
numeric_cols <- df %>% select(where(is.numeric)) %>%
  select(-c(Age, Education, Moca_MMSE, Number_of_Sentences, Std_Sentence_Length)) %>%
  select(-contains("...1")) # Exclude the column with "...1" if it exists

# Calculate the correlation matrix
corr_matrix <- cor(numeric_cols, use = "complete.obs")

# Perform hierarchical clustering on the features
dist_matrix <- as.dist(1 - corr_matrix)
hc_features <- hclust(dist_matrix, method = "complete")

# Cut the dendrogram to get clusters for the features
feature_clusters <- cutree(hc_features, k = 4)

# Create a dataframe to store feature clusters
feature_clusters_df <- data.frame(Feature = names(feature_clusters), Cluster = feature_clusters)
feature_clusters_df <- feature_clusters_df %>% arrange(Cluster)

# Print the features in each cluster
print(feature_clusters_df)

# Prepare the data for feature importance analysis
df_selected <- df %>%
  filter(Diagnosis != "Other") %>%
  select(Diagnosis, one_of(feature_clusters_df$Feature)) %>%
  na.omit()

# Convert Diagnosis to factor
df_selected$Diagnosis <- as.factor(df_selected$Diagnosis)

# Set up the model using Random Forest to get feature importance
model <- train(Diagnosis ~ ., data = df_selected, method = "rf", importance = TRUE)

# Get the importance of each feature
importance <- varImp(model, scale = FALSE)
print(importance)

# Plot the top 20 important features
plot(importance, top = 20)