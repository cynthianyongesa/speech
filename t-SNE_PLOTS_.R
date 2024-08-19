
#install.packages("Rtsne")
# Load the libraries
library(readxl)
library(tidyverse)
library(dendextend)
library(RColorBrewer)
library(ggplot2)
library(Rtsne)

# Load the dataset
file_path <- "/Users/cynthianyongesa/Desktop/DATA/1_SPEECH/Clean_All_Data_Unique.xlsx"
df <- read_excel(file_path)

# Select only numeric columns for analysis, excluding specific columns
numeric_cols <- df %>% select(where(is.numeric)) %>%
  select(-c(Age, Education, Moca_MMSE, Number_of_Sentences)) %>%
  select(-contains("...1")) # Exclude the column with "...1" if it exists

# Calculate the correlation matrix
corr_matrix <- cor(numeric_cols, use = "complete.obs")

# Perform hierarchical clustering
dist_matrix <- as.dist(1 - corr_matrix)
hc <- hclust(dist_matrix, method = "complete")

# Cut the dendrogram to get 4 clusters
clusters <- cutree(hc, k = 4)

# Transpose the data for t-SNE and clustering consistency
transposed_numeric_cols <- t(numeric_cols)

# Perform t-SNE
set.seed(42) # for reproducibility
tsne <- Rtsne(transposed_numeric_cols, dims = 2, perplexity = 10, verbose = TRUE, max_iter = 1000)

# Get the t-SNE results
tsne_results <- as.data.frame(tsne$Y)

# Add the cluster assignments to the t-SNE results
tsne_results$cluster <- as.factor(clusters)

# Define colors for clusters
cluster_colors <- brewer.pal(4, "Set1")

# Plot the t-SNE results with convex hulls
ggplot(tsne_results, aes(x = V1, y = V2, color = cluster, shape = cluster)) +
  geom_point(size = 3, alpha = 0.7) +
  scale_color_manual(values = cluster_colors) +
  theme_minimal() +
  labs(title = "t-SNE of Clusters",
       x = "t-SNE Dimension 1",
       y = "t-SNE Dimension 2",
       color = "Cluster") +
  theme(legend.position = "bottom") +
  stat_ellipse(aes(fill = cluster), type = "t", alpha = 0.2, geom = "polygon") +
  guides(fill = FALSE, shape = guide_legend(override.aes = list(size = 4)))

# Save the t-SNE plot
ggsave("tSNE_Clusters_Improved.png", width = 10, height = 8, dpi = 300)