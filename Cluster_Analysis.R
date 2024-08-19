# Install required packages if not already installed
#install.packages("readxl")
#install.packages("corrplot")
#install.packages("tidyverse")
#install.packages("dendextend")
#install.packages("RColorBrewer")

# Load the libraries
library(readxl)
library(corrplot)
library(tidyverse)
library(dendextend)
library(RColorBrewer)

# Load the dataset
file_path <- "/Users/cynthianyongesa/Desktop/DATA/1_SPEECH/Clean_All_Data_Unique.xlsx"
df <- read_excel(file_path)

# Select only numeric columns for correlation analysis, excluding specific columns
numeric_cols <- df %>% select(where(is.numeric)) %>%
  select(-c(Age, Education, Moca_MMSE, Number_of_Sentences)) %>%
  select(-contains("...1")) # Exclude the column with "...1" if it exists

# Calculate the correlation matrix
corr_matrix <- cor(numeric_cols, use = "complete.obs")

# Perform hierarchical clustering
dist_matrix <- as.dist(1 - corr_matrix)
hc <- hclust(dist_matrix, method = "complete")

# Reorder the correlation matrix based on clustering results
ordered_corr_matrix <- corr_matrix[hc$order, hc$order]

# Define a custom color palette based on the provided image
color_palette <- colorRampPalette(c("#562a79", "#62b8d7", "#f3f1aa", "#de8b44", "#bc2a21"))(200)

# Create the correlogram with clustered features using corrplot
corrplot(ordered_corr_matrix, method = "color", 
         order = "hclust", 
         addrect = 4, # Add rectangles to highlight clusters
         col = color_palette,
         tl.col = "black", 
         tl.cex = 0.6, 
         tl.srt = 90, # Rotate text labels to vertical
         cl.cex = 0.8,
         main = "Multi-Pass Clustering of a Correlation Matrix")

# Save the plot
png("FINAL_Enhanced_Multi_Pass_Clustering_Correlation_Matrix.png", width = 1800, height = 1800, res = 300)
corrplot(ordered_corr_matrix, method = "color", 
         order = "hclust", 
         addrect = 4, # Add rectangles to highlight clusters
         col = color_palette,
         tl.col = "black", 
         tl.cex = 0.6, 
         tl.srt = 90, # Rotate text labels to vertical
         cl.cex = 0.8,
         main = "Enhanced Multi-Pass Clustering of a Correlation Matrix")
dev.off()

# Cut the dendrogram to get 4 clusters
clusters <- cutree(hc, k = 4)

# Create a data frame with variable names and their cluster assignments
cluster_assignments <- data.frame(Variable = names(numeric_cols), Cluster = clusters)

# List the variables in each cluster
cluster_list <- cluster_assignments %>% 
  group_by(Cluster) %>% 
  summarise(Variables = list(Variable)) %>% 
  pull(Variables)

# Print the variables in each cluster
for (i in 1:length(cluster_list)) {
  cat(paste("Cluster", i, ":\n"))
  cat(paste(cluster_list[[i]], collapse = ", "), "\n\n")
}