import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load a subset of the dataset for visualization
df = pd.read_csv("movies 1.csv", nrows=1000)  # Adjust nrows based on available memory
#Removed extra space in filename: " movies 1.csv" -> "movies 1.csv"
# Print the available columns to check for the correct column name
print(df.columns)

# Example: Distribution of movie ratings
# Replace 'rating' with the actual column name if it's different
rating_column_name = 'rating'  # Update with the correct column name if needed
if rating_column_name in df.columns:
    plt.figure(figsize=(10, 5))
    sns.histplot(df[rating_column_name], bins=20, kde=True)
    plt.xlabel("Movie Rating")
    plt.ylabel("Frequency")
    plt.title("Distribution of Movie Ratings")
    plt.show()
else:
    print(f"Error: Column '{rating_column_name}' not found in the DataFrame.")

# Example: Revenue vs. Budget scatter plot
# Similarly, make sure 'budget' and 'revenue' are the correct column names

# Check if 'budget' and 'revenue' columns exist, and if not, provide alternative column names
# Replace 'your_budget_column_name' and 'your_revenue_column_name' with the actual column names from your DataFrame
budget_column = 'budget' if 'budget' in df.columns else 'movieId'  # Replace 'movieId' with the actual budget column name if different
revenue_column = 'revenue' if 'revenue' in df.columns else 'genres'  # Replace 'genres' with the actual revenue column name if different


plt.figure(figsize=(10, 5))
sns.scatterplot(x=df[budget_column], y=df[revenue_column])  # Use the updated column names
plt.xlabel(budget_column)  # Update x-axis label
plt.ylabel(revenue_column)  # Update y-axis label
plt.title(f"{budget_column} vs. {revenue_column} Correlation")  # Update title
plt.show()