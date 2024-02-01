import pandas
import matplotlib.pyplot

# Exercise 1: Data Loading and Inspection
#
# Load a CSV file called "data.csv" into a Pandas DataFrame. Then, perform the following tasks:
dataset=pandas.read_csv("data.csv")
print(dataset)
#
# Display the first 5 rows of the DataFrame.
for x in range(5):
    print(dataset.loc[x].iloc[1])
# Display the basic statistics (mean, median, etc.) for numeric columns.
print('Varsta mean and median: ',dataset["age"].mean(),' ',dataset["age"].median())
print('Salariu mean and median: ',dataset["salary"].mean(),' ',dataset["salary"].median())
print('Venit mean and median: ',dataset["income"].mean(),' ',dataset["salary"].median())

# ---------------------------------------------------------------------------------------------------------------------------
#
# Exercise 2: Data Selection and Filtering
#
# Using the DataFrame from Exercise 1, perform the following tasks:
# Select only the rows where the 'age' column is greater than 30.
print(dataset[dataset['age']>30])

# Filter the data to display only records where 'gender' is 'Female' and 'education' is 'Bachelor's Degree'.
options=["Bachelor's Degree"]
print(dataset[(dataset['gender']=="Female")])

#femeile nu au Bachelor's Degree, mai jos e un exemplu cu barbati pentru conditie multipla

print(dataset[(dataset['gender']=="Male")& (dataset['education'].isin(options))])

# Create a new DataFrame that includes only the 'name' and 'salary' columns.
df_nameSalary=dataset[['name','salary']]
print(df_nameSalary)

# ---------------------------------------------------------------------------------------------------------------------------
# Exercise 3: Data Aggregation and Grouping
#
# Using the same DataFrame, perform the following tasks:
#
# Group the data by 'gender' and calculate the average salary for each gender.
df_genderGroup=dataset.groupby('gender')
print('Average salary by gender groups',df_genderGroup['salary'].mean())

# Group the data by 'education' and find the maximum age for each education level.
df_educationGroup=dataset.groupby('education')
print('Maximum age by level of education \n',df_educationGroup.max())

# Calculate the total count of individuals for each combination of 'gender' and 'education'.
df_GenderEducationGroup=dataset.groupby(['gender','education'])
print('The total count of individuals for each combination of gender and education \n',df_GenderEducationGroup.count())

# ---------------------------------------------------------------------------------------------------------------------------
#
# Exercise 4: Data Cleaning and Transformation
#
# Load a new dataset or use the existing one from Exercise 1 and perform the following tasks:
#
# Remove duplicates from the dataset.
dataset_noDup=dataset.drop_duplicates()
print(dataset_noDup)
# Fill missing values in the 'income' column with the mean income.
dataset_filled=dataset.fillna(value=dataset['salary'].mean(),inplace=True)
print(dataset_filled)
# Create a new column 'age_group' that categorizes individuals into age groups (e.g., 'Under 30', '30-40', 'Over 40').
def get_age_group(age):
    if age <30:
        return "Under 30"
    elif age<=40:
        return "30-40"
    else:
        return "Over 40"

df_AgeGroupColumnAdded=dataset
df_AgeGroupColumnAdded['age_group']=df_AgeGroupColumnAdded['age'].apply(get_age_group)
print(df_AgeGroupColumnAdded)

# ---------------------------------------------------------------------------------------------------------------------------
#
# Exercise 5: Data Visualization with Pandas
#
# Load a dataset and use Pandas for visualization. Create the following plots:
#
# A histogram of the 'age' column.
dataset['age'].hist()
matplotlib.pyplot.show()

# A bar chart showing the count of each unique value in the 'education' column.
print(dataset.groupby('education').count())
dataset.groupby('education')['age'].count().plot(kind='bar')
matplotlib.pyplot.show()

# A scatter plot of 'age' vs. 'income' with different colors for 'gender'.
sp1=dataset[dataset['gender']=="Female"].plot.scatter(x='age',y='income',s=100,c='r')
sp2=dataset[dataset['gender']=="Male"].plot.scatter(x='age',y='income',s=100,c='b',ax=sp1)
matplotlib.pyplot.show()

# ---------------------------------------------------------------------------------------------------------------------------
#
# Exercise 6: Merging and Joining DataFrames
#
# Load two separate CSV files and perform the following tasks:
#
# Merge the two DataFrames using a common column (e.g., 'user_id').
dataset_PopLoc=pandas.read_csv("PopulatieLocalitati.csv")
dataset_Industrie=pandas.read_csv("Industrie.csv")
print(dataset_PopLoc)
print(dataset_Industrie)

# Join the DataFrames using a different column as the key.
dataset_PopLocInd=dataset_PopLoc.merge(dataset_Industrie,how='outer')
print(dataset_PopLocInd)

# Calculate the average 'score' for each 'user_id' after merging the DataFrames.
dataset_PopLocInd2=dataset_PopLoc.merge(dataset_Industrie,on='Localitate',how='outer')
print(dataset_PopLocInd2)

print(dataset_PopLocInd.groupby('Judet')['Siruta'].count())

