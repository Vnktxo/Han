import kagglehub
from kagglehub import KaggleDatasetAdapter

# This path will depend on the file inside the dataset
file_path = "sign_mnist_train.csv"  # Example file

# Load dataset
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "datamunge/sign-language-mnist",
    file_path,
)

print("First 5 records:")
print(df.head())
