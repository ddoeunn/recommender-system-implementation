from utils.dataset import criteo

path = "C:\\Users\\DoeunKim\\Desktop"
data = criteo.load_data(unzip_path=path)
print(data.head())

