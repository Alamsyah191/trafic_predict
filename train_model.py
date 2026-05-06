import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# 1. Koneksi ke Database
print("Mengambil data dari database...")
DB_URL = 'postgresql://postgres:Alamsyah_01@localhost:5432/traffic_db'
engine = create_engine(DB_URL)

try:
    query = "SELECT * FROM traffic_table"
    df = pd.read_sql(query, engine)
    print(f"Berhasil mengambil {len(df)} baris data.")
except Exception as e:
    print(f"Error: {e}")
    exit()

# 2. Preprocessing Data
print("Melakukan preprocessing data...")
df['DateTime'] = pd.to_datetime(df['DateTime'])
df['Hour'] = df['DateTime'].dt.hour
df['DayOfWeek'] = df['DateTime'].dt.dayofweek # 0=Senin, 6=Minggu

# Fitur yang digunakan untuk prediksi (X) dan target yang diprediksi (y)
# Kita memprediksi jumlah 'Vehicles' berdasarkan 'Junction', 'Hour', dan 'DayOfWeek'
features = ['Junction', 'Hour', 'DayOfWeek']
X = df[features]
y = df['Vehicles']

# Membagi data untuk training (80%) dan testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Melatih Model (Training)
print("Melatih model Random Forest...")
# n_estimators=100 berarti menggunakan 100 'pohon' keputusan
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluasi Model
print("Mengevaluasi model...")
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")
# Semakin kecil MSE, semakin baik. Targetnya biasanya < 3 seperti proyek filmmu.

# 5. Menyimpan Model
print("Menyimpan model ke file 'traffic_model.pkl'...")
joblib.dump(model, 'traffic_model.pkl')
print("Selesai! Model siap digunakan di app.py")