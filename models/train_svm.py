import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import time

# 1. Učitavanje dataset-a
file_path = "../data/mhealth.csv"
data = pd.read_csv(file_path, header=None, low_memory=False)

# 2. Postavljanje naziva kolona
columns = ['alx', 'aly', 'alz', 'glx', 'gly', 'glz',
           'arx', 'ary', 'arz', 'grx', 'gry', 'grz',
           'Activity', 'subject']
data.columns = columns

# 3. Priprema podataka
# Uklanjanje kolone 'subject'
data = data.drop(['subject'], axis=1)

# Konverzija numeričkih vrednosti
for col in data.columns[:-1]:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Popunjavanje nedostajućih vrednosti sa 0
data = data.fillna(0)

# Enkodiranje ciljne promenljive ('Activity')
label_encoder = LabelEncoder()
data['Activity'] = label_encoder.fit_transform(data['Activity'])
activity_labels = [
    "Stajanje u mestu", "Sedenje i opuštanje", "Ležanje", "Hodanje",
    "Penjanje uz stepenice", "Savijanje struka unapred",
    "Podizanje ruku napred", "Savijanje kolena (čučnjevi)",
    "Vožnja bicikla", "Lagano trčanje", "Trčanje", "Skakanje napred i nazad"
]

# Korišćenje celog dataset-a
X = data.drop('Activity', axis=1).values
y = data['Activity'].values

# Skaliranje podataka
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Podela na trening i test skup
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Kreiranje i treniranje SVM modela
start_time = time.time()
model = SVC(kernel='rbf', C=1, gamma='scale')
model.fit(X_train, y_train)
end_time = time.time()

# 5. Evaluacija modela
y_pred = model.predict(X_test)

# Tačnost modela
accuracy = accuracy_score(y_test, y_pred)
print(f"Tačnost modela: {accuracy:.4f}")
print(f"Vreme izvršavanja: {end_time - start_time:.2f} sekundi")

# Izveštaj o klasifikaciji
print("\nIzveštaj o klasifikaciji:")
print(classification_report(y_test, y_pred, target_names=activity_labels, labels=np.arange(12)))

# 6. Konfuziona matrica
conf_matrix = confusion_matrix(y_test, y_pred, labels=np.arange(12))
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d",
            xticklabels=activity_labels, yticklabels=activity_labels)
plt.xlabel('Predikcija')
plt.ylabel('Prava klasa')
plt.title('Konfuziona matrica za SVM (100% uzoraka)')
plt.tight_layout()
plt.show()
