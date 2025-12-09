import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import paho.mqtt.client as mqtt

# Estilo bonito de gráficos
sns.set(style="whitegrid")

# Cargar dataset
df = pd.read_csv("dataset_domotica_final.csv")

df.head()

# Información general
df.info()

# Estadísticas numéricas
df.describe()

# Ver si hay valores nulos
df.isnull().sum()

##Distribución de cada variable en histogramas
df.hist(figsize=(18, 12), bins=30, color='skyblue', edgecolor='black')
plt.suptitle("Distribuciones de las Variables", fontsize=16)
plt.show()

##Detección de outliers
plt.figure(figsize=(18, 12))
sns.boxplot(data=df, palette="Set3")
plt.title("Boxplots de las Variables")
plt.xticks(rotation=45)
plt.show()

#Mapa de correlaciones
df_corr = df.drop(columns=["timestamp"])
plt.figure(figsize=(14, 10))
sns.heatmap(df_corr.corr(), annot=False, cmap='coolwarm', linewidths=0.5)
plt.title("Mapa de Calor de Correlaciones")
plt.show()

##Correlación de usuario en casa
df_corr.corr()['usuario_en_casa'].sort_values(ascending=False)

# Correlación de lluvia inminente
df_corr.corr()['lluvia_inminente'].sort_values(ascending=False)

#Temperatura interior vs Temperatura exterior
plt.figure(figsize=(8,5))
sns.scatterplot(x=df['T_exterior'], y=df['T_interior'], alpha=0.3)
plt.title("Relación entre Temperatura Exterior e Interior")
plt.xlabel("T Exterior (°C)")
plt.ylabel("T Interior (°C)")
plt.show()

#Relacion de la con la presencia del usuario de la casa
plt.figure(figsize=(10,5))
sns.lineplot(x=df['hora'], y=df['usuario_en_casa'])
plt.title("Patrón de Presencia en Casa por Hora")
plt.xlabel("Hora del Día")
plt.ylabel("En Casa (1/0)")
plt.show()

##Relacion de la frecuencia del encendido de la bomba con la humedad del suelo
plt.figure(figsize=(8,5))
sns.scatterplot(x=df['humedad_suelo'], y=df['bomba_on'], alpha=0.3)
plt.title("Relación entre Humedad del Suelo y Activación de la Bomba")
plt.xlabel("Humedad del Suelo (%)")
plt.ylabel("Bomba ON (0/1)")
plt.show()

sns.pairplot(df[['T_exterior','T_interior','H_exterior','H_interior','humedad_suelo','usuario_en_casa']],
             diag_kind="kde", corner=True)
plt.suptitle("Pairplot de Varias Variables", y=1.02)
plt.show()

"""##Clasificacion de los datos para predecir la presencia de los Usuarios

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap

#Cargar el dataset
df = pd.read_csv("dataset_domotica_final.csv")
df_corr = df.drop(columns=["timestamp"])

#Seleccionar variables X y Y
X = df_corr[['hora','dia_semana','T_interior','H_interior']]

y = df_corr['usuario_en_casa']

#Conjuntos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

#Escalado para usuarios
sc_usuario = StandardScaler()
X_train = sc_usuario.fit_transform(X_train)
X_test = sc_usuario.transform(X_test)

"""Modelo de regresion logistica"""

#Entrenamiento
from sklearn.linear_model import LogisticRegression

log_usuario = LogisticRegression()
log_usuario.fit(X_train, y_train)

#Prediccion
y_pred_log = log_usuario.predict(X_test)

#Metricas
cm_log = confusion_matrix(y_test, y_pred_log)
accuracy_log = accuracy_score(y_test, y_pred_log)

print("\n--- Matriz de Confusión (Regresión Logística) ---")
print(cm_log)
print(f"\nPrecisión: {accuracy_log*100:.2f}%")

#Curva ROC
y_prob_log = log_usuario.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_prob_log)
roc_auc_log = auc(fpr, tpr)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_log:.2f}", lw=2)
plt.plot([0,1],[0,1],'--')
plt.title("Curva ROC – Usuario en Casa (Regresión Logística)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.show()

#PCA para visualizar
pca = PCA(n_components=2)
X_train_2D = pca.fit_transform(X_train)
X_test_2D = pca.transform(X_test)

#Visualizacion del conjunto de entrenamiento
x_set, y_set = X_train_2D, y_train

x1, x2 = np.meshgrid(
    np.arange(x_set[:,0].min()-1, x_set[:,0].max()+1, 0.01),
    np.arange(x_set[:,1].min()-1, x_set[:,1].max()+1, 0.01)
)

plt.contourf(
    x1, x2,
    log_usuario.predict(pca.inverse_transform(np.c_[x1.ravel(), x2.ravel()]))
               .reshape(x1.shape),
    alpha=0.75, cmap=ListedColormap(('red','green'))
)

plt.scatter(x_set[:,0], x_set[:,1], c=y_set, cmap=ListedColormap(('red','green')))
plt.title("Regresión Logística – TRAIN (Usuario en Casa)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()

#Visualizacion del conjunto de prueba
x_set, y_set = X_test_2D, y_test

x1, x2 = np.meshgrid(
    np.arange(x_set[:,0].min()-1, x_set[:,0].max()+1, 0.01),
    np.arange(x_set[:,1].min()-1, x_set[:,1].max()+1, 0.01)
)

plt.contourf(
    x1, x2,
    log_usuario.predict(pca.inverse_transform(np.c_[x1.ravel(), x2.ravel()]))
               .reshape(x1.shape),
    alpha=0.75, cmap=ListedColormap(('red','green'))
)

plt.scatter(x_set[:,0], x_set[:,1], c=y_set, cmap=ListedColormap(('red','green')))
plt.title("Regresión Logística – TEST (Usuario en Casa)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()

def predecir_usuario_log(hora, dia,T_int,H_int):
    entrada = np.array([[hora, dia,T_int,H_int]])
    entrada_scaled = sc_usuario.transform(entrada)
    pred = log_usuario.predict(entrada_scaled)[0]

    print("Regresión Logística – Usuario en Casa:", pred)
    if pred == 1:
        print("El modelo predice: EL USUARIO ESTÁ EN CASA.")
    else:
        print("El modelo predice: EL USUARIO NO ESTÁ EN CASA.")

predecir_usuario_log(17, 3, 18,40)

"""Modelo 2: Bosques aleatorios"""

#Entrenamiento
from sklearn.ensemble import RandomForestClassifier

rf_usuario = RandomForestClassifier(
    n_estimators=90,
    criterion="entropy",
    random_state=0
)

rf_usuario.fit(X_train, y_train)

#Prediccion
y_pred_rf = rf_usuario.predict(X_test)
#Metricas
cm_rf = confusion_matrix(y_test, y_pred_rf)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

print("\n--- Matriz de Confusión (Random Forest) ---")
print(cm_rf)
print(f"\nPrecisión: {accuracy_rf*100:.2f}%")

#Curva ROC
y_prob_rf = rf_usuario.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_prob_rf)
roc_auc_rf = auc(fpr, tpr)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_rf:.2f}", lw=2, color='green')
plt.plot([0,1],[0,1],'--')
plt.title("Curva ROC – Usuario en Casa (Random Forest)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.show()

#Visualizacion del conjunto de entrenamiento
x_set, y_set = X_train_2D, y_train

x1, x2 = np.meshgrid(
    np.arange(x_set[:,0].min()-1, x_set[:,0].max()+1, 0.01),
    np.arange(x_set[:,1].min()-1, x_set[:,1].max()+1, 0.01)
)

plt.contourf(
    x1, x2,
    rf_usuario.predict(pca.inverse_transform(np.c_[x1.ravel(), x2.ravel()]))
              .reshape(x1.shape),
    alpha=0.75, cmap=ListedColormap(('red','green'))
)

plt.scatter(x_set[:,0], x_set[:,1], c=y_set, cmap=ListedColormap(('red','green')))
plt.title("Random Forest – TRAIN (Usuario en Casa)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()

#Visualizacion del conjunto de prueba
x_set, y_set = X_test_2D, y_test

x1, x2 = np.meshgrid(
    np.arange(x_set[:,0].min()-1, x_set[:,0].max()+1, 0.01),
    np.arange(x_set[:,1].min()-1, x_set[:,1].max()+1, 0.01)
)

plt.contourf(
    x1, x2,
    rf_usuario.predict(pca.inverse_transform(np.c_[x1.ravel(), x2.ravel()]))
              .reshape(x1.shape),
    alpha=0.75, cmap=ListedColormap(('red','green'))
)

plt.scatter(x_set[:,0], x_set[:,1], c=y_set, cmap=ListedColormap(('red','green')))
plt.title("Random Forest – TEST (Usuario en Casa)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()

def predecir_usuario_rf(hora, dia, T_int, H_int):
    entrada = np.array([[hora, dia,T_int, H_int]])
    entrada_scaled = sc_usuario.transform(entrada)
    pred = rf_usuario.predict(entrada_scaled)[0]

    print(" Random Forest – Usuario en Casa:", pred)
    if pred == 1:
        print(" El modelo predice: EL USUARIO ESTÁ EN CASA.")
    else:
        print("El modelo predice: EL USUARIO NO ESTÁ EN CASA.")

predecir_usuario_rf(12, 3, 18,40)

"""##Clasificacion para predecir lluvia"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap

#cargar dataset y seleccionar las variables
df = pd.read_csv("dataset_domotica_final.csv")
df_corr = df.drop(columns=["timestamp"])

X_lluvia = df_corr[['T_exterior','H_exterior','P_exterior','humedad_suelo']]

y_lluvia = df_corr['lluvia_inminente']

#Conjuntos
X_l_train, X_l_test, y_l_train, y_l_test = train_test_split(
    X_lluvia, y_lluvia, test_size=0.25, random_state=0
)

#Escalador para lluvia
sc_lluvia = StandardScaler()
X_l_train = sc_lluvia.fit_transform(X_l_train)
X_l_test  = sc_lluvia.transform(X_l_test)

"""Modelo1: Regresion Logistica"""

#Entrenamiento
from sklearn.linear_model import LogisticRegression

log_lluvia = LogisticRegression()
log_lluvia.fit(X_l_train, y_l_train)

#Prediccion
y_pred_log_l = log_lluvia.predict(X_l_test)

#Metricas
cm_log_l = confusion_matrix(y_l_test, y_pred_log_l)
accuracy_log_l = accuracy_score(y_l_test, y_pred_log_l)

print("\n--- Matriz de Confusión (Regresión Logística - Lluvia) ---")
print(cm_log_l)
print(f"\nPrecisión: {accuracy_log_l*100:.2f}%")

#Curva ROC
y_prob_log_l = log_lluvia.predict_proba(X_l_test)[:, 1]

fpr, tpr, _ = roc_curve(y_l_test, y_prob_log_l)
roc_auc_log_l = auc(fpr, tpr)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_log_l:.2f}", lw=2)
plt.plot([0,1],[0,1],'--')
plt.title("Curva ROC – Lluvia (Regresión Logística)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.show()

#PCA para la visualizaicion
pca_l = PCA(n_components=2)
X_l_train_2D = pca_l.fit_transform(X_l_train)
X_l_test_2D = pca_l.transform(X_l_test)

#Visualizacion del conjunto de entrenamiento
x_set, y_set = X_l_train_2D, y_l_train

x1, x2 = np.meshgrid(
    np.arange(x_set[:,0].min()-1, x_set[:,0].max()+1, 0.01),
    np.arange(x_set[:,1].min()-1, x_set[:,1].max()+1, 0.01)
)

plt.contourf(
    x1, x2,
    log_lluvia.predict(pca_l.inverse_transform(np.c_[x1.ravel(), x2.ravel()]))
              .reshape(x1.shape),
    alpha=0.75, cmap=ListedColormap(('red','green'))
)

plt.scatter(x_set[:,0], x_set[:,1], c=y_set, cmap=ListedColormap(('red','green')))
plt.title("Regresión Logística – TRAIN (Lluvia)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()

#Visualizacion del conjunto de prueba
x_set, y_set = X_l_test_2D, y_l_test

x1, x2 = np.meshgrid(
    np.arange(x_set[:,0].min()-1, x_set[:,0].max()+1, 0.01),
    np.arange(x_set[:,1].min()-1, x_set[:,1].max()+1, 0.01)
)

plt.contourf(
    x1, x2,
    log_lluvia.predict(pca_l.inverse_transform(np.c_[x1.ravel(), x2.ravel()]))
              .reshape(x1.shape),
    alpha=0.75, cmap=ListedColormap(('red','green'))
)

plt.scatter(x_set[:,0], x_set[:,1], c=y_set, cmap=ListedColormap(('red','green')))
plt.title("Regresión Logística – TEST (Lluvia)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()

def predecir_lluvia_log( T_ext, H_ext, P_ext,humedad_suelo):
    entrada = np.array([[ T_ext, H_ext, P_ext,humedad_suelo]])
    entrada_scaled = sc_lluvia.transform(entrada)
    pred = log_lluvia.predict(entrada_scaled)[0]

    print("Regresión Logística – Lluvia Inminente:", pred)
    if pred == 1:
        print(" Se predice: LLUVIA INMINENTE.")
    else:
        print(" Se predice: NO habrá lluvia.")

predecir_lluvia_log( 18, 78, 1009,30)

"""Modelo 2: Bosque aleatorios"""

#Entrenamiento
from sklearn.ensemble import RandomForestClassifier

rf_lluvia = RandomForestClassifier(
    n_estimators=90,
    criterion='entropy',
    random_state=0
)
rf_lluvia.fit(X_l_train, y_l_train)

#Prediccion
y_pred_rf_l = rf_lluvia.predict(X_l_test)

#Metricas
cm_rf_l = confusion_matrix(y_l_test, y_pred_rf_l)
accuracy_rf_l = accuracy_score(y_l_test, y_pred_rf_l)

print("\n--- Matriz de Confusión (Random Forest - Lluvia) ---")
print(cm_rf_l)
print(f"\nPrecisión: {accuracy_rf_l*100:.2f}%")

#Curva ROC
y_prob_rf_l = rf_lluvia.predict_proba(X_l_test)[:, 1]

fpr, tpr, _ = roc_curve(y_l_test, y_prob_rf_l)
roc_auc_rf_l = auc(fpr, tpr)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_rf_l:.2f}", lw=2, color='green')
plt.plot([0,1],[0,1],'--')
plt.title("Curva ROC – Lluvia (Random Forest)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.show()

#Visualizacion del conjunto de entrenamiento
x_set, y_set = X_l_train_2D, y_l_train

x1, x2 = np.meshgrid(
    np.arange(x_set[:,0].min()-1, x_set[:,0].max()+1, 0.01),
    np.arange(x_set[:,1].min()-1, x_set[:,1].max()+1, 0.01)
)

plt.contourf(
    x1, x2,
    rf_lluvia.predict(pca_l.inverse_transform(np.c_[x1.ravel(), x2.ravel()]))
              .reshape(x1.shape),
    alpha=0.75, cmap=ListedColormap(('red','green'))
)

plt.scatter(x_set[:,0], x_set[:,1], c=y_set, cmap=ListedColormap(('red','green')))
plt.title("Random Forest – TRAIN (Lluvia)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()

#Visualizacion del conjunto de prueba
x_set, y_set = X_l_test_2D, y_l_test

x1, x2 = np.meshgrid(
    np.arange(x_set[:,0].min()-1, x_set[:,0].max()+1, 0.01),
    np.arange(x_set[:,1].min()-1, x_set[:,1].max()+1, 0.01)
)

plt.contourf(
    x1, x2,
    rf_lluvia.predict(pca_l.inverse_transform(np.c_[x1.ravel(), x2.ravel()]))
              .reshape(x1.shape),
    alpha=0.75, cmap=ListedColormap(('red','green'))
)

plt.scatter(x_set[:,0], x_set[:,1], c=y_set, cmap=ListedColormap(('red','green')))
plt.title("Random Forest – TEST (Lluvia)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()

def predecir_lluvia_rf(T_ext, H_ext, P_ext,humedad_suelo):
    entrada = np.array([[T_ext, H_ext, P_ext,humedad_suelo]])
    entrada_scaled = sc_lluvia.transform(entrada)
    pred = rf_lluvia.predict(entrada_scaled)[0]

    print("Random Forest – Lluvia Inminente:", pred)
    if pred == 1:
        print("Se predice: LLUVIA INMINENTE.")
    else:
        print("Se predice: NO habrá lluvia.")

predecir_lluvia_rf(10, 88, 1009, 70)

"""##Guardar los modelos mas eficientes"""

import pickle

# Guardar modelo de USUARIO
with open("modelo_usuario.pkl", "wb") as f:
    pickle.dump(rf_usuario, f)

# Guardar escalador de USUARIO
with open("scaler_usuario.pkl", "wb") as f:
    pickle.dump(sc_usuario, f)

# Guardar modelo de LLUVIA
with open("modelo_lluvia.pkl", "wb") as f:
    pickle.dump(rf_lluvia, f)

# Guardar escalador de LLUVIA
with open("scaler_lluvia.pkl", "wb") as f:
    pickle.dump(sc_lluvia, f)

print("Modelos y escaladores guardados correctamente")


# ============================================================
#     ENVÍO DE PREDICCIONES A LA ESP32 POR MQTT
# ============================================================

BROKER = "192.168.1.19"
PORT = 1883
TOPIC_HOME = "casa/ml/home"
TOPIC_RAIN = "casa/ml/rain"

client = mqtt.Client()
client.connect(BROKER, PORT, 60)

# GENERAR PREDICCIÓN DE USUARIO
hora_demo = 17
dia_demo = 3
T_int_demo = 18
H_int_demo = 40

entrada_usuario = np.array([[hora_demo, dia_demo, T_int_demo, H_int_demo]])
entrada_usuario_scaled = sc_usuario.transform(entrada_usuario)
pred_home = int(rf_usuario.predict(entrada_usuario_scaled)[0])

print("\n>>> Predicción final (USUARIO EN CASA):", pred_home)

# GENERAR PREDICCIÓN DE LLUVIA
T_ext = 10
H_ext = 88
P_ext = 1009
hum_suelo = 70

entrada_lluvia = np.array([[T_ext, H_ext, P_ext, hum_suelo]])
entrada_lluvia_scaled = sc_lluvia.transform(entrada_lluvia)
pred_rain = int(rf_lluvia.predict(entrada_lluvia_scaled)[0])

print(">>> Predicción final (LLUVIA INMINENTE):", pred_rain)

# PUBLICAR A MQTT
client.publish(TOPIC_HOME, pred_home, retain=True)
client.publish(TOPIC_RAIN, pred_rain, retain=True)

print("\n*** Predicciones enviadas a la ESP32 por MQTT ***")
print(f"   {TOPIC_HOME} = {pred_home}")
print(f"   {TOPIC_RAIN} = {pred_rain}")

client.disconnect()


