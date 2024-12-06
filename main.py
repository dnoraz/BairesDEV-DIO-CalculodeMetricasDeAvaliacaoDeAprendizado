from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Gerar dados sintéticos
num_classes = 5
num_features = 10
num_samples = 1000
X, y = make_classification(
    n_samples=num_samples, n_features=num_features, n_informative=8, n_redundant=2,
    n_classes=num_classes, random_state=42
)

# Dividir os dados em treino (70%) e teste (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Escalonar os dados para que todas as características tenham média 0 e desvio padrão 1.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Criar e treinar o modelo SVM com hiperparâmetros definidos manualmente
clf = SVC(kernel='rbf', C=1, gamma='scale', class_weight='balanced')
clf.fit(X_train, y_train)

# Avaliar o modelo
y_pred = clf.predict(X_test)

# Métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Matriz de Confusão
cm = confusion_matrix(y_test, y_pred)

# Calcular a especificidade para cada classe e a média
specificities = []
for i in range(num_classes):
    tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])  # True Negatives
    fp = cm[:, i].sum() - cm[i, i]  # False Positives
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    specificities.append(specificity)

specificity_avg = np.mean(specificities)

# Exibir a matriz de confusão de forma gráfica
print("\nMatriz de Confusão:")
ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, display_labels=range(num_classes), cmap='Blues')
plt.title("Matriz de Confusão")  # Título do gráfico
plt.show()

# Exibir as métricas globais
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))
print(f"Acurácia: {accuracy:.4f}")
print(f"Precisão: {precision:.4f}")
print(f"Revocação (Sensibilidade): {recall:.4f}")
print(f"Especificidade Média: {specificity_avg:.4f}")
print(f"F1 Score: {f1:.4f}")
