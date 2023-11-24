"""
pip install..
flask
flask_wtf
numpy 
matplotlib 
sklearn

python -m venv env_ml
.\env_ml\Scripts\activate

pip install flask
pip install flask_wtf
pip install numpy
pip install sklearn
pip install scikit-learn
pip install matplotlib

python app.py

http://127.0.0.1:5000

deactivate
"""
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.datasets import load_iris
from io import BytesIO
import base64
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

app = Flask(__name__)

iris = load_iris()
X = iris.data
y = iris.target

# Divisão dos Dados em Treinamento e Teste.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Dicionário de classificadores
classifiers = {
    '---': '',
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(),
    'MLP': MLPClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

classifiers_params = {
    'KNN': ['n_neighbors', 'algorithm', 'leaf_size'],
    'SVM': ['degree', 'gamma', 'max_iter'],
    'MLP': ['random_state', 'max_iter', 'activation'],
    'Decision Tree': ['max_leaf_nodes', 'random_state', 'max_depth'],
    'Random Forest': ['n_estimators', 'random_state', 'max_depth']
}

# Rota principal
@app.route('/', methods=['GET', 'POST'])
def index():
    confusion_matrix_img = ""
    classification_results = {}

    if request.method == 'POST':
        classifier_name = request.form['classifier']
        params = {}

        for param_name in classifiers_params[classifier_name]:
            form_field = f"{classifier_name}_{param_name}"
            param_value = request.form.get(form_field)

            if param_value and (param_name.endswith('n_neighbors') or param_name.endswith('max_leaf_nodes') or param_name.endswith('n_estimators') or param_name.endswith('max_depth') or param_name.endswith('random_state') or param_name == 'leaf_size'):
                param_value = int(param_value)
            
            params[param_name] = param_value
        
        classifier = classifiers[classifier_name].set_params(**params)

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        classes = iris.target_names.tolist()
        confusion_matrix_img = plot_confusion_matrix(y_test, y_pred, classes)

        classification_results = {"accuracy": round(accuracy_score(y_test, y_pred), 3), "macro-avg": round(f1_score(y_test, y_pred, average='macro'), 3)}

    return render_template('index.html', classifiers=list(classifiers.keys()), classifier_params=classifiers_params, confusion_matrix_img=confusion_matrix_img, classification_results=classification_results)

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, xticks_rotation='vertical')
    
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode('utf-8')

if __name__ == '__main__':
    app.run(debug=True)
