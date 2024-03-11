
from flask import Flask, request, session, jsonify, render_template, request,redirect, url_for, flash
import pandas as pd  # Assuming you use pandas for data manipulation
# from sklearn.externals import joblib  # Assuming you saved your fraud classification model using joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import auc, roc_curve,confusion_matrix
from sklearn.pipeline import Pipeline
import sklearn
import joblib
import json
import csv
import mysql.connector
from flask_mysqldb import MySQL
from decimal import Decimal
import app

app = Flask(__name__)

app.secret_key = 'HDioni'

# Configuration de la base de données MySQL
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root' # Utilisateur MySQL
app.config['MYSQL_PASSWORD'] = 'Haw@2120' # Mot de passe MySQL
app.config['MYSQL_DB'] = 'fraude_data' # Base de données MySQL
app.config['MYSQL_PORT'] = 3306


mysql = MySQL(app)


# import mysql.connector
# import csv

# # Configuration de la connexion
# cnx = {
#     'host': 'localhost',
#     'username': 'root',
#     'password': 'Haw@2120',
#     'database': 'data_fr'
    
    

    
# }

# # Établissement de la connexion
# connection = mysql.connector.connect(
#         host=cnx['host'],
#         user=cnx['username'],
#         password=cnx['password'],
#         database=cnx['database']
#     )
# print("Connected to MySQL database!")

#     # Création d'un objet curseur
# cursor = connection.cursor()

#     # Requête pour créer une table (assurez-vous que la table existe déjà dans votre base de données)
# create_table = """
#                     CREATE TABLE IF NOT EXISTS data_transactions (
#                     step INT,
#                     amount FLOAT,
#                     oldbalanceOrg FLOAT,
#                     newbalanceOrig FLOAT,
#                     oldbalanceDest FLOAT,
#                     newbalanceDest FLOAT,
#                     type_TRANSFER INT,
#                     origBalanceDiscrepancy FLOAT,
#                     destBalanceDiscrepancy FLOAT,
#                     vraiValeur INT,
#                     valeurPredite INT
#                 );"""

#     # Exécution de la requête pour créer la table
# cursor.execute(create_table)

#     # Chemin du fichier CSV
# csv_file_path = 'transaction_data.csv'

#     # Insertion des données depuis le fichier CSV
# with open(csv_file_path, 'r') as csv_file:
#         csv_reader = csv.reader(csv_file)
#         next(csv_reader)  # Skip header row

#         for row in csv_reader:
#             insert_data = tuple(row)
#             insert_statement = """INSERT INTO data_transactions 
#                                   (step, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, type_TRANSFER, origBalanceDiscrepancy, destBalanceDiscrepancy, vraiValeur, valeurPredite) 
#                                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s);"""
#             cursor.execute(insert_statement, insert_data)

#     # Commit des changements
# connection.commit()




#     # Fermeture du curseur et de la connexion
# if connection.is_connected():
#         cursor.close()
#         connection.close()
#         print("Connection closed.")




methods=['POST', 'PUT', 'GET', 'PATCH', 'DELETE']


def decimal_default(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError

@app.route('/')
def Accueil ():
    return render_template('accueil.html')


@app.route('/form')
def form():
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM  data_transactions")
    cur.connection.commit()
    trans_data = cur.fetchall()
    cur.close()
    return render_template('form.html', trans_data=trans_data )




@app.route('/streamlit')
def streamlit():
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM  data_transactions")
    # Fetch all rows as a list of dictionaries
    columns = [desc[0] for desc in cur.description]
    # print(columns)
    data_trans = [dict(zip(columns, row)) for row in cur.fetchall()]
    # print(data_trans)
    # Convert the list of dictionaries to JSON
    json_data = json.dumps(data_trans, default=decimal_default)

    # # Display or use the JSON data as needed
    print(json_data)

    cur.close()
    # return render_template('form.html', json_data=json_data )
    return jsonify(json_data)


 



@app.route('/add', methods=['POST'])
def add():
    if request.method == 'POST':
     step = request.form['step']
     amount = request.form['amount']
     oldbalanceOrg = request.form['oldbalanceOrg']
     newbalanceOrig = request.form['newbalanceOrig']
     oldbalanceDest = request.form['oldbalanceDest']
     newbalanceDest = request.form['newbalanceDest']
     type_TRANSFER = request.form['type_TRANSFER']
     origBalanceDiscrepancy = request.form['origBalanceDiscrepancy']
     destBalanceDiscrepancy = request.form['destBalanceDiscrepancy']
     vraiValeur = request.form['vraiValeur']
     
# Stocker les valeurs dans la base de données MySQL
     cur = mysql.connection.cursor()
     cur.execute("INSERT INTO data_transactions (step, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, type_TRANSFER, origBalanceDiscrepancy, destBalanceDiscrepancy,vraiValeur) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s,%s)", (step, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, type_TRANSFER, origBalanceDiscrepancy, destBalanceDiscrepancy,vraiValeur))
     mysql.connection.commit()
     cur.close()
     return redirect(url_for('form'))


# ""enregistrement du modèle
scikit_version = sklearn.__version__
scikit_version

data_finance= pd.read_csv('preprocessed_transaction_data.csv')

X = data_finance.drop(['isFraud'], axis = 1)
y = data_finance['isFraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 123)

y_train.shape, y_test.shape

rf_clf = RandomForestClassifier()

randforest_clf= rf_clf.fit(X_train, y_train)

y_pred_rf = randforest_clf.predict(X_test)

pred_result= pd.DataFrame({ 'y_test': y_test,
                           'y_pred': y_pred_rf})
pred_result

print('Random Forest\n')
accuracy = accuracy_score(y_test, y_pred_rf)
precision = precision_score(y_test, y_pred_rf)
recall = recall_score(y_test, y_pred_rf)


print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)

fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)

AUC_rf = auc(fpr_rf, tpr_rf)

print('AUC for Random Forest :', AUC_rf)

classifier = RandomForestClassifier()
clf_pipeline = Pipeline(steps=[('classifer', classifier)])
pipeline_model = clf_pipeline.fit(X_train,y_train)

y_pred= pipeline_model.predict(X_test)

pipe_clf_param= {}

pipe_clf_param['pipeline_clf']=pipeline_model
pipe_clf_param['sklearn_version'] = scikit_version
pipe_clf_param['accuracy']= accuracy
pipe_clf_param['precision']= precision
pipe_clf_param['recall']=recall
pipe_clf_param['AUC_rf ']=AUC_rf

"""### Enregistrement du model"""

# filename= 'model/pipe_clf.joblib'

joblib.dump(pipe_clf_param, 'pipe_clf.joblib')

model_clf = joblib.load('pipe_clf.joblib')
print('test sur le modèle charger')

model_fraud = model_clf['pipeline_clf']
model_fraud

y_pred= model_fraud.predict(X_test)
print(y_pred)
accuracy_score(y_test, y_pred)

print(model_clf['accuracy'])
print('model enregistrer')


# model_clf = joblib.load(open(filename, 'rb'))
# model_fraud = model_clf['pipeline_clf']


@app.route('/doprediction',methods=['GET'])
def do_prediction():
    return render_template('prediction.html')

@app.route('/showpredictions',methods=['GET'])
def show_prediction():
    # Retrieve the response from the session
    response = session.pop('prediction_data', None)
    if response is None:
        return jsonify({"error": "No prediction response found"}), 400

    return render_template('showpredicts.html', response=response)
 
 
 
@app.route('/api/prediction', methods=['POST'])
def predict_fraud():
    try:
        if request.method == 'POST':
            step = request.form['step']
            amount = request.form['amount']
            oldbalanceOrg = request.form['oldbalanceOrg']
            newbalanceOrig = request.form['newbalanceOrig']
            oldbalanceDest = request.form['oldbalanceDest']
            newbalanceDest = request.form['newbalanceDest']
            type_TRANSFER = request.form['type_TRANSFER']
            origBalanceDiscrepancy = request.form['origBalanceDiscrepancy']
            destBalanceDiscrepancy = request.form['destBalanceDiscrepancy']
        data = {
            "input_data": [
                {   
                "step" : step,
                "amount" : amount,
                "oldbalanceOrg" : oldbalanceOrg,
                "newbalanceOrig" : newbalanceOrig,
                "oldbalanceDest" : oldbalanceDest,
                "newbalanceDest" : newbalanceDest,
                "type_TRANSFER" : type_TRANSFER,
                "origBalanceDiscrepancy" : origBalanceDiscrepancy,
                "destBalanceDiscrepancy" : destBalanceDiscrepancy
                }
            ]
        }
        
        # data = request.get_json(force=True)
        input_data = data.get("input_data", [])  # Extract the list of input data from the input JSON

        if not input_data:
            return jsonify({"error": "No 'input_data' key found in the input JSON"}), 400

        
        if not input_data:
            return jsonify({"error": "No data found in the database"}), 400

        # Assuming 'df' is your dataset
        df = pd.DataFrame(input_data)

       # Assuming 'model_fraud' has a 'predict' method
        predictions = model_fraud.predict(df[['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'type_TRANSFER', 'origBalanceDiscrepancy', 'destBalanceDiscrepancy']])

        print("predictions")
        print(predictions)
        # Combine input data and fraud predictions into a response dictionary
        columns = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'type_TRANSFER', 'origBalanceDiscrepancy', 'destBalanceDiscrepancy']
        response = {"columns": columns, "input_data": input_data, "fraud_predictions": predictions.tolist()}
        
        response = jsonify(response)
        # response_base64 = base64.urlsafe_b64encode(json.dumps(response).encode()).decode()

        # Store the response in the session
        session['prediction_data'] = {
            "columns": columns,
            "input_data": input_data,
            "fraud_predictions": predictions.tolist()
        }
        
        print("response")
        print(response)
        return redirect(url_for('show_prediction'))
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/update_form/<int:id>', methods=['GET'])
def update_form(id):
    return render_template('update_form.html', id=id)
    

@app.route('/update_last', methods=['GET'])
def update_last():
    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT MAX(id) FROM data_transactions")
        last_id = cur.fetchone()[0]
        cur.close()
        return redirect(url_for('update_form', id=last_id))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    
@app.route('/update/<int:id>', methods=methods)
def update(id):
        try:
          if request.form.get('_method') == 'PUT':
              valeurPredite = request.form.get('valeurPredite')
              if valeurPredite is None:
                 return jsonify({'error': 'Nouvelle tâche non spécifiée'}), 400
              cur = mysql.connection.cursor()
              cur.execute("UPDATE data_transactions SET valeurPredite=%s WHERE id=%s", (valeurPredite, id))
              mysql.connection.commit()
              cur.close()
          return redirect(url_for('update_form', id=id))
        except Exception as e:
             return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True,port=5009)
    




