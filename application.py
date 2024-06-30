from flask import Flask,render_template,request
from src.pipelines.predict_pipeline import predictData

application = Flask(__name__)
app = application

@app.route("/")
def home():
    return "Yooo I am back"

@app.route("/predict",methods = ["GET", "POST"])
def predict():
    if request.method == "POST":
        gender = request.form.get('gender')
        race_ethnicity = request.form.get('race_ethnicity')
        parental_level_of_education = request.form.get('parental_level_of_education')
        lunch = request.form.get('lunch')
        test_preparation_course = request.form.get('test_preparation_course')
        reading_score = int(request.form.get('reading_score'))
        writing_score = int(request.form.get('writing_score'))
        X = [gender,race_ethnicity,parental_level_of_education,lunch,test_preparation_course,reading_score,writing_score]
        predict_data = predictData()
        mathScore = predict_data.predict(X=X)
        return render_template("predict.html", results = mathScore)
    else:
        return render_template("predict.html")
    


if __name__ == "__main__":
    app.run(debug=True)