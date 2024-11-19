from flask import Flask, render_template, request, jsonify
from typing import List
from src.pipeline.predict_pipeline import Prediction

application = Flask(__name__)
application.static_folder = 'static'

class StudentData:
    def __init__(self, data_dict):
        self.gender = data_dict['gender']
        self.race_ethnicity = data_dict['race_ethnicity']
        self.parental_level_of_education = data_dict['parental_level_of_education']
        self.lunch = data_dict['lunch']
        self.test_preparation_course = data_dict['test_preparation_course']
        self.reading_score = [int(score) for score in data_dict['reading_score']]
        self.writing_score = [int(score) for score in data_dict['writing_score']]
    
    def dict(self):
        return {
            'gender': self.gender,
            'race_ethnicity': self.race_ethnicity,
            'parental_level_of_education': self.parental_level_of_education,
            'lunch': self.lunch,
            'test_preparation_course': self.test_preparation_course,
            'reading_score': self.reading_score,
            'writing_score': self.writing_score
        }

@application.route('/', methods=['GET'])
def get_form():
    return render_template('index.html')

@application.route('/submit/', methods=['POST'])
def submit_form():
    # In Flask, request.form.getlist() is used to get multiple values for the same field
    data = StudentData({
        'gender': request.form.getlist('gender'),
        'race_ethnicity': request.form.getlist('race_ethnicity'),
        'parental_level_of_education': request.form.getlist('parental_level_of_education'),
        'lunch': request.form.getlist('lunch'),
        'test_preparation_course': request.form.getlist('test_preparation_course'),
        'reading_score': request.form.getlist('reading_score'),
        'writing_score': request.form.getlist('writing_score')
    })
    
    model = Prediction()
    df = model.make_df(data.dict())
    prediction = model.predict(df)
    
    return render_template('index.html', prediction=int(prediction[0]))

if __name__ == '__main__':
    application.run(debug=True)