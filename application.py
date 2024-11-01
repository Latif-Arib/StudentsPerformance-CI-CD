from fastapi import FastAPI, Form, Request
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from typing import Optional, List
from src.pipeline.predict_pipeline import Prediction

application = FastAPI()

application.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory='templates')

class StudentData(BaseModel):
    gender: List[str]
    race_ethnicity: List[str]
    parental_level_of_education: List[str]
    lunch: List[str]
    test_preparation_course: List[str]
    reading_score: List[int]
    writing_score: List[int]

@application.post("/submit/")
async def submit_form(
    request: Request,
    gender: list[str] = Form(...),
    race_ethnicity: list[str] = Form(...),
    parental_level_of_education: list[str] = Form(...),
    lunch: list[str] = Form(...),
    test_preparation_course: list[str] = Form(...),
    reading_score: list[int] = Form(...),
    writing_score: list[int] = Form(...)
):
    data = StudentData(
        gender=gender,
        race_ethnicity=race_ethnicity,
        parental_level_of_education=parental_level_of_education,
        lunch=lunch,
        test_preparation_course=test_preparation_course,
        reading_score=reading_score,
        writing_score=writing_score
    )
    
    model = Prediction()
    df = model.make_df(data.dict())
    prediction = model.predict(df)
    
    
    # return {"message": "Form submitted successfully!", "prediction": prediction[0]}
    return templates.TemplateResponse('index.html',{
        'request':request,
        'prediction': int(prediction[0])
    })


