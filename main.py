# You must install these dependencies
# pip install pandas
# pip install joblib
# pip install fastapi
# pip install pydantic

# Import necessary libraries
import pandas as pd  # For data manipulation
from joblib import load  # For loading the trained machine learning model
from fastapi import FastAPI, HTTPException  # For creating the API and handling exceptions
from fastapi.responses import JSONResponse  # For returning JSON responses
from pydantic import BaseModel  # For creating Pydantic models for request/response validation

# Load the trained machine learning model
JOBLIB_PATH = "trained_pipeline.pkl"
# Load the trained machine learning model using joblib
loaded_pipeline = load(JOBLIB_PATH)

# Define a Pydantic model for the input data
class ModelInputs(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Create a FastAPI instance
app = FastAPI()

# Define a route for making predictions
@app.post("/prediction")
async def prediction(model_inputs: ModelInputs):
    try:
        # Convert the input data into a DataFrame
        input_df = pd.DataFrame([model_inputs.dict()])
        # Rename the columns to match the trained model's expected input
        input_df.columns = ['petal length (cm)', 'sepal length (cm)', 'petal width (cm)', 'sepal width (cm)']
        # Use the loaded model to make predictions on the input data
        prediction = loaded_pipeline.predict(input_df)
        # Return the prediction as a JSON response
        
        # Optional: You can store the data in a database before returning the results to the user
        # ...
        # ...

        return JSONResponse(content={"prediction": prediction.tolist()})
    except Exception as e:
        # If an error occurs, raise an HTTPException with status code 500
        raise HTTPException(status_code=500, detail=str(e))