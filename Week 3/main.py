from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI(title="Steel Industry Energy Consumption Predictor")

BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

artifact = joblib.load(BASE_DIR / "week3_model_baseline.joblib")
model = artifact["model"]
scaler = artifact["scaler"]        
pca = artifact["pca"]             
feature_names = artifact["feature_names"]


def build_field_hint(feature_name: str) -> str:
    """Give the user a quick sense of the expected range for each field."""
    if feature_name.endswith("_sin") or feature_name.endswith("_cos"):
        return "-1 to 1"
    if feature_name == "is_weekend" or feature_name.startswith(
        ("Load_Type_", "Day_of_week_", "WeekStatus_")
    ):
        return "0 or 1"
    return "numeric"


FIELD_META = [{"name": f, "hint": build_field_hint(f)} for f in feature_names]


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "home.html", {"active": "home"})


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(request, "dashboard.html", {"active": "dashboard"})


@app.get("/predict", response_class=HTMLResponse)
async def predict_form(request: Request):
    return templates.TemplateResponse(
    request, "predict.html",
    {"active": "predict", "fields": FIELD_META, "prediction": None, "error": False},
)


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request):
    form_data = await request.form()

    try:
        from fastapi import UploadFile

        row = {}

        for f in feature_names:
            value = form_data.get(f, "0")

            if not isinstance(value, str):
                raise ValueError(f"Unexpected file uploaded for {f}")

            row[f] = float(value)
    except ValueError:
        fields = [
            {"name": f, "hint": build_field_hint(f), "value": form_data.get(f, "0")}
            for f in feature_names
        ]
        return templates.TemplateResponse(
            request,
            "predict.html",
            {
                "fields": fields,
                "prediction": "Invalid input — please enter numbers only.",
                "error": True,
            },
        )

    X_input = pd.DataFrame([row], columns=feature_names)

    # Apply the same preprocessing the pipeline was trained with. For the
    # baseline model these are both None, but this keeps main.py compatible
    # if you point the loaded artifact at a PCA pipeline instead.
    if scaler is not None:
        X_input = pd.DataFrame(scaler.transform(X_input), columns=feature_names)
    if pca is not None:
        X_input = pca.transform(X_input)

    pred = model.predict(X_input)[0]

    fields = [
        {"name": f, "hint": build_field_hint(f), "value": row[f]} for f in feature_names
    ]
    return templates.TemplateResponse(
        request,
        "predict.html",
        {
            "fields": fields,
            "prediction": f"{pred:.2f} kWh",
            "error": False,
        },
    )