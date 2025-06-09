# FlowDesk

An intelligent customer service chatbot system built using the Action-Based Conversations Dataset (ABCD) from ASAPP Research. The system handles customer interactions through intent classification and action tracking.

## Dataset Reference
This project uses the Action-Based Conversations Dataset (ABCD) from ASAPP Research:
- Paper: [Action-Based Conversations Dataset: A Corpus for Building More In-Depth Task-Oriented Dialogue Systems](https://arxiv.org/abs/2104.00783)
- Repository: [asappresearch/abcd](https://github.com/asappresearch/abcd)

## Project Structure

```
FlowDesk/
├── data/                    # Dataset files
│   ├── abcd_v1.1.json
│   ├── ontology.json
│   ├── utterances.json
│   └── guidelines.json
│
├── src/                     # Application code
│   ├── __init__.py
│   ├── data_loader.py       # Data loading functions
│   ├── preprocess.py        # Text preprocessing
│   ├── intent_classifier.py # Intent classification model
│   └── api.py              # FastAPI application
│
├── notebooks/               # Development notebooks
│   ├── 01_EDA.ipynb        # Exploratory Data Analysis
│   ├── 02_Modeling.ipynb   # Model development
│   └── 03_Intent_Classification.ipynb  # Intent model training
│
├── models/                  # Trained models
│   └── intent_classifier/   # Intent classification model files
│
├── requirements.txt         # Project dependencies
└── README.md               # Project documentation
```

## Setup

1. Create virtual environment:
```bash
python -m venv .env1
source .env1/bin/activate  # Linux/Mac
.env1\Scripts\activate     # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the API:
```bash
uvicorn src.api:app --reload
```

## API Endpoints

- `POST /chat`: Main chat endpoint for customer interactions
- `GET /health`: Health check endpoint
- `GET /session/{session_id}`: Get session history

## Development

- `01_EDA.ipynb`: Exploratory Data Analysis of customer service interactions
- `02_Modeling.ipynb`: Model development and experimentation
- `03_Intent_Classification.ipynb`: Intent classification model training

## Current Features

- Intent Classification: Identifies customer intents from messages
- Session Management: Maintains conversation context
- Conversation History Tracking: Stores and retrieves chat history
- API Documentation: Interactive Swagger UI at `/docs`

## Next Steps

- Action Tracking: Predict next actions based on intent and context
- Response Generation: Generate appropriate responses
- Enhanced Session Management: Add user preferences and context
- Model Retraining Pipeline: Enable model updates with new data
