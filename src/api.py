from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
from .intent_classifier import IntentClassifier
import os
from datetime import datetime
import uuid

app = FastAPI(title="FlowDesk")

# Initialize the intent classifier
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'intent_classifier')
print(f"Looking for model at: {MODEL_PATH}")

try:
    intent_classifier = IntentClassifier(model_path=MODEL_PATH)
    print("Intent classifier initialized successfully")
except Exception as e:
    print(f"Error initializing intent classifier: {str(e)}")
    intent_classifier = None

# Session management
class Session:
    def __init__(self):
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.messages = []
        self.context = {}

    def add_message(self, message: str, intent: Dict[str, float]):
        self.messages.append({
            'timestamp': datetime.now(),
            'message': message,
            'intent': intent
        })
        self.last_activity = datetime.now()

    def get_conversation_history(self, max_messages: int = 5) -> List[str]:
        return [msg['message'] for msg in self.messages[-max_messages:]]

# Store active sessions
sessions: Dict[str, Session] = {}

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    intent: Optional[Dict[str, float]] = None
    session_id: str
    conversation_history: Optional[List[str]] = None

def get_or_create_session(session_id: Optional[str] = None) -> Session:
    if not session_id or session_id not in sessions:
        session_id = str(uuid.uuid4())
        sessions[session_id] = Session()
    return sessions[session_id]

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        if intent_classifier is None:
            return ChatResponse(
                response="Model not trained yet. Please train the model first.",
                session_id=request.session_id or "new_session"
            )
        
        # Get or create session
        session = get_or_create_session(request.session_id)
        
        # Get intent prediction
        intent_probs = intent_classifier.predict(request.message)
        
        # Add message to session history
        session.add_message(request.message, intent_probs)
        
        # Get conversation history
        history = session.get_conversation_history()
        
        # For now, just echo the message and return intent probabilities
        return ChatResponse(
            response=f"Received: {request.message}",
            intent=intent_probs,
            session_id=session.created_at.strftime("%Y%m%d") + "_" + str(uuid.uuid4())[:8],
            conversation_history=history
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Add endpoint to get session history
@app.get("/session/{session_id}")
async def get_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[session_id]
    return {
        "session_id": session_id,
        "created_at": session.created_at,
        "last_activity": session.last_activity,
        "message_count": len(session.messages),
        "messages": session.messages
    }
