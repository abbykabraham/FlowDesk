from typing import List, Dict, Any, Tuple
import re

class ABCDPreprocessor:
    def __init__(self):
        self.max_context_length = 512  # For transformer models
        
    @staticmethod
    def clean_text(text: str) -> str:
        """Basic text cleaning"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-z0-9\s.,!?]', '', text)
        
        # Strip whitespace
        text = text.strip()
        
        return text
    
    def prepare_intent_data(self, data: List[Tuple[str, str]]) -> Dict[str, List]:
        """Prepare data for intent classification"""
        processed = {
            'texts': [],
            'labels': []
        }
        
        for text, intent in data:
            processed['texts'].append(self.clean_text(text))
            processed['labels'].append(intent)
            
        return processed
    
    def prepare_action_data(self, data: List[Dict[str, Any]]) -> Dict[str, List]:
        """Prepare data for action tracking"""
        processed = {
            'contexts': [],
            'actions': [],
            'flows': []
        }
        
        for item in data:
            # Convert context turns into a single string
            context_str = " ".join([
                f"{turn[0]}: {self.clean_text(turn[1])}"
                for turn in item['context']
            ])
            
            processed['contexts'].append(context_str)
            processed['actions'].append(item['next_action'])
            processed['flows'].append(item['flow'])
            
        return processed
