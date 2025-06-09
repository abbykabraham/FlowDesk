import json
from typing import Dict, List, Tuple, Any
from pathlib import Path

class ABCDDataLoader:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.dialogs = None
        self.ontology = None
        self.utterances = None
        self.guidelines = None
        
    def load_all(self) -> None:
        """Load all necessary data files"""
        # Load main dialogue data
        with open(self.data_dir / 'abcd_v1.1.json', 'r', encoding='utf-8') as f:
            self.dialogs = json.load(f)
            
        # Load ontology (intent/action definitions)
        with open(self.data_dir / 'ontology.json', 'r', encoding='utf-8') as f:
            self.ontology = json.load(f)
            
        # Load candidate utterances
        with open(self.data_dir / 'utterances.json', 'r', encoding='utf-8') as f:
            self.utterances = json.load(f)
            
        # Load guidelines
        with open(self.data_dir / 'guidelines.json', 'r', encoding='utf-8') as f:
            self.guidelines = json.load(f)
    
    def get_intent_classification_data(self) -> List[Tuple[str, str]]:
        """Extract (text, intent) pairs for intent classification"""
        data = []
        for split in self.dialogs:
            for dialog in self.dialogs[split]:
                # Get first customer message and flow
                for turn in dialog['original']:
                    if turn[0] == 'customer':  # First customer turn
                        text = turn[1]
                        intent = dialog['scenario']['flow']
                        data.append((text, intent))
                        break
        return data
    
    def get_action_tracking_data(self) -> List[Dict[str, Any]]:
        """Extract context-action pairs for action tracking"""
        data = []
        for split in self.dialogs:
            for dialog in self.dialogs[split]:
                context = []
                for turn in dialog['original']:
                    context.append(turn)
                    # If it's an agent turn, create a training example
                    if turn[0] == 'agent':
                        data.append({
                            'context': context[:-1],  # Previous turns
                            'next_action': turn[1],  # Agent's response
                            'flow': dialog['scenario']['flow']
                        })
        return data
    
    def get_splits(self) -> Dict[str, List]:
        """Return train/dev/test splits"""
        return self.dialogs
