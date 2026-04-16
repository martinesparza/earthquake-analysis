"""
SessionMetadata: Map sessions to conditions, animals, and dates.
This is Phase 1, Step 1 of the behavior analysis platform.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
from pathlib import Path
import pandas as pd
from datetime import datetime


@dataclass
class SessionMetadata:
    """Metadata for a single recording session."""
    
    session_name: str       # e.g., "M103_2026_02_17_14_00"
    animal_id: str          # e.g., "M103"
    condition: str          # "control", "normal", or "muscimol"
    date: datetime          # Session date
    day_num: int            # Day within condition (0 for control baseline)
    notes: str = ""         # Optional notes
    
    @property
    def label(self) -> str:
        """Human-readable label for display."""
        return f"{self.animal_id} {self.condition.capitalize()} Day{self.day_num}"
    
    @property
    def short_label(self) -> str:
        """Short label for plots."""
        cond_map = {"control": "Ctl", "normal": "Norm", "muscimol": "Musc"}
        return f"{self.animal_id}-{cond_map.get(self.condition, self.condition[:3])}-D{self.day_num}"


class SessionRegistry:
    """Load and query session metadata from the sessions.csv file."""
    
    def __init__(self, sessions_csv_path: Optional[Path] = None):
        """
        Parameters
        ----------
        sessions_csv_path : Path, optional
            Path to sessions.csv. If None, looks for metadata/sessions.csv in repo root.
        """
        if sessions_csv_path is None:
            # Default location relative to tools/behavior/
            default_path = Path(__file__).parent.parent.parent / "metadata" / "sessions.csv"
            sessions_csv_path = default_path
        
        self.csv_path = Path(sessions_csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"sessions.csv not found at {self.csv_path}")
        
        self._load_sessions()
    
    def _load_sessions(self):
        """Load sessions from CSV."""
        df = pd.read_csv(self.csv_path)
        df['date'] = pd.to_datetime(df['date'])
        
        self.sessions: List[SessionMetadata] = [
            SessionMetadata(
                session_name=row['session_name'],
                animal_id=row['animal_id'],
                condition=row['condition'],
                date=row['date'],
                day_num=int(row['day_num']),
                notes=str(row['notes']) if pd.notna(row['notes']) else ""
            )
            for _, row in df.iterrows()
        ]
        
        # Create lookup dictionaries
        self._by_name: Dict[str, SessionMetadata] = {s.session_name: s for s in self.sessions}
        self._by_animal: Dict[str, List[SessionMetadata]] = {}
        for s in self.sessions:
            if s.animal_id not in self._by_animal:
                self._by_animal[s.animal_id] = []
            self._by_animal[s.animal_id].append(s)
    
    def get(self, session_name: str) -> SessionMetadata:
        """Get metadata for a specific session by name."""
        if session_name not in self._by_name:
            raise ValueError(f"Session '{session_name}' not found in registry")
        return self._by_name[session_name]
    
    def filter(self, animal_ids: Optional[List[str]] = None, 
               conditions: Optional[List[str]] = None) -> List[SessionMetadata]:
        """
        Filter sessions by animal ID and/or condition.
        
        Parameters
        ----------
        animal_ids : List[str], optional
            Filter to specific animals (e.g., ["M103", "M106"])
        conditions : List[str], optional
            Filter to specific conditions (e.g., ["normal", "muscimol"])
        
        Returns
        -------
        List[SessionMetadata]
            Filtered list of sessions.
        """
        result = self.sessions
        
        if animal_ids is not None:
            result = [s for s in result if s.animal_id in animal_ids]
        
        if conditions is not None:
            result = [s for s in result if s.condition in conditions]
        
        return result
    
    def animals(self) -> List[str]:
        """Get list of all unique animal IDs."""
        return sorted(set(s.animal_id for s in self.sessions))
    
    def conditions(self) -> List[str]:
        """Get list of all unique conditions."""
        return sorted(set(s.condition for s in self.sessions))
    
    def __repr__(self) -> str:
        """Summary of loaded sessions."""
        animals = self.animals()
        conds = self.conditions()
        return (
            f"SessionRegistry with {len(self.sessions)} sessions:\n"
            f"  Animals: {animals}\n"
            f"  Conditions: {conds}\n"
            f"  Date range: {min(s.date for s in self.sessions).date()} "
            f"to {max(s.date for s in self.sessions).date()}"
        )


# Convenience function
def load_session_registry(sessions_csv_path: Optional[Path] = None) -> SessionRegistry:
    """Load the session registry."""
    return SessionRegistry(sessions_csv_path)
