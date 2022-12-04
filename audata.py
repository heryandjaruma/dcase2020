from dataclasses import dataclass
from pydub import AudioSegment
import pandas as pd
from abc import ABC

@dataclass
class Audata(ABC):
    origin: str
    audio: list[AudioSegment]
    dataframe: pd.DataFrame
    time_start: int
    time_end: int
    time_duration: int
    dataframe_duration: int