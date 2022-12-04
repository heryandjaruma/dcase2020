from dataclasses import dataclass
from row import Row

@dataclass
class History:
    origin:str
    row: list[Row]