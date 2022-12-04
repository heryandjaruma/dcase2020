from audata import Audata

class Paudata(Audata):
    def __repr__(self) -> str:
        return '<Paudata('+self.origin+') at '+ str(hex(id(self)))+'>'
    def set_sound_class(self, sound_class) -> None:
        self.sound_class = sound_class
    def set_ordo(self, ordo:int) -> None:
        self.ordo = ordo