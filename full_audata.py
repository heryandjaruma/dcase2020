from audata import Audata

class Faudata(Audata):
    paudata: list

    def set_paudata(self, paudata) -> None:
        '''append paudata tp faudata'''
        self.paudata = paudata
    def set_audio(self, audio) -> None:
        '''set audio to faudata'''
        self.audio.append(audio)
    def set_dataframe(self, dataframe) -> None:
        '''set dataframe to faudata'''
        self.dataframe = dataframe
    def set_size(self, total_paudata:int) -> None:
        '''set paudata size to this faudata'''
        self.size = total_paudata
    def __repr__(self) -> str:
        return '<Faudata('+self.origin+') at '+ str(hex(id(self)))+'>'