from audata import Audata
from processor import *
from pydub import AudioSegment

mapping = [
    [1,2],
    [2,1],
    [3,1],
    [4,1],
    [5,1],
    [6,1]
]

if __name__ == '__main__':

    # OVERLAYINGt
    faudatas = list()
    for item in through_file('origin.txt'):
        faudatas.append(
            from_file(str(item.strip('\n')))
        )
        set_particle_into_faudata(faudatas[-1])
    # for maps in mapping:
    #     overlay(
    #         faudatas[maps[0]-1],
    #         faudatas[maps[1]-1],
    #         str(maps[0])+'_'+str(maps[1])
    #     )
    print('Done overlaying')

    # SEPARATION
    histories = list()
    for item in through_file('history.txt'):
        histories.append(
            retrieve_history(item.strip('\n'))
        )
    for history,maps in zip(histories,mapping):
        through_history(
            history=history,
            maps=maps,
            old_faudata1=faudatas[maps[0]-1],
            old_faudata2=faudatas[maps[1]-1]
        )
    print('Done separating')
    faudatas.clear()
    histories.clear()