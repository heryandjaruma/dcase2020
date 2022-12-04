from full_audata import Faudata
from particle_audata import Paudata
from row import Row
from history import History
from goto import Goto
from pydub import AudioSegment
import pandas as pd
pd.set_option('display.max_rows', None)
from colorama import Fore
import numpy as np
import nussl

goto = Goto()

def through_file(filename:str):
    goto.cd(goto.mainpath)
    # goto.trees()
    # input()
    f = open(filename,'r')
    return f.readlines()

def from_file(origin:str) -> Faudata:
    '''retreive Audata from origin'''

    goto.subf('raw','foa')
    audio = list()
    for i in range(1,5):
        path = origin+'_channel'+str(i)+'.wav'
        audio.append(AudioSegment.from_file(path,format='wav'))
    
    goto.subf('raw','metadata')
    dataframe = pd.read_csv(origin+'.csv',header=None)
    dataframe.columns = ['Frm', 'Class', 'Track', 'Azmth', 'Elev']

    time_start = dataframe.iloc[0][0]
    time_end = dataframe.iloc[-1][0]
    time_duration = time_end - time_start
    dataframe_duration = dataframe.shape[0]

    return Faudata(origin,audio,dataframe,time_start,time_end,time_duration,dataframe_duration)

def extract_dataframe(start: int, end: int, dataframe_to_extract: pd.DataFrame) -> pd.DataFrame:
    dataframe_to_extract = dataframe_to_extract.set_index('Frm')
    temp = dataframe_to_extract.loc[start:end,:].copy()
    dataframe_to_extract = dataframe_to_extract.reset_index()
    return temp

def extract_audio(start: int, end: int, audio: list) -> list:
    temp = list()
    temp.append(audio[0][start*100:end*100])
    temp.append(audio[1][start*100:end*100])
    temp.append(audio[2][start*100:end*100])
    temp.append(audio[3][start*100:end*100])
    return temp

def set_particle_into_faudata(faudata: Faudata) -> None:
    '''set particle into faudata'''

    start = end = faudata.dataframe.iloc[0][0]
    sound_class = faudata.dataframe.iloc[0][1]

    temped = list()
    existed_class = set()
    ordo = 0

    for i,row in faudata.dataframe.iterrows():
        if row['Class'] == sound_class:
            end = row['Frm']
        else:
            if sound_class not in existed_class:
                extracted_dataframe = extract_dataframe(start,end,faudata.dataframe)
                extracted_audio = extract_audio(start,end,faudata.audio)
                temped.append(
                    Paudata(
                        origin = faudata.origin,
                        audio = extracted_audio,
                        dataframe = extracted_dataframe,
                        time_start = start,
                        time_end = end,
                        time_duration = end-start,
                        dataframe_duration = extracted_dataframe.shape[0]
                    )
                )
                temped[-1].set_sound_class(sound_class)
                temped[-1].set_ordo(ordo)
                ordo+=1
                existed_class.add(sound_class)
            start = end = row['Frm']
            sound_class = row['Class']
    if sound_class not in existed_class:
        extracted_dataframe = extract_dataframe(start,end,faudata.dataframe)
        extracted_audio = extract_audio(start,end,faudata.audio)
        temped.append(
            Paudata(
                origin = faudata.origin,
                audio = extracted_audio,
                dataframe = extracted_dataframe,
                time_start = start,
                time_end = end,
                time_duration = end-start,
                dataframe_duration = extracted_dataframe.shape[0]
            )
        )
        temped[-1].set_sound_class(sound_class)
        temped[-1].set_ordo(ordo)
    faudata.set_paudata(paudata=temped)
    faudata.set_size(total_paudata=ordo+1)

def mecha_increment(increment_list:list, index_to_increment:int) -> None:
    increment_list[index_to_increment] += 1
    return increment_list[index_to_increment]-1

def name_wav_tunggal_cut(origin:str, sound_class:int, increment:int) -> str:
    return '_'.join([origin,str(sound_class),'%03d' % (increment)])

def name_mix_wav_tunggal_cut(origin:str, sound_class1:str, sound_class2:str,increment:int) -> str:
    return '_'.join([origin, str(sound_class1), str(sound_class2), '%03d' % (increment)])

def name_mix_wav(origin_name:str,increment:str) -> None:
    return origin_name[:-10]+'mix%03d'%increment +'_ov2'

def name_history_csv(faudata1:Faudata, faudata2:Faudata) -> str:
    return faudata1.origin + '_OVERLAY_' + faudata2.origin 

def cut_end_4channel(audio:list,time_end:int) -> list:
    temp = list()
    for item in audio:
        temp.append(
            item[:time_end*100]
        )
    return temp

def cut_begining_4channel(audio:list,time_start:int) -> list:
    temp = list()
    for item in audio:
        temp.append(
            item[time_start*100:]
        )
    return temp

def merge_4_channel(audio:list) -> AudioSegment:
    temp = AudioSegment.from_mono_audiosegments(audio[0],audio[1],audio[2],audio[3])
    return temp

def export_4_channel(audio:list, name:str, parentfolder:str, duration:str, *subf:str):
    '''exporting audio in the specified parentfolder and subfolder'''
    goto.subf(parentfolder)
    for sub in subf:
        goto.fwardf(sub)
    for i,item in enumerate(audio):
        temp = item[:duration*100]
        temp.export(name + '_channel'+str(i+1)+'.wav',format='wav')

def export_single_4_channel(audio:AudioSegment, name:str, parentfolder:str, duration:str, *subf:str) -> None:
    goto.subf(parentfolder)
    for sub in subf:
        goto.fwardf(sub)
    temp = audio[:duration*100]
    temp.export(name + '.wav',format='wav')

def export_4_channel_no_duration(audio:AudioSegment, name:str, parentfolder:str, *subf:str):
    goto.subf(parentfolder)
    for sub in subf:
        goto.fwardf(sub)
    audio.export(name + '.wav',format='wav')

def export_csv(dataframe:pd.DataFrame, name, parentfolder:str,*subf:str) -> None:
    goto.subf(parentfolder) 
    for sub in subf:
        goto.fwardf(sub)
    dataframe.to_csv(name+'.csv', header=False,index=False)

def export_history(dataframe: pd.DataFrame, name:str) -> None:
    goto.subf('overlay','history')
    dataframe.to_csv(name+'.csv',header=False,index=False)

def export_merged_4_channel(audio:list, name:str, parentfolder:str, duration:str, *subf:str) -> None:
    '''exporting merged 4 channel audio in the specified parentfolder and subfolder'''
    goto.subf(parentfolder)
    for sub in subf:
        goto.fwardf(sub)
    
    multichannel = AudioSegment.from_mono_audiosegments(audio[0],audio[1],audio[2],audio[3])
    multichannel = multichannel[:duration*100]
    # print(multichannel.duration_seconds)
    # input()
    multichannel.export(name +'.wav',format='wav')

    return multichannel

def extract_for_helper(borrow:bool,dataframe:pd.DataFrame,class1:int,class2:int,subf:str,NAMES:list) -> pd.DataFrame:
    goto.subf(
        'overlay',
        'csv_helper',
        subf
    )

    temp = dataframe.copy()

    temp1 = pd.DataFrame()
    temp2 = pd.DataFrame()
    for i,row in temp.iterrows():
        row_temp = np.array([
            row[0],
            row[1],
            row[2],
            row[3],
            row[4]
        ])
        if row[1] == class1:
            rdf = pd.DataFrame(row_temp.reshape(1,-1))
            temp1 = pd.concat([temp1,rdf])
        else:
            rdf = pd.DataFrame(row_temp.reshape(1,-1))
            temp2 = pd.concat([temp2,rdf])
    
    if borrow:
        # print(temp1.shape[0])
        # print(temp2.shape[0])
        # print(temp1.shape[0]-temp2.shape[0])
        for i in range(temp2.shape[0], temp1.shape[0]):
            rti = np.array(temp1.iloc[i])
            tempi = pd.DataFrame(rti.reshape(1,-1))
            temp2 = pd.concat([temp2,tempi])
            # print(temp1.iloc[i])
        # input()
    # print(temp1)
    # print(temp2)

    temp1.to_csv(NAMES[0]+'_helper.csv',header=None,index=None)
    temp2.to_csv(NAMES[1]+'_helper.csv',header=None,index=None)
    # input()



def overlay(faudata1: Faudata, faudata2: Faudata, subf:str) -> None:
    '''Overlay two Faudata'''

    print('Processing',Fore.YELLOW,faudata1.origin,faudata2.origin,Fore.WHITE)

    history = pd.DataFrame()

    faudata1_inc = list(np.ones(faudata1.size,dtype=int))
    faudata2_inc = list(np.ones(faudata2.size,dtype=int))

    increment = 1
    for paudata1 in faudata1.paudata:
        for paudata2 in faudata2.paudata:
            print(Fore.GREEN,f"Processing #{increment}",Fore.WHITE)

            faudata1.dataframe = faudata1.dataframe.set_index('Frm')

            l_dataframe = faudata1.dataframe.loc[:paudata1.time_start-1,:].copy()
            m_dataframe = faudata1.dataframe.loc[paudata1.time_start:paudata1.time_end,:].copy()
            r_dataframe = faudata1.dataframe.loc[paudata1.time_end+1:,:].copy()
            faudata1.dataframe = faudata1.dataframe.reset_index()

            p_dataframe = paudata2.dataframe.copy() # particle dataframe

            dataframe_combined = l_dataframe.copy()

            m_dataframe['unique_id'] = np.arange(0, m_dataframe.shape[0]*2,2)
            p_dataframe['unique_id'] = np.arange(1, p_dataframe.shape[0]*2,2)

            p_dataframe = p_dataframe.reset_index()
            m_dataframe = m_dataframe.reset_index()

            if paudata1.time_duration <= paudata2.time_duration:
                p_dataframe['Frm'] = m_dataframe['Frm']

                take_duration = paudata1.time_duration

                WAV_TUNGGAL_CUTS = [
                    name_wav_tunggal_cut(
                        origin=paudata1.origin,
                        sound_class=paudata1.sound_class,
                        increment=mecha_increment(faudata1_inc,paudata1.ordo)
                    ),
                    name_wav_tunggal_cut(
                        origin=paudata2.origin,
                        sound_class=paudata2.sound_class,
                        increment=mecha_increment(faudata2_inc,paudata2.ordo)
                    )
                ]

                particle_audata1 = export_merged_4_channel( #of faudata1-
                    paudata1.audio,
                    WAV_TUNGGAL_CUTS[0],
                    'overlay',
                    take_duration,
                    'wav_tunggal_cut',
                    subf
                )
                particle_audata2 = export_merged_4_channel( #of faudata2
                    paudata2.audio,
                    WAV_TUNGGAL_CUTS[1],
                    'overlay',
                    take_duration,
                    'wav_tunggal_cut',
                    subf
                )

                borrow = False # for helper 

            # elif paudata1.time_duration == paudata2.time_duration:
            #     pass
            else:
                p_dataframe['Frm'] = m_dataframe['Frm'].iloc[:paudata1.time_duration]

                take_duration = paudata2.time_duration

                WAV_TUNGGAL_CUTS = [
                    name_wav_tunggal_cut(
                        origin=paudata1.origin,
                        sound_class=paudata1.sound_class,
                        increment=mecha_increment(faudata1_inc,paudata1.ordo)
                    ),
                    name_wav_tunggal_cut(
                        origin=paudata2.origin,
                        sound_class=paudata2.sound_class,
                        increment=mecha_increment(faudata2_inc,paudata2.ordo)
                    )
                ]

                particle_audata1 = export_merged_4_channel( #of faudata1
                    paudata1.audio,
                    WAV_TUNGGAL_CUTS[0],
                    'overlay',
                    take_duration,
                    'wav_tunggal_cut',
                    subf
                )
                particle_audata2 = export_merged_4_channel( #of faudata2
                    paudata2.audio,
                    WAV_TUNGGAL_CUTS[1],
                    'overlay',
                    take_duration,
                    'wav_tunggal_cut',
                    subf
                )

                borrow = True

            new_dataframe = pd.concat([m_dataframe,p_dataframe])

            new_dataframe = new_dataframe.sort_values(by=['unique_id'])
            new_dataframe = new_dataframe.drop(columns='unique_id')
            new_dataframe = new_dataframe.dropna()
            new_dataframe['Frm'] = new_dataframe['Frm'].astype(int)

            extract_for_helper(
                borrow,
                new_dataframe,
                paudata1.sound_class,
                paudata2.sound_class,
                subf,
                WAV_TUNGGAL_CUTS
            )

            new_dataframe = new_dataframe.set_index('Frm')
            

            dataframe_combined = pd.concat([dataframe_combined, new_dataframe])
            dataframe_combined = pd.concat([dataframe_combined, r_dataframe])
            dataframe_combined = dataframe_combined.reset_index()

            # overlay
            MIX_WAV_TUNGGAL_CUT = name_mix_wav_tunggal_cut(faudata1.origin,paudata1.sound_class,paudata2.sound_class,increment)
            # export audio
            oolayed = particle_audata1.overlay(particle_audata2)
            export_single_4_channel(
                oolayed,
                MIX_WAV_TUNGGAL_CUT,
                'overlay',
                take_duration,
                'mix_wav_tunggal_cut'
            )
            #export csv
            MIX_WAV = name_mix_wav(faudata1.origin,increment)
            export_csv(
                dataframe_combined,
                MIX_WAV,
                'overlay',
                'metadata'
            )
            # export complete overlay wav
            full_audata1 = merge_4_channel(faudata1.audio)
            oolayed = full_audata1.overlay(particle_audata2,position=paudata1.time_start*100)
            export_4_channel_no_duration(
                oolayed,
                MIX_WAV,
                'overlay',
                'mix'
            )

            # into row dataframe
            row = np.array([
                faudata1.origin+'.wav',
                faudata2.origin+'.wav',
                paudata1.sound_class,
                paudata2.sound_class,
                MIX_WAV + '.wav',
                WAV_TUNGGAL_CUTS[0]+'.wav',
                WAV_TUNGGAL_CUTS[1]+'.wav',
                MIX_WAV_TUNGGAL_CUT+'.wav',
                take_duration
            ])
            history_dataframe = pd.DataFrame(row.reshape(1,-1))
            history = pd.concat([history,history_dataframe])

            increment+=1
        
    # export history
    HISTORY = name_history_csv(faudata1, faudata2)
    export_history(history,HISTORY)


def retrieve_faudata(origin:str) -> Faudata:
    '''retrieve faudata from specified path for separation'''

    goto.subf(
        'overlay',
        'mix'
    )
    audio = AudioSegment.from_file(origin+'.wav',format='wav')

    goto.subf(
        'overlay',
        'metadata'
    )
    dataframe = pd.read_csv(origin+'.csv',header=None)
    dataframe.columns = ['Frm', 'Class', 'Track', 'Azmth', 'Elev']

    time_start = dataframe.iloc[0][0]
    time_end = dataframe.iloc[-1][0]
    time_duration = time_end - time_start
    dataframe_duration = dataframe.shape[0]

    print(dataframe)
    print(origin)

    input()

    return Faudata(
        origin=origin,
        audio=audio,
        dataframe=dataframe,
        time_start=time_start,
        time_end=time_end,
        time_duration=time_duration,
        dataframe_duration=dataframe_duration
    )


# SEPARATION #
def retrieve_history(history_origin:str) -> list:
    goto.subf('overlay','history')

    rows = list()
    history_dataframe = pd.read_csv(history_origin+'.csv',header=None)

    for i,row, in history_dataframe.iterrows():
        rows.append(Row(
            faudata1= row[0],
            faudata2= row[1],
            class1= row[2],
            class2= row[3],
            mix= row[4],
            paudata1= row[5],
            paudata2= row[6],
            paudata_mix= row[7],
            overlay_duration= row[8]
        ))

    return History(history_origin, rows)

def retrieve_helper(origin:str,subf:str) -> None:
    '''get the ov2 dataframe to extract the overlayed part'''

    goto.subf(
        'overlay',
        'csv_helper',
        subf
    )

    try:
        temp = pd.read_csv(origin+'_helper.csv',header=None)
        temp.columns = ['Frm', 'Class', 'Track', 'Azmth', 'Elev']
    except:
        print(Fore.RED,origin,Fore.WHITE)
    return temp


def extract_overlayed_dataframe(dataframe:pd.DataFrame,duration:int,even:bool=True) -> pd.DataFrame:

    '''extract dataframe to only even or odd row.
    dataframe given must already a desired length'''

    temp = pd.DataFrame

    if even:
        index = 0
    else:
        index = 1

    for index in range(1,duration+1):
        row = dataframe.iloc[index]
        temp = pd.concat([temp,row])

        index += 2
    
    return temp

def separate(row:Row, maps:list, old_faudata1:Faudata, old_faudata2:Faudata,increment:int) -> list:
    '''separate row history
    return a list of separation_history data'''

    temp_2_rows = pd.DataFrame()

    map_name = str(maps[0])+'_'+str(maps[1])

    paudatas_to_loop = [
        row.paudata1,
        row.paudata2
    ]

    class_to_loop = [
        row.class1,
        row.class2
    ]

    ############# code bu ranny    
    dataMix_path = goto.subf(
        'overlay',
        'mix_wav_tunggal_cut'
    ) + f'\{row.paudata_mix}'
    dataMix = nussl.AudioSignal(dataMix_path)

    for i, (paudata,_class) in enumerate(zip(paudatas_to_loop,class_to_loop)):
        dataSingle_path = goto.subf(
            'overlay',
            'wav_tunggal_cut',
            map_name
        ) + f'\{paudata}'
        dataSingle = nussl.AudioSignal(dataSingle_path)

        mask_data = (
            np.abs(dataSingle.stft()) /
            np.maximum(
            np.abs(dataMix.stft()),
            np.abs(dataSingle.stft())
            ) + nussl.constants.EPSILON
        )

        magnitude, phase = np.abs(dataMix.stft_data), np.angle(dataMix.stft_data)
        masked_abs = magnitude * mask_data
        masked_stft = masked_abs * np.exp(1j * phase)

        drum_est = dataMix.make_copy_with_stft_data(masked_stft)
        drum_est.istft()

        export_path = goto.subf(
            'separation',
            'separation_wav_tunggal_cut',
            map_name
        ) + f'\{paudata[:-4]}_split{i+1}.wav'
        drum_est.write_audio_to_file(export_path,sample_rate=24000)
        ############# code bu ranny

        ############# code ryan
        NUSSL_NAME = export_path.split('\\')[-1]
        goto.subf('separation','separation_wav_tunggal_cut',map_name)
        audio_nussl = AudioSegment.from_file(NUSSL_NAME)

        # get paudata
        paudata1 = [paudata for paudata in old_faudata1.paudata if paudata.sound_class == row.class1][0]
        paudata2 = [paudata for paudata in old_faudata2.paudata if paudata.sound_class == row.class2][0]

        new_dataframe = pd.DataFrame()

        for curr_paudata in old_faudata1.paudata:
            if curr_paudata != paudata1:
                tempe = curr_paudata.dataframe.copy()
                tempe = tempe.reset_index()
                # print(Fore.YELLOW, tempe, Fore.WHITE)
                # input()
                new_dataframe = pd.concat([new_dataframe,tempe])
                # print(Fore.CYAN,new_dataframe,Fore.WHITE)
                # input()
            else:
                print('\t',Fore.CYAN,row.paudata1,row.paudata2,Fore.WHITE)
                dataframe = retrieve_helper(
                    paudata[:-4],
                    map_name
                )

                # print(Fore.GREEN,dataframe,Fore.WHITE)
                # input()

                new_dataframe = pd.concat([new_dataframe,dataframe])
                # print(Fore.CYAN,new_dataframe,Fore.WHITE)
                # input()

                # berbagai macam case
                # if i == 0: # kalau looping ke 1, yang ngrauh cuman concat audio nussl aja
                audio_before = merge_4_channel(old_faudata1.audio)
                audio_before = audio_before[:paudata1.time_start*100]

                audio_after = merge_4_channel(old_faudata1.audio)
                audio_after = audio_after[(paudata1.time_start+row.overlay_duration)*100:]

                new_audio = audio_before + audio_nussl + audio_after 

                print('A',Fore.GREEN,new_audio.duration_seconds,'processed',Fore.WHITE)
                # elif i == 1:
                #     # : HELPER DONE, NOW GET THE DATA AND PROCESS WITH USING DATAFRAME FROM HELPER
        
        # export 
        SEPARATION_NAME = old_faudata1.origin[:-10] + 'mix%03d' % increment + '_ov1'
        export_csv(
            new_dataframe,
            SEPARATION_NAME,
            'separation',
            'metadata_dev'
        )
        export_4_channel_no_duration(
            new_audio,
            SEPARATION_NAME,
            'separation',
            'foa_dev'
        )

        temp_np_row = np.array([
            row.paudata1,
            row.paudata2,
            row.paudata_mix,
            _class,
            SEPARATION_NAME + '.wav'
        ])

        temp_df_row = pd.DataFrame(temp_np_row.reshape(1,-1))
        temp_2_rows = pd.concat([temp_2_rows,temp_df_row])
        # print(temp_2_rows)
        # input()
        increment += 1

    return increment, temp_2_rows


def through_history(history:History, maps:list,old_faudata1:Faudata, old_faudata2:Faudata) -> None:
    '''loop through history and do separation
    creating a separation_history file in each folder'''

    print('Processing',Fore.YELLOW,old_faudata1.origin,old_faudata2.origin,Fore.WHITE)

    n_history = pd.DataFrame() #new history

    increment = 1

    for row in history.row:
        increment, temp_2_rows = separate(row=row,maps=maps,old_faudata1=old_faudata1,old_faudata2=old_faudata2,increment=increment)

        n_history = pd.concat([n_history,temp_2_rows])
    
    export_csv(
        n_history,
        old_faudata1.origin+'_SEPARATE_'+old_faudata2.origin,
        'separation',
        'history'
    )