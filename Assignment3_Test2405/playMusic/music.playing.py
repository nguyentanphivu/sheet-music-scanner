from threading import Thread
import pygame as pg 
import time 
import numpy as np

piano_notes = ['A0',  'B0', 'C1', 'D1', 'E1', 'F1', 'G1',
               'A1',  'B1', 'C2', 'D2', 'E2', 'F2', 'G2',
               'A2',  'B2', 'C3', 'D3', 'E3', 'F3', 'G3',
               'A3',  'B3', 'C4', 'D4', 'E4', 'F4', 'G4',
               'A4',  'B4', 'C5', 'D5', 'E5', 'F5', 'G5',
               'A5',  'B5', 'C6', 'D6', 'E6', 'F6', 'G6',
               'A6',  'B6', 'C7', 'D7', 'E7', 'F7', 'G7',
               'A7',  'B7', 'C8']

white_notes = ['A0', 'B0', 'C1', 'D1', 'E1', 'F1', 'G1',
               'A1', 'B1', 'C2', 'D2', 'E2', 'F2', 'G2',
               'A2', 'B2', 'C3', 'D3', 'E3', 'F3', 'G3',
               'A3', 'B3', 'C4', 'D4', 'E4', 'F4', 'G4',
               'A4', 'B4', 'C5', 'D5', 'E5', 'F5', 'G5',
               'A5', 'B5', 'C6', 'D6', 'E6', 'F6', 'G6',
               'A6', 'B6', 'C7', 'D7', 'E7', 'F7', 'G7',
               'A7', 'B7', 'C8']

black_notes = ['Bb0', 'Db1', 'Eb1', 'Gb1', 'Ab1',
               'Bb1', 'Db2', 'Eb2', 'Gb2', 'Ab2',
               'Bb2', 'Db3', 'Eb3', 'Gb3', 'Ab3',
               'Bb3', 'Db4', 'Eb4', 'Gb4', 'Ab4',
               'Bb4', 'Db5', 'Eb5', 'Gb5', 'Ab5',
               'Bb5', 'Db6', 'Eb6', 'Gb6', 'Ab6',
               'Bb6', 'Db7', 'Eb7', 'Gb7', 'Ab7',
               'Bb7']

black_labels = ['A#0', 'C#1', 'D#1', 'F#1', 'G#1',
                'A#1', 'C#2', 'D#2', 'F#2', 'G#2',
                'A#2', 'C#3', 'D#3', 'F#3', 'G#3',
                'A#3', 'C#4', 'D#4', 'F#4', 'G#4',
                'A#4', 'C#5', 'D#5', 'F#5', 'G#5',
                'A#5', 'C#6', 'D#6', 'F#6', 'G#6',
                'A#6', 'C#7', 'D#7', 'F#7', 'G#7',
                'A#7']

noteLst = ['C5', 'B4', 'C5', 'D5', 'C5', 'A5', 0, 'C5', 'B4', 'C5', 'D5', 'C5', 'G5', 0, 'C5', 'B4', 'C5', 'D5', 'C5', 'F5', 'F5', 'E5', 'F5', 'E5', 'D5', 'C5', 'F5', 'E5', 'C5', 'B4', 'C5', 'D5', 'C5', 'A5', 0, 'C5', 'B4', 'C5', 'D5', 'C5', 'G5', 0, 'C5', 'B4', 'C5', 'D5', 'C5', 'F5', 0, 'F5', 'E5', 'F5', 'E5', 'D5', 'C5', 'D5', 'C5']
durationLst = [8, 8, 8, 8, 8, 8, 4, 8, 8, 8, 8, 8, 8, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 8, 8, 8, 8, 8, 8, 4, 8, 8, 8, 8, 8, 8, 4, 8, 8, 8, 8, 8, 8, 8, 8]
durationLst = np.array(durationLst)
durationLst = 1/durationLst
scalingFactor = 2.5
durationLst = durationLst * scalingFactor

st = time.time()
pg.mixer.init()
pg.init()

pg.mixer.set_num_channels(len(noteLst))

def play_notes(notePath, duration):
	time.sleep(0) # make a pause 
	pg.mixer.Sound(notePath).play()
	time.sleep(duration) # Let the sound play 
	print(notePath) # To see which note is now playing
	
path  = r"C:\Users\nguye\Downloads\notes-20230523T175502Z-001\notes\\"

cnt = 0	# A counter to delay once a line is finished as there
# are 6 total lines

th = {}
# for t in test:
# 	th[t] = Thread(target = play_notes,args = (path+'{}.wav'.format(t),0.1))
# 	if t in white_notes:
# 	    # th[t] = Thread(target = play_notes,args = (path+'{}.wav'.format(t),0.1))
#             th[t].start()
#             th[t].join()
a = bool 
for i in range(len(noteLst)):
	note = noteLst[i]
	noteDuration = durationLst[i]
	if note not in white_notes:
		cnt += note 
		print(cnt)
		if note == 0:
			time.sleep(noteDuration)
	if note in white_notes:
		th[note] = Thread(target = play_notes, args = (path+'{}.wav'.format(note), noteDuration))
		th[note].start()
		th[note].join()
		if cnt == 4:
			cnt = 0
		if a == True: 
			# time.sleep(0.5)
			pass


et = time.time()

rt = et - st 
print(rt)



	# if cnt%7==0:
	# 	print("---Long Pause---")
	# 	time.sleep(0.2) # Let the sound play for the last note of each line
		
	# cnt+=1

