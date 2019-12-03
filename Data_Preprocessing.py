import os, glob
import pretty_midi
import pypianoroll

path = '/Users/jooyoungson/Datasets/midikar'

a = glob.glob(path + '/**')
print(type(a[0]))

new_beat = pretty_midi.PrettyMIDI(a[0])
for instrument in new_beat.instruments:
    print(instrument)
    print(instrument.notes)
    for note in instrument.notes:
        i = instrument.notes[0]
        print(i)
#        print(note.velocity, note.pitch, note.start, note.end)
    break
