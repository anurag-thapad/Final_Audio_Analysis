import pickle
import numpy
from music21 import instrument, note, stream, chord, converter

# function to test for unknown file with the trained model
# the unknown file may have some notes on which the model is not trained on.
# This needs to be checked.
def check_notes(notes, file_path):
    """
    notes: the notes file on which model is trained
    file_path: The path of unknown midi song
    """
    test_notes = []
    test_midi = converter.parse(file_path)
    notes_to_parse = None
    try: # file has instrument parts
        s2 = instrument.partitionByInstrument(test_midi)
        notes_to_parse = s2.parts[0].recurse() 
    except: # file has notes in a flat structure
        notes_to_parse = test_midi.flat.notes

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            test_notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            test_notes.append('.'.join(str(n) for n in element.normalOrder))

    test_notes_unq = set(test_notes)
    # test_notes_unq is the list of unique notes in test_notes
    check =  all(item in notes for item in test_notes_unq)
    return check

