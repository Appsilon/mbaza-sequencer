# mbaza-sequencer

A small post-processing script to convert image predictions to sequence-based predictions based on EXIF timestamps.

## Installation

The package can be installed directly from Github:

```
pip install git+https://github.com/Appsilon/mbaza-sequencer.git
```

## Running

Once installed, run the `mbaza_sequencer` script to process the `.csv` output from Mbaza, for example:

```
mbaza_sequencer /path/to/csv /path/to/images --max_images 50 --max_delay 3
```

This will output a new csv file (appended with `"_sequenced"`) which contains sequence information and predictions.

For help with commands run:

```
mbaza_sequencer -h
```