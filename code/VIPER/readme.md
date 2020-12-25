# VIPER (VIsual PERturber)

We offer three different types of VIPER attacks, each following a different "embedding space": (i) ICES is an image based embedding space (two characters are close if they have a similar image); (ii) DCES is a description based embedding space (two characters are close if they have the same textual Unicode description); (iii) ECES is an "easy" character space (each standard character has one neighbor, given by a diacritic added to the character). 

Note that all three embedding spaces are informed by visual similarity (to different degrees).

See the paper for more details. 

## VIPER(p,ICES)

Run VIPER(p,ICES) using

```
python3 viper_ices.py -e vce.normalized -p 0.4 --perturbations-file dummy_store.txt < sample.txt
```

* `vce.normalized` is a dense embedding file. 
* `p` is the character perturbation probability. 
* `dummy_store.txt` stores the disturbed characters used (can be used to create OOV test disturbances).

## VIPER(p,DCES)

To perturb an input text by selecting random neighboring tokens in the DCES, run the following script for a plain text file:

```
python3 viper_dces.py -p 0.4 -d sample.txt
```
Or run it on some CoNLL data:

```
python3 viper_dces.py -p 0.4 -d sample_conll.txt --conll
```

* -p `<arg>`: argument gives the probability of perturbing each input character

* -d `<arg>`: argument gives the path to the text data you want to perturb

* --conll: indicates whether data (both the input and output) is in CONLL (tab) separated format. If conll format is not set, the perturb text is printed as plain text with no formatting.

The descriptions are read from NamesList.txt to construct the DCES. Perturbed data is printed to stdout.


## VIPER(p,ECES)

Run VIPER(p,ECES) using

```
python3 viper_eces.py 0.9 selected.neighbors < sample.txt
```
The second argument is the character perturbation probability.



# Drawing an image for a character

Run

```
python3 mydraw_individual.py < mychars
```

* `mychars` is a line separated list with characters

