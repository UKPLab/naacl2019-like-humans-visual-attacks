# VIPER(p,ICES)

Run VIPER(p,ICES) using

```python3 disturb_plain.py -e efile.norm -p 0.4 --perturbations-file dummy_store.txt < sample.txt```

* `efile.norm` is a dense embedding file. 
* `p` is the perturbation probability. 
* `dummy_store.txt` stores the disturbed characters used (can be used to create OOV test disturbances).

# VIPER(p,DCES)

To perturb an input text by selecting random neighboring tokens in the DCES, run the following script:

```
python3 disturb_gold.py -p 0.4 -d ../G2P_data/train.1k --conll
```

* -p `<arg>`: argument gives the probability of perturbing each input character

* -d `<arg>`: argument gives the path to the text data you want to perturb

* --conll: indicates whether data (both the input and output) is in CONLL (tab) separated format. If conll format is not set, the perturb text is printed as plain text with no formatting.

The descriptions are read from NamesList.txt to construct the DCES. Perturbed data is printed to stdout.


# Drawing an image for a character

Run

```python3 mydraw_individual.py < mychars```

* `mychars` is a line separated list with characters

