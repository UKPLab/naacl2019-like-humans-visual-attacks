# VIPER(p,ICES)

Run VIPER(p,ICES) using

```python3 disturb_plain.py -e efile.norm -p 0.4 --perturbations-file dummy_store.txt```

* `efile.norm` is a dense embedding file. 
* `p` is the perturbation probability. 
* `dummy_store.txt` stores the disturbed characters used (can be used to create OOV test disturbances).

# Getting an image for a character

Run

```python3 mydraw_individual.py < mychars```

* `mychars` is a line separated list with characters

 
