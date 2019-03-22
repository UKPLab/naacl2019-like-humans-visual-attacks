# Text Processing Like Humans Do: Visually Attacking and Shielding NLP Systems

This repository contains selected code and data for our NAACL 2019 long paper on [Visually Attacking and Shielding NLP Systems]().

## Citation

```
@inproceedings{Eger_naacl:2019,
            title = {Text Processing Like Humans Do: Visually Attacking and Shielding NLP Systems},
            year = {2019},
          author = {Steffen Eger and G{\"o}zde G{\"u}l {\c S}ahin and Andreas R{\"u}ckl{\'e} and Ji-Ung Lee and Claudia Schulz and Mohsen Mesgar and Krishnkant Swarnkar and Edwin Simpson and Iryna Gurevych},
       booktitle = {Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics},
           month = {Februar},
         journal = {NAACL 2019},
             url = {http://tubiblio.ulb.tu-darmstadt.de/111643/}
}
```
> **Abstract:** Visual modifications to text are often used to obfuscate offensive comments in social media (e.g., ''!d10t'') or as a writing style (''1337'' in ''leet speak''), among other scenarios.
We consider this as a new type of adversarial attack in NLP,  a 
setting
to which humans are very robust, 
as our experiments with both simple and more difficult visual perturbations demonstrate.
We 
investigate 
the impact of visual adversarial attacks
on current NLP systems on character-, word-, and sentence-level tasks, 
showing 
that both neural and non-neural models are, in contrast to humans, 
extremely sensitive to such attacks, 
suffering performance decreases of up to 82\%. 
We then explore three shielding methods---visual character embeddings, adversarial training, and rule-based recovery---which
substantially improve the robustness of the models.
However, the shielding methods still fall behind performances achieved in 
non-attack scenarios, which demonstrates the difficulty of dealing with visual attacks.


Contact person: Steffen Eger, eger@ukp.informatik.tu-darmstadt.de, Gözde Gül Sahin, Andreas Rückle, Ji-Ung Lee, ...

https://www.ukp.tu-darmstadt.de/

https://www.tu-darmstadt.de/


Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions. 

## Project Description

### Embeddings

Visual Character Embeddings (576 dimensional): 

### G2P model

See `code/G2P`

### VIPER

In `code/VIPER`:

```python3 disturb_plain.py -e efile.norm -p 0.4 --perturbations-file dummy_store.txt```

### Sanity Check / Semantic Similarity 

See Appendix
