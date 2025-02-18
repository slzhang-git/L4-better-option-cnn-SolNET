# SolNET
```
Implementation of the paper "SolNet: A Convolutional Neural Network for Detecting Dust on Solar Panels"
Authors:
	Md. Saif Hassan Onim,
	Zubayar Mahatab Md Sakif,
	Adil Ahnaf,
	Ahsan Kabir,
	Rafina Afreen,
	Sumaita Tanjim Hridy,
	Mahtab Hossain,
	Abul Kalam Azad,
	Taskeed Jabid and
	Md Sawkat Ali
```

<p align="center"><img src="data/workflow_all.png" width="90%" alt="" style="background-color:blue"/></p>
<em align="center"> Workflow </em>
<p align="center"><img src="data/solnet_arch.png" width="90%" alt="" style="background-color:white"/></p>
<em align="center"> SolNet Architecture </em>

## Experimental Steps
- Get the dataset from here: [Dataset](https://drive.google.com/drive/folders/12Q3MBI8SPw0vHsO_kkS5izkxw0F7tXx4?usp=sharing)

- Keep the dataset in the dataset folder. You can keep your own dataset for testing purpose.

- Run the 'train.py' from 'utils' folder to train the model





## Cite us with the following bibtex:
```
@Article{SolNet2022,
AUTHOR = {Onim, Md Saif Hassan and
	Sakif, Zubayar Mahatab Md and
	Ahnaf, Adil and
	Kabir, Ahsan and
	Azad, Abul Kalam and
	Oo, Amanullah Maung Than and
	Afreen, Rafina and
	Hridy, Sumaita Tanjim and
	Hossain, Mahtab and
	Jabid, Taskeed and
	Ali, Md Sawkat},
TITLE = {SolNet: A Convolutional Neural Network for Detecting Dust on Solar Panels},
JOURNAL = {Energies},
VOLUME = {16},
YEAR = {2023},
NUMBER = {1},
ARTICLE-NUMBER = {155},
URL = {https://www.mdpi.com/1996-1073/16/1/155},
ISSN = {1996-1073},
DOI = {10.3390/en16010155}
}
```

# File structure
--The folder "dataset" will include the clean and dirty PV panel images to train a classification model from

--The folder "models" will contain models trained by the algorithm

--The folder "utils" contains scripts to train the classification model, where the file "train.py" is the mainfunction file and "model.py" is invoked by "model.py"
