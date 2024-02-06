
<div align="center">
  <img width="500px" src="imgs/logo.png"/>
  
  # RoboGen: Towards Unleashing Infinite Data for Automated Robot Learning via Generative Simulation
</div>

---
<div align="center">
  <img src="imgs/teaser.png"/>
</div> 

<p align="left">
    <a href='http://arxiv.org/abs/2311.01455'>
      <img src='https://img.shields.io/badge/Paper-arXiv-green?style=plastic&logo=arXiv&logoColor=green' alt='Paper arXiv'>
    </a>
    <a href='https://robogen-ai.github.io/'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=plastic&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
</p>
This is the official repo for the paper:

> **[RoboGen: Towards Unleashing Infinite Data for Automated Robot Learning via Generative Simulation](https://robogen-ai.github.io/)**  
> [Yufei Wang*](https://yufeiwang63.github.io/), [Zhou Xian*](https://zhou-xian.com/), [Feng Chen*](https://robogen-ai.github.io/), [Tsun-Hsuan Wang](https://zswang666.github.io/), [Yian Wang](https://wangyian-me.github.io/), [
Katerina Fragkiadaki](https://www.cs.cmu.edu/~katef/), [Zackory Erickson](https://zackory.com/), [David Held](https://davheld.github.io/), [Chuang Gan](https://people.csail.mit.edu/ganchuang/)   

RoboGen is a **self-guided** and **generative** robotic agent that autonomously proposes **new tasks**, generates corresponding **environments**, and acquires **new robotic skills** continuously.

RoboGen is powered by [Genesis](https://github.com/Genesis-Embodied-AI/Genesis), a multi-material multi-solver generative simulation engine for general-purpose robot learning. 
Genesis is still under active development and will be released soon. This repo contains a re-implementation of RoboGen using PyBullet, containing generation and learning of rigid manipulation and locomotion tasks. Our full pipeline containing soft-body manipulation and more tasks will be released later together with Genesis.

We are still in the process of cleaning the code & testing, please stay tuned!

## Table of Contents
- [Setup](#setup)
  - [RoboGen](#RoboGen)
  - [OMPL](#Open-Motion-Planning-Library)
  - [Dataset](#dataset)
- [Let's Rock!](#lets-rock)
  - [Automated Task Generation & Skill Learning](#One-click-for-all)
  - [Generate Tasks](#Generate-tasks)
  - [Learn Skills](#Learn-skills)
  - [Pre-generated Tasks](#Pre-generated-tasks)
## Setup
### RoboGen
Clone this git repo.
```
git clone https://github.com/Genesis-Embodied-AI/RoboGen.git
```
We recommend working with a conda environment.
```
conda env create -f environment.yaml
conda activate robogen
```
If installing from this yaml file doesn't work, manual installation of missing packages should also work.

### Open Motion Planning Library
RoboGen leverages [Open Motion Planning Library (OMPL)](https://ompl.kavrakilab.org/) for motion planning as part of the pipeline to solve the generated task. 
To install OMPL, run
```
./install_ompl_1.5.2.sh --python
```
which will install the ompl with system-wide python. Note at line 19 of the installation script OMPL requries you to run `sudo apt-get -y upgrade`. This might cause trouble for your system packages, so you could probably comment this line during installation (the installation might fail, not fully tested with the line commented).
Then, export the installation to the conda environment to be used with RoboGen:
```
echo "path_to_your_ompl_installation_from_last_step/OMPL/ompl-1.5.2/py-bindings" >> ~/miniconda3/envs/robogen/lib/python3.9/site-packages/ompl.pth
```
remember to change the path to be your ompl installed path and conda environment path.

Note: if you are having trouble building OMPL from source, the maintainer of OMPL has suggested to use the prebuilt python wheels at here: https://github.com/ompl/ompl/releases/tag/prerelease. They also plan to release on PyPI soon, so stay tuned. For more info, check [this issue](https://github.com/Genesis-Embodied-AI/RoboGen/issues/8#issuecomment-1918092507).

### Dataset
RoboGen uses [PartNet-Mobility](https://sapien.ucsd.edu/browse) for task generation and scene population. We provide a parsed version [here](https://drive.google.com/file/d/1d-1txzcg_ke17NkHKAolXlfDnmPePFc6/view?usp=sharing) (which parses the urdf to extract the articulation tree as a shortened input to GPT-4). After downloading, please unzip it and put it in the `data` folder, so it looks like `data/dataset`.

For retrieving objects from objaverse, we embed object descriptions from objaverse using [SentenceBert](https://www.sbert.net/). 
If you want to generate these embeddings by yourself, run
```
python objaverse_utils/embed_all_annotations.py
python objaverse_utils/embed_cap3d.py
python objaverse_utils/embed_partnet_annotations.py
```
We also provide the embeddings [here](https://drive.google.com/file/d/1dFDpG3tlckTUSy7VYdfkNqtfVctpn3T6/view?usp=sharing) if generation takes too much time. After downloading, unzip and put it under `objaverse_utils/data/` folder, so it looks like 
```
objaverse_utils/data/default_tag_embeddings_*.pt
objaverse_utils/data/default_tag_names_*.pt
objaverse_utils/data/default_tag_uids_*.pt
objaverse_utils/data/cap3d_sentence_bert_embeddings.pt
objaverse_utils/data/partnet_mobility_category_embeddings.pt
```


## Let's Rock!
### One click for all
Put your OpenAI API key at the top of `gpt_4/query.py`, and simply run
```
source prepare.sh
python run.py
``` 
RoboGen will then generate the task, build the scene in pybullet, and solve it to learn the corresponding skill.  
If you wish to generate manipulation tasks relevant to a specific object, e.g., microwave, you can run  
```
python run.py --category Microwave
```

### Generate tasks
If you wish to just generate the tasks, run
```
python run.py --train 0
```
which will generate the tasks, scene config yaml files, and training supervisions. The generated tasks will be stored at `data/generated_tasks_release/`.  
If you want to generate task given a text description, you can run
```
python gpt_4/prompts/prompt_from_description.py --task_description [TASK_DESCRIPTION] --object [PARTNET_ARTICULATION_OBJECT_CATEGORY]
``` 
For example,
```
python gpt_4/prompts/prompt_from_description.py --task_description "Put a pen into the box" --object "Box"
```

### Learn skills
If you wish to just learn the skill with a generated task, run
```
python execute.py --task_config_path [PATH_TO_THE_GENERATED_TASK_CONFIG] # for manipulation tasks
python execute_locomotion.py --task_config_path [PATH_TO_THE_GENERATED_TASK_CONFIG] # for locomotion tasks
```
For example,
```
python execute.py --task_config_path example_tasks/Change_Lamp_Direction/Change_Lamp_Direction_The_robotic_arm_will_alter_the_lamps_light_direction_by_manipulating_the_lamps_head.yaml  
python execute_locomotion.py --task_config_path example_tasks/task_Turn_right/Turn_right.yaml
```

### Pre-generated tasks
In `example_tasks` we include a number of generated tasks from RoboGen. We hope this could be useful for, e.g., language conditioned multi-task learning & transfer learning & low-level skill learning. We hope to keep updating the list! 

## Acknowledgements
- The interface between OMPL and pybullet is based on [pybullet_ompl](https://github.com/lyfkyle/pybullet_ompl).
- Part of the objaverse annotations are from [Scalable 3D Captioning with Pretrained Models](https://arxiv.org/abs/2306.07279)

## Citation
If you use find this codebased/paper useful for your research, please consider citing:
```
@article{wang2023robogen,
  title={Robogen: Towards unleashing infinite data for automated robot learning via generative simulation},
  author={Wang, Yufei and Xian, Zhou and Chen, Feng and Wang, Tsun-Hsuan and Wang, Yian and Fragkiadaki, Katerina and Erickson, Zackory and Held, David and Gan, Chuang},
  journal={arXiv preprint arXiv:2311.01455},
  year={2023}
}
```


