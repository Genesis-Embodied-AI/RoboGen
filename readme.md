
<div align="center">
  <img width="500px" src="imgs/logo.png"/>
  
  # RoboGen: Towards Unleashing Infinite Data for Automated Robot Learning via Generative Simulation
  ### ICML 2024
</div>

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


**[RoboGen: Towards Unleashing Infinite Data for Automated Robot Learning via Generative Simulation](https://robogen-ai.github.io/)**  
[Yufei Wang*](https://yufeiwang63.github.io/), [Zhou Xian*](https://zhou-xian.com/), [Feng Chen*](https://robogen-ai.github.io/), [Tsun-Hsuan Wang](https://zswang666.github.io/), [Yian Wang](https://wangyian-me.github.io/), [
Katerina Fragkiadaki](https://www.cs.cmu.edu/~katef/), [Zackory Erickson](https://zackory.com/), [David Held](https://davheld.github.io/), [Chuang Gan](https://people.csail.mit.edu/ganchuang/)  
published at ICML 2024  

RoboGen is a **self-guided** and **generative** robotic agent that autonomously proposes **new tasks**, generates corresponding **environments**, and acquires **new robotic skills** continuously.

RoboGen is powered by [Genesis](https://github.com/Genesis-Embodied-AI/Genesis), a multi-material multi-solver generative simulation engine for general-purpose robot learning. 
Genesis is still under active development and will be released soon. This repo contains a re-implementation of RoboGen using PyBullet, containing generation and learning of rigid manipulation and locomotion tasks. Our full pipeline containing soft-body manipulation and more tasks will be released later together with Genesis.

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

Note: if you are having trouble building OMPL from source, the maintainer of OMPL has suggested to use the prebuilt python wheels at here: https://github.com/ompl/ompl/releases/tag/prerelease. Use the wheel that matches your python version, e.g., if you are using python3.9, download [this wheel](https://github.com/ompl/ompl/releases/download/prerelease/ompl-1.6.0-cp39-cp39-manylinux_2_28_x86_64.whl), and then run `pip install ompl-1.6.0-cp39-cp39-manylinux_2_28_x86_64.whl`.
They also plan to release on PyPI soon, so stay tuned. For more info, check [this issue](https://github.com/Genesis-Embodied-AI/RoboGen/issues/8#issuecomment-1918092507).

To install OMPL from source, run
```
./install_ompl_1.5.2.sh --python
```
which will install the ompl with system-wide python. Note at line 19 of the installation script OMPL requries you to run `sudo apt-get -y upgrade`. This might cause trouble for your system packages, so you could probably comment this line during installation (the installation might fail, not fully tested with the line commented).
Then, export the installation to the conda environment to be used with RoboGen:
```
echo "path_to_your_ompl_installation_from_last_step/OMPL/ompl-1.5.2/py-bindings" >> ~/miniconda3/envs/robogen/lib/python3.9/site-packages/ompl.pth
```
remember to change the path to be your ompl installed path and conda environment path.



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

After running the above commands, to check the skill learning results, 
- For manipulation tasks,
  - For substeps using RL: we use SAC implemented in [RAY-RLlib](https://docs.ray.io/en/latest/rllib/index.html) for RL training. Ray RLlib will log the learning progress at data/local/ray_results/{task_name}_{time_stamp}. You can check that folder for visualizations of the reward curves. For example, you can plot episode_reward_mean field in progress.csv to show the evaluation episode reward during training. Ray RLlib also uses tensorboard to store the results in events.out.tfevents*, so one can also use tensorboard for visualization.
The final skill learned by RL will be stored at {path_to_your_task}/RL_sac/{time_stamp}/{substep_name}, e.g., data/example_tasks/close_the_oven_door_Oven_101940_2023-09-21-22-28-23/task_close_the_oven_door/RL_sac/2023-09-22-02-05-05/close_the_oven_door. A gif showing the execution of the learned RL policy, all the simulation states during the policy execution, and the best policy weights will be stored there. RL training can take 1-2 hours for convergence. We are working to switch to better RL learning libraries, and will update the repo once the tests are done. 
  - For substeps using motion planning based action primitives: The results of the primitives will be stored at {path_to_your_task}/primitive_states/{time_stamp}/{substep_name}, e.g., data/example_tasks/close_the_oven_door_Oven_101940_2023-09-21-22-28-23/task_close_the_oven_door/primitive_states/2023-10-06-03-00-07/grasp_the_oven_door. A gif showing the execution of the primitive, as well as all the simulation states during the primitive execution will be stored. Motion planning based action primitive should take less than 10 minutes to finish. 
- For locomotion tasks, we use CEM to solve it. The learning results should be stored at {path_to_your_task}/cem/, e.g., data/generated_tasks/locomotion_2023-10-30-17-43-31/task_Turn_right/cem/. A mp4 showing the results and all simulation states during the execution will be stored. CEM usally takes ~10 minutes to finish. NOTE: The program might end with an error message of "AttributeError:'NoneType' object has no attribute 'dumps'". This can be safely ignored if the mp4 file is successfully generated. 

### Pre-generated tasks
In `example_tasks` we include a number of generated tasks from RoboGen. We hope this could be useful for, e.g., language conditioned multi-task learning & transfer learning & low-level skill learning. We hope to keep updating the list! 

## Acknowledgements
- The interface between OMPL and pybullet is based on [pybullet_ompl](https://github.com/lyfkyle/pybullet_ompl).
- Part of the objaverse annotations are from [Scalable 3D Captioning with Pretrained Models](https://arxiv.org/abs/2306.07279)

## Citation
If you find this codebase/paper useful for your research, please consider citing:
```
@article{wang2023robogen,
  title={Robogen: Towards unleashing infinite data for automated robot learning via generative simulation},
  author={Wang, Yufei and Xian, Zhou and Chen, Feng and Wang, Tsun-Hsuan and Wang, Yian and Fragkiadaki, Katerina and Erickson, Zackory and Held, David and Gan, Chuang},
  journal={arXiv preprint arXiv:2311.01455},
  year={2023}
}
```


