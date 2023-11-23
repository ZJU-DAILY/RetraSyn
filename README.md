# RetraSyn

This is our Python implementation for the paper:
> Yujia Hu, Yuntao Du, Zhikun Zhang, Ziquan Fang, Lu Chen, Kai Zheng, Yunjun Gao. Real-Time Trajectory Synthesis with Local Differential Privacy. 2023.


## Environment Requirements

+ Python >= 3.11.3 (Anaconda3 is recommended)
+ numpy == 1.24.3


## Dataset

### Dataset Statistics

| Dataset    | Size      | # of Points | Average  Length | Timestamps |
| ---------- | --------- | ----------- | --------------- | ---------- |
| T-Drive    | 232,640   | 3,167,316   | 13.61           | 886        |
| Oldenburg  | 260,000   | 15,597,242  | 59.98           | 500        |
| SanJoaquin | 1,010,000 | 55,854,936  | 55.30           | 1,000      |

### Dataset Format

For each dataset, there are three corresponding files:

+ `{datasetname}.xz`: contains all trajectory streams in the format of List[List[Tuple[float,float,int]]]

  [Traj1:[(x,y,t),(x,y,t),...],

  Traj2:[(x,y,t),(x,y,t),...],......]

+ `{datasetname}_transition.xz` transforms the original streams into transition states and aligns with the discrete timestamps. The format is List[List[Tuple[float,float,float,float,int]]]:

  [Timestamp1:[(x1,y1,x2,y2,flag)],...],

  Timestamp2:[(x1,y1,x2,y2,flag)],...],......]

  Each 5-tuple represents a transition from (x1,y1) to (x2,y2), flag $\in\{0,1,2\}$ represents movement transition, entering transition and quitting transition.

+ `{datasetname}_transition_id.xz` is similar to `{datasetname}_transition.xz`. The difference is the use of a 6-tuple, which add an additional integer to represent user id.

### T-Drive

+ T-Drive records trajectories of 10,357 taxis in Beijing during one week.
+ The dataset is available in `./data/`, containing the above mentioned three data files.

### Oldenburg

+ Oldenburg is a synthetic dataset simulated by Brinkhoff's network-based moving objects generator. It is based on the map of Oldenburg city, Germany.
+ Please refer to  http://iapg.jade-hs.de/personen/brinkhoff/generator/ to generate the Oldenburg dataset. The parameter used in generation are as follows:
  + obj./begin: 10000	0
  + obj./time: 500 	 0
  + maximum time: 500
  + classes: 1    0
  + max. speed div.: 50
  + report probability: 1000
+ The generated dataset needs to be transformed into the above mentioned data format, containing three data files and located in `./data/`

### SanJoaquin

+ SanJoaquin is based on the map of San Joaquin County, USA.
+ Please refer to  http://iapg.jade-hs.de/personen/brinkhoff/generator/ to generate the SanJoaquin dataset. The parameter used in generation are as follows:
  + obj./begin: 10000	0
  + obj./time: 1000 	 0
  + maximum time: 1000
  + classes: 1    0
  + max. speed div.: 50
  + report probability: 1000
+ The generated dataset needs to be transformed into the above mentioned data format, containing three data files and located in `./data/`

## Reproducibility & Run

+ Please make sure the data files are located in `./data/` and create directories called `./data/syn_data/{datasetname}/` before running RetraSyn.

+ There're two executable files named `./code/RetraSyn_b.py` and `./code/RetraSyn_b.py`, containing the budget-division strategy and population-division strategy, respectively. Here's an example of running RetraSyn:

  ```
  python RetraSyn_b.py --dataset tdrive --grid_num 6 --epsilon 1.0 --w 20 --method retrasyn --multiprocessing
  ```

  The program will generate a synthesized dataset and save it in `./data/syn_data/{datasetname}/`

+ To evaluate the synthesized dataset, run `./code/evaluation_b.py` or `./code/evaluation_b.py`:

  ```
  python evaluation_b.py --dataset tdrive --grid_num 6 --epsilon 1.0 --w 20 --method retrasyn --phi 20 --multiprocessing
  ```

## Configurations

The parameters includes:

+ --dataset: dataset name, 'tdrive' for default. It should match the name of data files.
+ --epsilon: privacy budget
+ --grid_num: discretization granularity $K$
+ --w: window size
+ --phi: size of evaluation time range
+ --multiprocessing: whether to use multiprocessing in preprocessing and evaluation
+ --method
  + retrasyn: our proposed approach
  + lbd (only in RetraSyn_b.py)
  + lba (only in RetraSyn_b.py)
  + lpd (only in RetraSyn_p.py)
  + lpa (only in RetraSyn_p.py)

