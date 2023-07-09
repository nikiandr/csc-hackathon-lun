# CSC Hackathon 2023 - LUN track solution
## Team GARCH (Andrii Yerko, Nikita Fordui, Andrii Shevtsov, Sofiia Shaposhnikova)
Second place solution for LUN track of [CSC Hackathon 2023](https://csc23.hackathon.expert/).
### Task
The task for the hackathon was near duplicate image comparison. The model should take two same slightly changed images (different gamma, watermark, etc.) and output one if the images are the same, and 2 - if different.
![img1.png](./img1.png)
### Solution
Our team's solution solved this task and placed second on the private leaderboard of [the competition](https://www.kaggle.com/competitions/copy-of-csc-hackathon-2023-lunua-task-2). Our solution is based on different image hashing and keypoint detection algorithms and combining results into gradient boosting model.
![img2.png](./img2.png)
Presentation for the solution can be found [here](./presentation.pdf).

### Repository structure

**data**  folder contains all data splits and other *.csv* files used in development e.g. test submission format. 

**code** folder contains all the code produced during the hackathon. All code files split into 5 groups based on the part of the solution they are connected to. Each group has its name prefix:
- 0 - data analysis and preprocessing
- 1 - hash and key points experiments
- 2 - neural networks experiments
- 3 - boosting experience
- 4 - evaluation and MVP

Same folder also contains some util code for smoother workflow.

### Run

To reproduce the results of the projects, you can use virtual environment with *requirements.txt* file. 

First start with cloning this GitHub repository

To create virtual environment:
```bash
python -m venv <venv_name>
```

To install all the necessary dependencies, first, activate the environment:
```bash
source <venv_name>/bin/activate
```

After that, to install the dependencies run:
```bash
pip install -r requirements.txt
```

All the described steps are described for Linux operating system.
