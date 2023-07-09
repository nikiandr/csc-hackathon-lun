# CSC Hackathon 2023 - LUN track solution
## Team GARCH (Andrii Yerko, Nikita Fordui, Andrii Shevtsov, Sofiia Shaposhnikova)
Second place solution for LUN track of [CSC Hackathon 2023](https://csc23.hackathon.expert/).
### Task
The task for the hackathon was near duplicate image comparison. The model should take two same slightly changed images (different gamma, watermark, etc.) and output one if the images are the same, and 2 - if different.
![img1.png](./img1.png)
### Solution
Our team's solution solved this task and placed second on the private leaderboard of [the competition](https://www.kaggle.com/competitions/copy-of-csc-hackathon-2023-lunua-task-2). Our solution is based on different image hashing and keypoint detection algorithms and combining results into gradient boosting model.
Presentation for the solution can be found [here](./presentation.pdf).
