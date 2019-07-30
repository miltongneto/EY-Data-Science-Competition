# EY-Data-Science-Competition
Solution for EY NextWave Data Science Competition 2019

## Challenge
This competition was to improve the Smart Cities with Data Science solution. The goal was predict how many people are in the city center between 15:00 and 16:00 in city center of Atlanta using geolocation records.

## Results
The code in this repository is a clean and summarized version of my work. I put here only the main approaches that I used on final version submitted, other tecnicals ware omitted.

Final ranking with F1-score:

|Leaderboard     |Score     |Brazilian Ranking| Global Ranking|
|----------------|----------|-----------------|---------------|
|Public          |0.89547   |1st              |17th           |
|Private         |0.88571   |2nd              |25th           |

I presenting my project in the Brazilian final and I finished in 4th place. It was an amazing experience!

## Methodology

### Change grain and become trajectories as characteristics. 
First, I will explain the most important approach. In the initial data set, the grain is the trajectory. But each prediction was performed to a hash (user or device). So, for build the final data set, I used all trajectories as a sequence, they become features in descending order. The intention is to maintain the order of trajectories in relation to the last ones and give more importance to them.

Each trajectory has some information witch generates many attributes, or columns, to be more exact. 

### Feature Engineering
This phase is very important for final result, the creation of features can be improve the performance. It is possible include human knowledge and create important features. The majority features was created by each trajectoy, remembering that for traj_last some features are not created, because the exit point is unknown (a posteriori information).

#### Standard features:
These features are named “standard” because they use the raw information. This is, the information is maintained.
The features are:
- Vmax
- Vmin
- Vmean
- x_entry
- y_entry
- x_exit
- y_exit

#### Time features:
- Duration of the trajectory
- Hour and minute
- Period of the day:
  - night, early morning, morning, afternoon

#### Working with the points:
With entry and exit points, some features were created, manipulating points and including human knowledge for answer the main question, “the user (hash) is in center”? 
Remembering that for traj_last some features are not created, because the exit point is unknown.

- Is the center?
- Travelled distance
- Distance from the center
- Distance from the boundary center
- Distance from the center point
- Approach to the center
- Approach to the center point
- My velocity mean
- My_Vmean = Travelled distance ÷ Duration of the trajectory

#### Aggregation features:
Each user (hash) has the collection of trajectories, and theses becomes features. But some information about the set is important, so some functions have been applied to extract some relevant information.
- Number of trajectories
- Average duration
- Average distance
- Average velocity (with My_Vmean)

### Modeling
This is Machine Learning phase. Many classifiers and techniques were tested. The final was modeled with xbgoost (Extreme Gradient Boosting). 

Parameters:
- n_estimators=1000
- learning_rate=0.05
- max_depth=6
- colsample_bytree=0.9 
- subsample=0.8
- min_child_weight=4
- reg_alpha=0.005
