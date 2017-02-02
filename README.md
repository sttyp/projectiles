# projectiles
Dependencies:
This project requires following python libraries to run:
numpy
pandas
tensorflow

Files:
projectiles.csv: original data file recording positions of projectiles

data_process.py: process data and organise it such that each row contains consecutive three projectile positions, namely x(i),y(i),x(i+1),y(i+1),x(i+2),y(i+2) where i is an arbitary time point

data.csv: output of data processed by data_process.py

proj.py: read data from data.csv and train a model that can predict the ball postion x(i+2),y(i+2) given two consecutive positions

result.csv: predict result of projectile position sequence starting at position x(0)=0.0,y(0)=0.0 and x(1)=0.707106781187,y(1)=0.658106781187
