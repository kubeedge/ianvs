# How to config algorithm


## The configuration of hyperparameter in algorithm

The following is an example of hyperparameter configuration: 
```yaml
learning_rate: 0.1, 0.2
momentum: 0.8, 0.9
```
Ianvs will test for all the hyperparameter combination, that means it will run all the following 4 test:

| Num. | learning_rate | momentum |
|------|---------------|----------|
| 1    | 0.1           | 0.8      |
| 2    | 0.1           | 0.9      |
| 3    | 0.2           | 0.8      |
| 4    | 0.2           | 0.9      |

Currently, Ianvs is not restricted to validity of the hyperparameter combination. 
That might lead to some invalid parameter combination,
and it is controlled by the user himself. In the further version of Ianvs, it will support excluding invalid parameter combinations to improve efficiency. 
 

