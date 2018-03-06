# Distributed Training Framework with Tensorflow
- - -

## Run Example
Traing MLP with titanic dataset.
The running script is supposed to runing parameter server and worker server at the same single machine.
Change the shell script with your environments.


### How to train.
```bash
# Machine 1.
./run_ps_0.sh
```

```bash
# Machine 2.
./run_wk_0.sh
```

### How to generate deploy model.
Generate deploy model with saved model.

```bash
./run_make_deploy.sh

```

## How to use my own model? 
working...


- - -

### Author
+ Sung-ju Kim
+ goddoe2@gmail.com

