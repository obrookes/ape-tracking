
## Requirements

```bash
...
```

## Usage

A quick start script is provided below:

```bash
python track.py --results_file=results.pkl # output of torch model
                --video=videos # path to videos corresponding to output
                --confidence=0.5 # remove predictions lower than this
                --distance_matrix=euclidean # type of distance matrix
                --length=20 # remove tracklets smaller than this
                --write=True # write to video
```


