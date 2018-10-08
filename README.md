# pytorch-vsumm-reinforce

Fork of [pytorch-vsumm-reinforce](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce)

## Additions

1. `features.py`

    Feature extraction example code that I used. It is necessary to modify it to your 
    needs if you want to use it.
    
    *Note:* Original segmentation is from here - [KTS](https://github.com/pathak22/videoseg/tree/7d88be917d64ff721e0fa61ad106c8b3fa63c994/lib/kts)

2. `create_folds.py`
    
    Original `create_splits.py` does not actually create proper folds, but just splits the data 
    randomly `n` times. Which is not the correct way to do K-Fold analysis. 
    
    Use the same as you would use `create_splits.py`:
    
    ```angular2html
    python create_folds.py -d dataset.h5 --save-dir datasets --save-name dataset_folds  --num-splits 5
    ```

3. `parse_log.py`

    Add additional functionality to `parse_log.py`, such as:
    * Ability to specify range in what you want to plot rewards:
        * `-se, --start-epoch, default=0`
        * `-ee, --end-epoch, default=None`
    * Ability to choose filename to save to:
        * `-f, --filename, default="overall_reward.png"`
    

## Please cite the Author
```
@article{zhou2017reinforcevsumm, 
   title={Deep Reinforcement Learning for Unsupervised Video Summarization with Diversity-Representativeness Reward},
   author={Zhou, Kaiyang and Qiao, Yu and Xiang, Tao}, 
   journal={arXiv:1801.00054}, 
   year={2017} 
}
```