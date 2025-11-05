# Getting started
```
uv sync
```
then you should be able to find a test under lib2/test_hello_world.py (in vscode it is "Explorer: Focus on Test Explorer View" command)


# Run summarized test
- setup env `conda install -c anaconda matplotlib numpy pandas jupyter gymnasium scipy tensorboard seaborn yfinance pandas-datareader -y`
- place `SPY Options 2025.csv` under `data/`
- run `python -m unittest lib/qlbs2/test_summarized.py` at project root directory

# Misc
- tensor board: `tensorboard --logdir=runs`
