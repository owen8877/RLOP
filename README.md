# Misc
- tensor board: `tensorboard --logdir=playground/runs`


# How to run experiments

- Install the environment by `conda env create -n [NAME] --file env.yml`
- For QLBS: run by unittest in the module `qlbs`
    - `experiment1.Experiment1.test_mutation_or_not`, followed by `experiment1.Experiment1.test_draw_plot` (Fig. 3a &
      3b)
    - `experiment2.Experiment2.test_bs_value`, followed by `experiment2.Experiment2.test_draw_plot` (Fig. 4)
    - `experiment3.Experiment3.test_tc`, followed by `experiment3.Experiment3.test_draw_plot` (Fig. 5)
    - `experiment4.Experiment4.test_mixed` and `experiment4.Experiment4.test_focused`, followed
      by `experiment4.Experiment4.test_draw_plot` (Fig. 6)
- For RLOP: run by unittest in the module `rlop`
    - `experiment1.Experiment1.test_mutation_or_not`, followed by `experiment1.Experiment1.test_draw_plot`
      and `experiment1.Experiment1.test_compared_with_bs` (Fig, 8a & 8b)
    - `experiment2.Experiment2.test_tc`, followed by `experiment2.Experiment2.test_draw_plot` (Fig. 9)