# Lean Reinforcement Learning

This is the repo for Learn Reinforcement Learning.

This codebase contains the general RL code, the free Needle Master environment,
and code to interface with the closed-source dVSS-RL environment.

To use this code, you must use python 3. Use the `requirements.txt` file, preferably in a virtual environment:

```bash
pip3 install -r requirements.txt
```
You should now have all the needed packages installed.

To run the Needle Master environment with lean RL, use the following command:

```bash
python3 -m rl.main --env needle --procs 8 --mode state --policy ddqn --record --add-delay 0.5
```

The `--procs` argument chooses how many parallel environments you want to run.
In general this should be <= to the number of cores in your processor.

`--mode` can be `state`, `image` or `mixed`, where mixed is a combination of image
and minimal state data. In general, you'll want to record in `state` mode,
and play back with one of the others.

`--record` specifies that all environment transitions should be recorded to permanent
storage. You'll find the videos and state residing in `./saved_data`.

`--add-delay` is optional. In order to simulate a slow environment, some artificial delay is required for
Needle Master, which is very simple and fast naturally.

To train the same environment with some playback, use the command

```bash
python3 -m rl.main --env needle --procs 8 --mode state --policy ddqn --playback 2 --play-rate 80 --add-delay 0.5
```

`--play-rate` tries to keep the rate of playback-to-real data around the percent requested.

`--playback` determines how many environments (out of the maximum of `procs`)
are used to play back old data. You must have some old data collected already in order for this to work.

To see details about more options, use the `python3 -m rl.main --help` command.
