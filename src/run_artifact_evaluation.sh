#!/bin/bash

# Start a new tmux session named 'my_session' and split the screen vertically
tmux new-session -d -s artifact_eval

# Split the window into left and right panes
tmux split-window -h

# Select the left pane (pane 0) and run the first command
tmux select-pane -t 0
tmux send-keys "bash run_physical.sh | tee -a log_physical.txt" C-m

# Select the right pane (pane 1) and run the second command
tmux select-pane -t 1
tmux send-keys "sleep 3 && python job_submission_driver.py" C-m
# tmux send-keys "sleep 5 && echo HI" C-m

# Attach to the session
tmux attach -t artifact_eval 
