#!/bin/bash

# Script to run main.py with each different training mode twice
# Training modes are based on lls_layers.py

# Define training modes
# Format: actor_mode:critic_mode
declare -a training_modes=(
    "PPO_LLS:LLS"
    "PPO_LLS_M:LLS_M"
    "PPO_LLS_Random:LLS_Random"
    "PPO_LLS_M_Random:LLS_M_Random"
    "PPO_LLS_MxM_Random:LLS_MxM_Random"
    "PPO_Classifier:Classifier"
)

# "PPO_LLS_MxM:LLS_MxM"

# Counter for tracking runs
run_count=0

# Loop through each training mode
for mode_pair in "${training_modes[@]}"; do
    # Split the mode pair into actor and critic modes
    IFS=':' read -r actor_mode critic_mode <<< "$mode_pair"
    
    echo "=========================================="
    echo "Training Mode: Actor=$actor_mode, Critic=$critic_mode"
    echo "=========================================="
    
    # Run each mode twice
    for run in {1..2}; do
        run_count=$((run_count + 1))
        echo ""
        echo "--- Run $run of 2 for this mode (Total run: $run_count) ---"
        echo "Executing: python main.py --mode train_lls --actor_training_mode $actor_mode --critic_training_mode $critic_mode"
        echo ""
        
        # Run the training
        python main.py --mode train_lls --actor_training_mode "$actor_mode" --critic_training_mode "$critic_mode"
        
        # Check if the command succeeded
        if [ $? -eq 0 ]; then
            echo "✓ Run $run completed successfully"
        else
            echo "✗ Run $run failed with exit code $?"
        fi
        
        echo ""
        echo "Waiting 2 seconds before next run..."
        sleep 2
    done
    
    echo ""
    echo "Completed 2 runs for $actor_mode / $critic_mode"
    echo "=========================================="
    echo ""
done

echo "All training runs completed!"
echo "Total runs executed: $run_count"

