#!/bin/bash

# Array of noise types from main.py lines 66-87
noise_types=(
    "mixup_obs"
    "dropout_obs"
    "normal_obs"
    "uniform_obs"
    "scale_obs"
    "scale_reward"
    "noisy_reward"
    "normal_reward"
    "none"
)

# Array of noise rates to test
noise_rates=(0.1 0.3 0.5)

# Counter for tracking runs
run_count=0

# Loop through each noise type
for noise_type in "${noise_types[@]}"; do
    # Loop through each noise rate
    for noise_rate in "${noise_rates[@]}"; do
        run_count=$((run_count + 1))
        echo "Run $run_count/27: noise_type=$noise_type, noise_rate=$noise_rate"
        
        # Run main.py with the current noise type and noise rate
        python main.py --mode train_lls --noise_type "$noise_type" --noise_rate "$noise_rate"
        
        # Check if the run was successful
        if [ $? -ne 0 ]; then
            echo "Error: Run $run_count failed with noise_type=$noise_type, noise_rate=$noise_rate"
            exit 1
        fi
    done
done

echo "All 27 runs completed successfully!"

