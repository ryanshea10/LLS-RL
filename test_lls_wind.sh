# Run with increasing wind and turbulence powers
# wind_power: 5, 10, 15
# turbulence_power: 0.5, 1, 1.5
wind_powers=(0 5 10 15)
turbulence_powers=(0 0.5 1 1.5)
run_count=0

echo "=========================================="
echo "Testing with increasing wind and turbulence powers"
echo "=========================================="
echo ""

for i in {0..3}; do
    wind_power=${wind_powers[$i]}
    turbulence_power=${turbulence_powers[$i]}
    
    echo "=========================================="
    echo "Testing with wind_power=$wind_power, turbulence_power=$turbulence_power"
    echo "=========================================="
    echo ""
    
    # Run 10 times for each combination
    for run in {1..10}; do
        run_count=$((run_count + 1))
        echo "--- Run $run of 10 for this combination (Total run: $run_count) ---"
        echo "Executing: python main.py --mode train_lls --enable_wind --wind_power $wind_power --turbulence_power $turbulence_power"
        echo ""
        
        python main.py --mode train_lls --enable_wind --wind_power "$wind_power" --turbulence_power "$turbulence_power"
        
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
    echo "Completed 10 runs for wind_power=$wind_power, turbulence_power=$turbulence_power"
    echo "=========================================="
    echo ""
done

echo "All wind testing runs completed!"
echo "Total runs executed: $run_count"

