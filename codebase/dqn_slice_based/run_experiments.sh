#!/bin/bash
# Run experiments with different coefficient combinations
# Coefficients: cont, grad, manhattan
# Values: 0, 0.1, 0.2, 0.5

# Create logs directory if it doesn't exist
mkdir -p logs

# Define coefficient values
COEFF_VALUES=(0 0.1 0.2 0.5)

# Loop through all combinations
for cont in "${COEFF_VALUES[@]}"; do
    for grad in "${COEFF_VALUES[@]}"; do
        for manhattan in "${COEFF_VALUES[@]}"; do
            LOG_FILE="logs/rapid-p-32-grid-cont-${cont}-grad-${grad}-manh-${manhattan}.log"
            
            echo "Starting experiment: cont=${cont}, grad=${grad}, manhattan=${manhattan}"
            echo "Log file: ${LOG_FILE}"
            
            nohup python main.py \
                --epochs 20 \
                --cont ${cont} \
                --grad ${grad} \
                --manhattan ${manhattan} \
                > "${LOG_FILE}" 2>&1 &
            
            # Wait for this job to finish before starting the next
            # Remove the 'wait' line if you want to run jobs in parallel
            wait
            
            echo "Finished: cont=${cont}, grad=${grad}, manhattan=${manhattan}"
            echo "----------------------------------------"
        done
    done
done

echo "All experiments completed!"
