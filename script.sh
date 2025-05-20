export MODEL_NAME=chemical_system
export RESULTS_PATH="results/$MODEL_NAME/"  # Samples will be written to this directory
export END_PATH="/Data/auguste.de-lambilly/mattergenbis/results/Vladimir"  # Directory containing the zip files

# Number of samples to generate
NB=$1

#Systems that will be generated
systems=(Cu-Si-K Cu-Si-P Mo-Si-K Mo-Si-P Mo-Si-B Mo-Si-B-P Si-B-P Mo-B-P
         Cu-Si Si-K Cu-K Cu-P Si-P Mo-Si Mo-K Mo-P Mo-B Si-B) 
         #Mo Cu Si K P B)
mkdir -p /Data/auguste.de-lambilly/mattergenbis/$RESULTS_PATH
mkdir -p $END_PATH

for system in "${systems[@]}"; do
    echo "Generating $NB samples for $system"
    start_time=$(date +%s)
    # Generate samples for each system
    mkdir -p /Data/auguste.de-lambilly/mattergenbis/$RESULTS_PATH$system
    mattergen-generate "$RESULTS_PATH$system" --pretrained-name=$MODEL_NAME --batch_size=$NB --properties_to_condition_on="{'chemical_system': '$system'}" --diffusion_guidance_factor=2.0 > /Data/auguste.de-lambilly/mattergenbis/$RESULTS_PATH$system/log.txt 2>&1
    # Copy the generated samples to the end path
    cp -r "/Data/auguste.de-lambilly/mattergenbis/$RESULTS_PATH$system/generated_crystals_cif.zip" "$END_PATH/$system.zip"
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    echo "Duration: ${duration} seconds"
done

# Zip the generated samples
cd /Data/auguste.de-lambilly/mattergenbis/results/
zip -r "Vladimir.zip" "Vladimir"
echo "Zipped the generated samples to $END_PATH.zip"
