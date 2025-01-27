#!/bin/bash

export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"

# Prompt user for 5 integers between 0 and 8
read -p "Enter 5 integers (r1 r2 r3 r4 r5) separated by space: " r1 r2 r3 r4 r5

read -p "Enter the main process port: " main_process_port

read -p "Enter the num of process:" num_process

# Verify the inputs are valid integers between 0 and 8
for r in $r1 $r2 $r3 $r4 $r5; do
    if ! [[ "$r" =~ ^[0-8]$ ]]; then
        echo "Invalid input: $r. Please enter integers between 0 and 8."
        exit 1
    fi
done

# Create the output directory
OUTPUT_DIR="./rl_track_index/${r1}_${r2}_${r3}_${r4}_${r5}"
mkdir -p "$OUTPUT_DIR"

# Initialize a file to log created directories
LOG_FILE="created_directories.txt"
echo "Created directories:" > "$LOG_FILE"

# Function to get TRAIN_DIR based on r value
get_train_dir() {
    local r=$1
    case $r in
        1) echo "./group_1" ;;
        2) echo "./group_1+2" ;;
        3) echo "./group_1+2+3" ;;
        4) echo "./group_3+5+6+8" ;;
        5) echo "./group_1+2+3+4+5" ;;
        6) echo "./group_1+2+3+4+5+6" ;;
        7) echo "./group_1+2+3+4+5+6+7" ;;
        8) echo "./group_1+2+3+4+5+6+7+8" ;;
        *) echo "Invalid value of r" ;;
    esac
}

python_script=$(cat <<EOF
import sys

def max_identical_prefix_length(array1, array2):
    max_length = 0
    for i in range(len(array1)):
        if array1[i] == array2[i]:
            max_length += 1
        else:
            break
    return max_length

def main():
    array1 = list(map(int, sys.argv[1].strip('[]').split(',')))
    array2 = list(map(int, sys.argv[2].strip('[]').split(',')))
    print(max_identical_prefix_length(array1, array2))

if __name__ == "__main__":
    main()
EOF
)

check_existing_folder() {
    local r_sequence="$1"
    local round_to_start=1
    local checkpoint_folder=""
    local last_matched_index=0

    echo "Checking existing folders for r_sequence: $r_sequence" >&2

    # Flag to check if any match was found
    local found_match=false

    # Iterate over directories in rl_track_index
    for existing_dir in ./rl_track_index/*/; do
        existing_dir=${existing_dir%/}  # Remove trailing slash
        dir_name=$(basename "$existing_dir")

        echo "Examining existing directory: $dir_name" >&2

        # Split names into arrays
        IFS="_" read -r -a r_array <<< "$r_sequence"
        IFS="_" read -r -a dir_array <<< "$dir_name"
        
        r_array_str=$(printf '%s,' "${r_array[@]}" | sed 's/,$//')
        dir_array_str=$(printf '%s,' "${dir_array[@]}" | sed 's/,$//')

        echo "r_array: ${r_array[@]}" >&2
        echo "dir_array: ${dir_array[@]}" >&2

        # Run the Python script
        max_length=$(python3 -c "$python_script" "[$r_array_str]" "[$dir_array_str]")

        echo "Match result: matched_index=$max_length" >&2

        if [ "$max_length" -gt 0 ]; then
            last_matched_index=$max_length
            checkpoint_folder="${existing_dir}/checkpoint-$((9500 + (last_matched_index) * 500))"
            round_to_start=$((last_matched_index + 1))

            echo "Using checkpoint folder: $checkpoint_folder" >&2
            echo "Starting from round: $round_to_start" >&2

            if [ -d "$checkpoint_folder" ]; then
                cp -r "$checkpoint_folder" "$OUTPUT_DIR/"
                echo "Checkpoint folder copied to $OUTPUT_DIR/" >&2
            else
                echo "Checkpoint folder $checkpoint_folder does not exist" >&2
                round_to_start=1
            fi
            found_match=true
            break
        fi
    done

    if [ "$found_match" = false ]; then
        echo "No matching folders found." >&2
        round_to_start=1
    fi

    echo "$round_to_start"  # This should be the only output
}



# Generate the r_sequence
r_sequence="${r1}_${r2}_${r3}_${r4}_${r5}"

# Determine the round to start from
start_round=$(check_existing_folder "$r_sequence")
# echo "Debug: start_round value = '$start_round'"  # Single quotes to visualize spaces

# Validate start_round
if ! [[ "$start_round" =~ ^[0-9]+$ ]]; then
    echo "Invalid round number: $start_round"
    exit 1
fi

# Loop over each round starting from the determined round
for round in $(seq "$start_round" 5); do
    r_var="r$round"
    r=${!r_var}
    echo "Debug: start_round value = '$start_round'"
    # Check if the current r is 0, if so, break the loop
    if [ "$r" -eq 0 ]; then
        echo "Round $round: r${round} is 0. Stopping training."
        break
    fi
    
    # Define TRAIN_DIR based on r
    TRAIN_DIR=$(get_train_dir "$r")
    
    # Define CHECKPOINT_PATH
    if [ "$round" -eq 1 ]; then
        CHECKPOINT_PATH="${OUTPUT_DIR}/checkpoint-9500"
        
        # Copy the initial checkpoint if it's the first round
        if [ ! -d "$CHECKPOINT_PATH" ]; then
            mkdir -p "$CHECKPOINT_PATH"
            cp -r ./artbench_expressionism_200/checkpoint-9500/* "$CHECKPOINT_PATH"
        fi
    else
        prev_round=$((round - 1))
        prev_checkpoint="${OUTPUT_DIR}/checkpoint-$((9500 + (prev_round - 1) * 500))"
        CHECKPOINT_PATH="${OUTPUT_DIR}/checkpoint-$((9500 + (round - 1) * 500))"
        
        # Ensure the checkpoint directory exists
        if [ ! -d "$CHECKPOINT_PATH" ]; then
            mkdir -p "$CHECKPOINT_PATH"
            echo "$CHECKPOINT_PATH" >> "$LOG_FILE"
        fi
    fi

    # Calculate num_train_epochs using the formula [(9500 + round * 500) * 2] / (r * 100)
    num_train_epochs=$(( ( (9500 + round * 500) * 2) / (r * 100) ))
    # num_train_epochs=$(( (9500 + round * 500) - ((9500 + (round - 1) * 500) * num_process / (r * 100)) ))


    # Print TRAIN_DIR and OUTPUT_DIR
    echo "Round $round:"
    echo "TRAIN_DIR=$TRAIN_DIR"
    echo "OUTPUT_DIR=$OUTPUT_DIR"
    echo "num_train_epochs=$num_train_epochs"

    # Run the training script
    accelerate launch --main_process_port $main_process_port --num_processes $num_process train_text_to_image_lora_sdxl.py \
      --pretrained_model_name_or_path="$MODEL_NAME" \
      --pretrained_vae_model_name_or_path="$VAE_NAME" \
      --train_data_dir="$TRAIN_DIR" --caption_column="caption" \
      --resolution=256 --random_flip \
      --train_batch_size=1 \
      --num_train_epochs="$num_train_epochs" --checkpointing_steps=100 \
      --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=100 \
      --mixed_precision="fp16" \
      --seed=42 \
      --resume_from_checkpoint="$CHECKPOINT_PATH" \
      --output_dir="$OUTPUT_DIR" \
      --validation_prompt="a close up of a drawing of a man with glasses and a mustache, an ink drawing inspired by Stanisaw Tondos, reddit, shin hanga, man with glasses, portrait of sigmund freud, stanisaw"

    fid=$(python3 inference.py --model_path "$CHECKPOINT_PATH")
    echo "${CHECKPOINT_PATH}: $fid" >> inference_results.txt
done

echo "Training completed. Check $LOG_FILE for created directories."



# Function to check existing folders in rl_track_index and determine if training can be resumed
# check_existing_folder() {
#     local r_sequence="$1"
#     local round_to_start=1

#     for existing_dir in $(ls -d ./rl_track_index/*/); do
#         existing_dir=${existing_dir%/}  # Remove trailing slash
#         dir_name=$(basename $existing_dir)
        
#         # Compare the r_sequence with existing directories
#         IFS="_" read -r -a r_array <<< "$r_sequence"
#         matched_prefix="true"
#         for i in {0..4}; do
#             if [ "${r_array[$i]}" != "${dir_name%%_*}" ]; then
#                 matched_prefix="false"
#                 break
#             fi
#             dir_name=${dir_name#*_}  # Remove the matched part
#         done
        
#         if [ "$matched_prefix" == "true" ]; then
#             round_to_start=$((i + 1))
#             # Ensure the checkpoint for this round exists before proceeding
#             if [ -d "${existing_dir}/checkpoint-$((9500 + (i) * 500))" ]; then
#                 cp -r "${existing_dir}/checkpoint-$((9500 + (i) * 500))" "$OUTPUT_DIR/"
#             else
#                 # If the checkpoint doesn't exist, reset round_to_start to 1
#                 round_to_start=1
#             fi
#             break
#         fi
#     done

#     echo $round_to_start
# }







# # Prompt user for 5 integers between 1 and 8
# read -p "Enter 5 integers (r1 r2 r3 r4 r5) separated by space: " r1 r2 r3 r4 r5

# # Verify the inputs are valid integers between 1 and 8
# for r in $r1 $r2 $r3 $r4 $r5; do
#     if ! [[ "$r" =~ ^[1-8]$ ]]; then
#         echo "Invalid input: $r. Please enter integers between 1 and 8."
#         exit 1
#     fi
# done

# # Create the output directory
# OUTPUT_DIR="./rl_track_index/${r1}_${r2}_${r3}_${r4}_${r5}"
# mkdir -p $OUTPUT_DIR

# # Initialize a file to log created directories
# LOG_FILE="created_directories.txt"
# echo "Created directories:" > $LOG_FILE

# # Function to get TRAIN_DIR based on r value
# get_train_dir() {
#     local r=$1
#     case $r in
#         1) echo "./group_1" ;;
#         2) echo "./group_1+2" ;;
#         3) echo "./group_1+2+3" ;;
#         4) echo "./group_1+2+3+4" ;;
#         5) echo "./group_1+2+3+4+5" ;;
#         6) echo "./group_1+2+3+4+5+6" ;;
#         7) echo "./group_1+2+3+4+5+6+7" ;;
#         8) echo "./group_1+2+3+4+5+6+7+8" ;;
#         *) echo "Invalid value of r" ;;
#     esac
# }

# # Loop over each round
# for round in {1..5}; do
#     r_var="r$round"
#     r=${!r_var}
    
#     # Define TRAIN_DIR based on r
#     TRAIN_DIR=$(get_train_dir $r)
    
#     # Define CHECKPOINT_PATH
#     if [ $round -eq 1 ]; then
#         CHECKPOINT_PATH="${OUTPUT_DIR}/checkpoint-9500"
        
#         # Copy the initial checkpoint if it's the first round
#         mkdir -p $CHECKPOINT_PATH
#         cp -r ./artbench_expressionism_200/checkpoint-9500/* $CHECKPOINT_PATH
#     else
#         prev_round=$((round - 1))
#         prev_checkpoint="${OUTPUT_DIR}/checkpoint-$((9500 + (prev_round - 1) * 500))"
#         CHECKPOINT_PATH="${OUTPUT_DIR}/checkpoint-$((9500 + (round - 1) * 500))"
        
#         # Ensure the checkpoint directory exists
#         if [ ! -d "$CHECKPOINT_PATH" ]; then
#             mkdir -p $CHECKPOINT_PATH
#             echo "$CHECKPOINT_PATH" >> $LOG_FILE
#         fi
#     fi

#     # Calculate num_train_epochs using the formula [(9500 + round * 500) * 2] / r * 100
#     num_train_epochs=$(( ( (9500 + round * 500) * 2) / r * 100 ))

#     # Print TRAIN_DIR and OUTPUT_DIR
#     echo "Round $round:"
#     echo "TRAIN_DIR=$TRAIN_DIR"
#     echo "OUTPUT_DIR=$OUTPUT_DIR"
#     echo "num_train_epochs=$num_train_epochs"

#     # Run the training script
#     accelerate launch --num_processes 2 train_text_to_image_lora_sdxl.py \
#       --pretrained_model_name_or_path=$MODEL_NAME \
#       --pretrained_vae_model_name_or_path=$VAE_NAME \
#       --train_data_dir=$TRAIN_DIR --caption_column="caption" \
#       --resolution=256 --random_flip \
#       --train_batch_size=1 \
#       --num_train_epochs=$num_train_epochs --checkpointing_steps=100 \
#       --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=100 \
#       --mixed_precision="fp16" \
#       --seed=42 \
#       --resume_from_checkpoint=$CHECKPOINT_PATH \
#       --output_dir=$OUTPUT_DIR \
#       --validation_prompt="a close up of a drawing of a man with glasses and a mustache, an ink drawing inspired by Stanisaw Tondos, reddit, shin hanga, man with glasses, portrait of sigmund freud, stanisaw" 
# done

# echo "Training completed. Check $LOG_FILE for created directories."
