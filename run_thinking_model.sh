#!/bin/bash
#SBATCH -p scavenger
#SBATCH -A scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:l40s:4
#SBATCH --time=1-12:00:00
#SBATCH --mem=128GB
#SBATCH -J gencap
#SBATCH --output=/fs/nexus-scratch/zsodagar/MILS/logs/gencap-%j.out
#SBATCH --open-mode=append

set -euo pipefail

# ----------------------------
# Env / setup
# ----------------------------
git clone https://github.com/facebookresearch/MILS.git
cd MILS

set +u
source /fs/nexus-scratch/zsodagar/anaconda3/etc/profile.d/conda.sh
conda env create -f environment.yml
conda activate MILS
set -u
module load cuda/12.6.3 gcc/11.2.0


wget http://images.cocodataset.org/zips/val2014.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip 

unzip val2014.zip
unzip annotations_trainval2014.zip


# ----------------------------
# Config
# ----------------------------
TEXT_MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
OUTPUT_DIR="/fs/nexus-scratch/zsodagar/MILS/outputs/reasoning"
PROMPT="/fs/nexus-scratch/zsodagar/MILS/prompts/image_captioning_shorter_reasoning.txt"

# ----------------------------
# Run (4 GPUs)
# ----------------------------
CUDA_VISIBLE_DEVICES=0 python main_image_captioning.py --process 0 --num_processes 4 --batch_size 32 --ablation --text_model "$TEXT_MODEL" --output_dir "$OUTPUT_DIR" --prompt "$PROMPT" &
CUDA_VISIBLE_DEVICES=1 python main_image_captioning.py --process 1 --num_processes 4 --batch_size 32 --ablation --text_model "$TEXT_MODEL" --output_dir "$OUTPUT_DIR" --prompt "$PROMPT" &
CUDA_VISIBLE_DEVICES=2 python main_image_captioning.py --process 2 --num_processes 4 --batch_size 32 --ablation --text_model "$TEXT_MODEL" --output_dir "$OUTPUT_DIR" --prompt "$PROMPT" &
CUDA_VISIBLE_DEVICES=3 python main_image_captioning.py --process 3 --num_processes 4 --batch_size 32 --ablation --text_model "$TEXT_MODEL" --output_dir "$OUTPUT_DIR" --prompt "$PROMPT" &

wait
