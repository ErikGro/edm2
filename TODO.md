# Creating the dataset
python dataset_tool.py convert --source=/graphics/scratch2/students/grosskop/IHCDataset/train/ \
    --dest=datasets/ihc512.zip --resolution=512x512 --transform=random-flip-crop --iterations=8

# Convert the pixel data to VAE latents
python dataset_tool.py encode --source=datasets/ihc512.zip \
    --dest=datasets/ihc512-sd.zip

# Compute dataset reference statistics for calculating metrics
python calculate_metrics.py ref --data=datasets/ihc512.zip \
    --dest=dataset-refs/ihc512.pkl

# Train XS-sized model for ImageNet-512 using 8 GPUs
torchrun --standalone --nproc_per_node=2 train_edm2.py \
    --outdir=training-runs/00000-erik \
    --data=datasets/ihc512-sd.zip \
    --preset=erik \
    --batch-gpu=64 \
    --status=64Ki \
    --snapshot=2Mi \
    --checkpoint=16Mi \
    --cond=False

# Generate images
python generate_images.py \
    --net=training-runs/00000-erik/network-snapshot-0098566-0.100.pkl \
    --outdir=out

# Generate many output images :D
torchrun --standalone --nproc_per_node=4 generate_images.py \
    --net=training-runs/00000-erik/network-snapshot-0098566-0.100.pkl --outdir=out-0098566 --subdirs --seeds=0-8200

