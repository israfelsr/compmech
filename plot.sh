export PYTHONPATH=$(pwd)

python scripts/plot_probes.py \
    --results_dir results/week-33/clip-vit-large-patch14-336 \
    --taxonomy_file dataset/mcrae-x-things-taxonomy-simp.json \
    --save plots/week-33/clip-vit-large-patch14-336