export PYTHONPATH=$(pwd)

python scripts/plot_probes.py \
    --results_dir results/week-33/llava1.5-clip336 \
    --taxonomy_file dataset/mcrae-x-things-taxonomy-simp.json \
    --save plots/week-33/llava1.5-clip336