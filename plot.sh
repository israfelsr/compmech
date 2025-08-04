export PYTHONPATH=$(pwd)

python scripts/plot_probes.py \
    --results_dir results/08-04 \
    --taxonomy_file dataset/mcrae-x-things-taxonomy.json \
    --save plots/08-04