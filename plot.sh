export PYTHONPATH=$(pwd)

python scripts/plot_probes.py \
    --results_dir results \
    --taxonomy_file dataset/mcrae-x-things-taxonomy.json \
    --save plots