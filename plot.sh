export PYTHONPATH=$(pwd)

python scripts/plot_probes.py \
    --results_dir results/week-34 \
    --taxonomy_file dataset/mcrae-x-things-taxonomy-simp.json \
    --save plots/week-34/llava-vlm