# python scripts/compare_probes.py \
#     --results_dir1 results/week-33/clip-vit-large-patch14-336 \
#     --results_dir2 results/week-33/llava1.5-clip336 \
#     --labels "CLS" "AVG Pool" \
#     --layers proj \
#     --metric f1

python scripts/compare_probes.py \
    --results_dir1 results/week-33/clip-vit-large-patch14-336 \
    --results_dir2 results/week-33/llava1.5-clip336 \
    --labels "CLS" "AVG Pool" \
    --layers proj \
    --flipping-analysis \
    --detailed-layer proj \
    --taxonomy dataset/mcrae-x-things-taxonomy-simp.json
    --metric f1
