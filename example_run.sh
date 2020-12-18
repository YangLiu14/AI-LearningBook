
python tracking/prop_preprocess.py --input_dir /storage/slurm/liuyang/TAO_eval/TAO_VAL_Proposals/Panoptic_Cas_R101_NMSoff_forTracking/json/ \
                          --outdir /storage/slurm/liuyang/TAO_eval/TAO_VAL_Proposals/Panoptic_Cas_R101_NMSoff_forTracking/preprocessed/ \
                          --image_dir /storage/slurm/liuyang/data/TAO/TAO_VAL/val/ \
                          --track_result_format unovost \
                          --datasrc ArgoVerse

python tracking/visualization.py --tracks_folder /storage/slurm/liuyang/Tracking/SORT_results/ \
      --img_folder /storage/slurm/liuyang/data/TAO_VAL/val/ \
      --track_format mot \
      --phase objectness \
      --topN_proposals 1000 \
      --datasrc ArgoVerse