
python tracking/prop_preprocess.py --input_dir /storage/user/liuyang/TAO_eval/TAO_VAL_Proposals/Panoptic_Cas_R101_NMSoff_forTracking_Embed/npz/ \
                          --outdir /storage/user/liuyang/TAO_eval/TAO_VAL_Proposals/Panoptic_Cas_R101_NMSoff_forTracking_Embed/preprocessed/ \
                          --image_dir /storage/slurm/liuyang/data/TAO/TAO_VAL/val/ \
                          --track_result_format unovost \
                          --datasrc HACS

python tracking/visualization.py --tracks_folder /storage/slurm/liuyang/Tracking/SORT_results/ \
      --img_folder /storage/slurm/liuyang/data/TAO/TAO_VAL/val/ \
      --gt_path /storage/slurm/liuyang/data/TAO/TAO_annotations/validation.json \
      --track_format mot \
      --phase objectness \
      --topN_proposals 10000 \
      --only_annotated \
      --datasrc ArgoVerse



python tracking/visualization.py --tracks_folder /storage/slurm/liuyang/Tracking/unovost_RAFT_noReID002/ \
      --img_folder /storage/slurm/liuyang/data/TAO/TAO_VAL/val/ \
      --gt_path /storage/slurm/liuyang/data/TAO/TAO_annotations/validation.json \
      --track_format mot \
      --phase objectness \
      --topN_proposals 10000 \
      --datasrc Charades