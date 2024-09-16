# Reproducing the dataset
Also has some dependency hell troubleshooting tips at the bottom.

Preparing the dataset was a hacky process, so things are a bit more roundabout than they need to be, but the following instructions (not tested) should reproduce the dataset:

First, we need to filter our dataset down to the subset that we actually use for training (there doesn't seem to be a way to filter out rows after the fact). Run `duplic_dataset.ipynb` which will:
- Remove any utterances corresponding to multiple speakers
- Normalize the audio (I suspect this might be beneficial for the neural audio codec. For some reason this uses a large amount of memory, so I couldn't use multiprocessing on my machine).

**Note** that you need to change the hardcoded repo name at the end of the notebook so you upload to your own huggingface account. Then, run the stage 1 dataset script (change the corresponding dataset) which will estimate things like pitch and SDR (this took me ~2hrs on a 3080Ti for 48k hrs of audio):
```
python main.py "therealvul/parlertts-pony-speech-audio" --configuration "default"  --output_dir ./tmp_pony_speech/  --text_column_name "transcription"  --audio_column_name "audio"  --cpu_num_workers 8  --rename_column  --apply_squim_quality_estimation
```

This will write to an output directory called `tmp_pony_speech`.

The next step of processing requires a `speaker_id` column. Run `speaker_id_gender_fixer.ipynb` which will add this column based on all the speakers in the dataset. It also has a cell for removing rows corresponding to multiple speakers--redundant here, but was used the first time I made the dataset. Then change the repo at the bottom and push to your HF (the next few scripts require datasets hosted on huggingface).

Then we can run the stage 2 dataset script (change the input/output repo names accordingly) which will convert the extracted statistics into phrases like "noisy" or "very clean" or "expressive" based on text bins; I used the text bins that came with the dataspeech repo.

```
python ./scripts/metadata_to_text.py "therealvul/parlertts_pony_speech_ids_fixed_stage1" --repo_id "parlertts_pony_speech_tags_stage2" --configuration "default" --output_dir "./tmp_pony_speech_tagged" --cpu_num_workers "8" --leading_split_for_bins "train" --plot_directory "./plots/" --path_to_text_bins "./examples/tags_to_annotations/v02_text_bins.json" --apply_squim_quality_estimation
```

The stage 3 dataset script is by far the most compute intensive and requires a GPU capable of running a decent text model in 4-bit mode (Mistral 7B-0.3), as it uses a language model to generate synthetic descriptions for the audio based on the stage 2 phrases. You may have to modify the batch size for your particular setup.

```
python ./scripts/run_prompt_creation.py  --speaker_ids_to_name_json ./speaker_ids_to_names.json --speaker_id_column "speaker_id" --speaker_ids_to_name_json "./speaker_ids_to_names.json" --is_new_speaker_prompt --dataset_name "therealvul/parlertts_pony_speech_tags_stage2" --dataset_config_name "default"  --model_name_or_path "mistralai/MIstral-7B-Instruct-v0.3"  --per_device_eval_batch_size 8 --attn_implementation "sdpa"  --output_dir "./tmp_pony_speech_tagged"  --load_in_4bit  --push_to_hub  --hub_dataset_id "parlertts_pony_speech_tagged_stage3"  --preprocessing_num_workers 8 --dataloader_num_workers 8
```

Finally: I noticed that the prompt for named speakers in dataspeech doesn't include information for gender, which means the language model sometimes mis-guesses the gender of the speaker. You can either try modifying the prompt yourself (which I didn't do) or just do a dumb replacement of his/her in the descriptions accordingly (which I did do, under `gender_fixer_stage3.ipynb`).

Aside: In the middle of training my first model, new rows were added to the pony-speech dataset which required me to re-process the entire dataset. To save on computations, I used my existing natural language descriptions for the existing rows and just generated new descriptions (inefficiently) for the new rows, the code for which you'll find in `fill_missing_descriptions.ipynb`.

# Troubleshooting

## pip._vendor.packaging._tokenizer.ParserSyntaxError: .* suffix can only be used with `==` or `!=` operators
`python.exe -m pip install -U pip==23.2.1`

## ImportError: cannot import name '_compare_version' from 'torchmetrics.utilities.imports'
Try downgrading torchmetrics to `torchmetrics==0.11.4`
    
## ImportError: cannot import name 'get_ref_type' from 'omegaconf._utils'
`pip install -U == omegaconf==2.2.0`

## ModuleNotFoundError: No module named 'lightning_fabric'
https://github.com/pyannote/pyannote-audio/issues/1400

In file: `/opt/conda/lib/python3.10/site-packages/pyannote/audio/core/model.py`
 
Change:

`from lightning_fabric.utilities.cloud_io import _load as pl_load`
 
to
 
`from lightning.fabric.utilities.cloud_io import _load as pl_load`