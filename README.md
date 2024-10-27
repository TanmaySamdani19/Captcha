# Captcha

To download dataset from server --> python file_download.py


To generate captchas  -->  python generate.py

To train best length model  -->  python length_model.py --resume best_captcha_model.pth

TO train captcha character identifier model  -->  python train.py --width 198 --height 96 --length 6 --batch_size 32 --train_data ./train6 --valid_data ./validate6 --model_name captcha_model6 --symbols symbols.txt --epochs 30 --resume captcha_model6_best.pth

To Convert Models  -->  python convert_models.py --output_dir ./ --num_classes 42

To run on Local Machine  -->  python captcha_classify.py   --length_model best_captcha_model.pth     --symbols ./symbols.txt     --model_dir ./    --dir dataset-marisha    --output result.csv    --log-level INFO

To run on TFLite  --> python3 tflite_classify.py --symbols ./symbols.txt --model_dir ./ --dir ./samdani-tanmay --output results.csv
