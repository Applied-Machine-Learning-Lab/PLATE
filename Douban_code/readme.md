To run the code,

for learning stage:
* python douban_main_all.py --model_name pdfm_fusion  --learning_rate 2e-3 --job 1

for tuning stage
* python douban_main_music.py --model_name pdfm_fusion --learning_rate 5e-3 --freeze 5
* dataset_name=[douban_music,douban_book,douban_movie].

model choices=[pdfm_user_autodis, pdfm_usermlp], which corresponds to Fusion and Generative user prompt. Meanwhile, the learning rate differs on different domains.
