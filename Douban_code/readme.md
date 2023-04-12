To run the code,

for learning stage:
* python douban_main_all.py --model_name pdfm_user_autodis  --learning_rate 2e-3 --job 1

for tuning stage
* python douban_main_music.py --model_name pdfm_user_autodis --learning_rate 5e-3 --freeze 5
* python douban_main_book.py --model_name pdfm_user_autodis --learning_rate 5e-3 --freeze 5
* python douban_main_movie.py --model_name pdfm_user_autodis --learning_rate 5e-4 --freeze 5

model choices=[pdfm_user_autodis, pdfm_usermlp], which corresponds to Fusion and Generative user prompt.
