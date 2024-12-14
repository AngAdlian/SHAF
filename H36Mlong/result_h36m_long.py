python main_h36m.py --past_length 10 --future_length 25 --channel 108 --model_name ckpt_long_12 --test


{'Total': 2557168, 'Trainable': 2557168}
Loading model from: h36m/saved_models/ckpt_long_12.pth.tar
Act: walking        |,  ErrT: 9.629 18.489 33.081 38.632 45.960 53.233, TestError 33.1707
Act: eating         |,  ErrT: 6.652 13.960 29.114 35.989 47.914 72.165, TestError 34.2990
Act: smoking        |,  ErrT: 6.138 12.295 24.426 30.076 39.992 61.858, TestError 29.1309
Act: discussion     |,  ErrT: 9.053 20.780 45.878 56.486 72.748 101.122, TestError 51.0112
Act: directions     |,  ErrT: 6.863 16.183 39.255 50.228 68.944 99.945, TestError 46.9031
Act: greeting       |,  ErrT: 13.086 30.320 67.709 83.763 107.332 140.415, TestError 73.7707
Act: phoning        |,  ErrT: 7.766 16.951 37.138 46.721 64.123 100.458, TestError 45.5263
Act: posing         |,  ErrT: 8.904 20.806 47.293 59.342 80.366 131.014, TestError 57.9543
Act: purchases      |,  ErrT: 11.540 26.368 58.219 72.324 95.458 138.290, TestError 67.0332
Act: sitting        |,  ErrT: 8.295 18.188 41.095 52.628 73.892 116.559, TestError 51.7761
Act: sittingdown    |,  ErrT: 13.063 26.370 55.520 69.696 95.705 149.242, TestError 68.2660
Act: takingphoto    |,  ErrT: 8.170 17.897 40.855 52.362 73.713 117.716, TestError 51.7857
Act: waiting        |,  ErrT: 8.117 17.974 40.323 50.872 69.066 102.582, TestError 48.1556
Act: walkingdog     |,  ErrT: 17.092 36.043 69.704 82.139 102.267 137.894, TestError 74.1899
Act: walkingtogether |,  ErrT: 8.251 16.681 31.127 36.963 45.770 58.247, TestError 32.8399
avg mpjpe: [  9.50797726  20.62031579  44.04911339  54.54806588  72.21674642
 105.38278928]
