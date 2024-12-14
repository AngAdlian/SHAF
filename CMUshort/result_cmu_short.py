python main_cmu.py --past_length 10 --future_length 10 --channel 96 --model_name ckpt_65 --test

Act: basketball     |,  ErrT: 9.254 17.071 35.445 45.349, TestError 26.7797
Act: basketball_signal |,  ErrT: 2.395 4.402 10.016 13.376, TestError 7.5474
Act: directing_traffic |,  ErrT: 4.623 9.332 22.355 30.079, TestError 16.5972
Act: jumping        |,  ErrT: 11.806 24.444 53.051 67.806, TestError 39.2770
Act: running        |,  ErrT: 8.965 14.371 25.957 29.464, TestError 19.6892
Act: soccer         |,  ErrT: 9.063 16.238 32.161 41.171, TestError 24.6581
Act: walking        |,  ErrT: 5.789 9.260 15.452 18.739, TestError 12.3102
Act: washwindow     |,  ErrT: 4.308 8.686 20.655 27.565, TestError 15.3034
avg mpjpe: [ 7.02549193 12.97538217 26.88664183 34.19352158]
