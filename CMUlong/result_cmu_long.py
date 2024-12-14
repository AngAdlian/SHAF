>python main_cmu.py --past_length 10 --future_length 25 --channel 108 --model_name ckpt_long_66 --test


Loading model from: cmu/saved_models/ckpt_long_66.pth.tar
Act: basketball     |,  ErrT: 9.599 17.802 35.541 44.175 59.114 82.394, TestError 41.4374
Act: basketball_signal |,  ErrT: 2.622 4.557 10.588 14.412 22.778 47.303, TestError 17.0435
Act: directing_traffic |,  ErrT: 4.913 10.061 24.160 32.565 50.484 100.928, TestError 37.1851
Act: jumping        |,  ErrT: 12.443 25.591 53.989 67.194 89.106 125.035, TestError 62.2265
Act: running        |,  ErrT: 10.281 16.177 27.070 31.068 34.758 45.946, TestError 27.5503
Act: soccer         |,  ErrT: 9.922 17.655 34.551 43.450 59.401 92.682, TestError 42.9435
Act: walking        |,  ErrT: 5.883 9.414 15.501 18.067 22.106 31.310, TestError 17.0469
Act: washwindow     |,  ErrT: 4.582 9.351 22.662 30.056 44.058 73.226, TestError 30.6556
avg mpjpe: [ 7.53079379 13.82588848 28.00774557 35.12349814 47.72571375 74.8530104 ]
