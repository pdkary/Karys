rm ./examples/progressive_gan/test_output/gan_board_logs/*
rm ./examples/progressive_gan/test_output/*.jpg
rm ./examples/progressive_gan/test_output/disc_gen/*.jpg
rm ./examples/progressive_gan/test_output/disc_real/*.jpg
rm ./examples/progressive_gan/test_output/generated/*.jpg
rm ./examples/progressive_gan/test_output/architecture_diagrams/*.png
rm ./**/__pycache__/*
python runner.py & tensorboard --logdir ./examples/progressive_gan/test_output/gan_board_logs/