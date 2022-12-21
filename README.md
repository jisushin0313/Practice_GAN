# Practice_GAN

Excecution command
~~~
# for training
python model/main.py --do_train=True

# for evaluation
python model/main.py --do_eval=True
~~~

Arguments
~~~
# --arg = default_value
--seed = 42
--gpu_id = '0'
--output_dir = './checkpoint' # save or load checkpoint file on
                                'model/checkpoint/{model_name}/rand_{seed}_b_{batch_size}_netG_best.pth' and
                                'model/checkpoint/{model_name}/rand_{seed}_b_{batch_size}_netD_best.pth
--data_path = './../dataset'

--model = 'dcgan'

--nz = 256
--nc = 3
--ngf = 64
--ndf = 64

--epochs = 50
--lr = 2-4
--weight_decay = 1e-5
--beta1 = 0.5
--batch_size = 128

--do_train = False
--do_eval = False
~~~
