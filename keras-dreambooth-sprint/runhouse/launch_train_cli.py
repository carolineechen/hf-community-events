import runhouse as rh


# Based on https://github.com/sayakpaul/dreambooth-keras/blob/main/train_dreambooth.py
if __name__ == "__main__":
    # spin up lambda cluster, using SkyPilot handling
    # use sky check to ensure provider credentials are set up correctly
    gpu = rh.cluster(name='rh-cluster', instance_type='A100:1', provider='lambda')
    gpu.up_if_not()

    # byo cluster using ssh credentials
    # gpu = rh.cluster(name='rh-cluster', ssh_creds={'user':<user>, 'ssh_creds':{'ssh_private_key':<path_to_id_rsa>}})

    command = "conda install -y -c conda-forge cudatoolkit=11.2.2 cudnn=8.1.0; \
               mkdir -p $CONDA_PREFIX/etc/conda/activate.d; \
               echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh; \
               export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda/; \
               python3 -m pip install tensorflow"
    gpu.run([command])
    
    
    # pulled from https://github.com/huggingface/community-events/blob/main/keras-dreambooth-sprint/requirements.txt
    gpu.install_packages([
            rh.GitPackage(git_url='https://github.com/sayakpaul/dreambooth-keras.git', install_method='local'),
            'keras_cv==0.4.0',
            'tensorflow>=2.10.0',
            'tensorflow_datasets>=4.8.1',
            'pillow==9.4.0',
            'imutils',
            'opencv-python',
            'wandb',
    ])

    # check if TF GPU support is set up properly on the cluster
    gpu.run(['python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices(\'GPU\'))"'])
    gpu.run(['cp /usr/lib/cuda/nvvm/libdevice/libdevice.10.bc .'])

    instance_images_url = 'https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/instance-images.tar.gz'
    class_images_url = 'https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/class-images.tar.gz'

    # launch training -- you can modify or add any additional args
    gpu.run([f'python dreambooth-keras/train_dreambooth.py --mp '
             f'--instance_images_url={instance_images_url} '
             f'--class_images_url={class_images_url}'])
