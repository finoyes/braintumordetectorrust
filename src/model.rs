use burn::prelude::*;
use burn::nn::{
    conv::{Conv2d, Conv2dConfig},
    pool::{MaxPool2d, MaxPool2dConfig},
    Linear, LinearConfig, Relu,
    BatchNorm, BatchNormConfig,
};

use crate::data::{IMG_SIZE, NUM_CHANNELS};


#[derive(Module, Debug)]
pub struct BrainTumorCNN<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B, 2>,
    conv2: Conv2d<B>,
    bn2: BatchNorm<B, 2>,
    conv3: Conv2d<B>,
    bn3: BatchNorm<B, 2>,
    pool: MaxPool2d,
    fc1: Linear<B>,
    fc2: Linear<B>,
    relu: Relu,
}

impl<B: Backend> BrainTumorCNN<B> {
    pub fn new(device: &B::Device) -> Self {
        
        let flat_size = 128 * (IMG_SIZE / 8) * (IMG_SIZE / 8); // 128 * 16 * 16 = 32768

        Self {
            conv1: Conv2dConfig::new([NUM_CHANNELS, 32], [3, 3])
                .with_padding(burn::nn::PaddingConfig2d::Same)
                .init(device),
            bn1: BatchNormConfig::new(32).init(device),
            conv2: Conv2dConfig::new([32, 64], [3, 3])
                .with_padding(burn::nn::PaddingConfig2d::Same)
                .init(device),
            bn2: BatchNormConfig::new(64).init(device),
            conv3: Conv2dConfig::new([64, 128], [3, 3])
                .with_padding(burn::nn::PaddingConfig2d::Same)
                .init(device),
            bn3: BatchNormConfig::new(128).init(device),
            pool: MaxPool2dConfig::new([2, 2])
                .with_strides([2, 2])
                .init(),
            fc1: LinearConfig::new(flat_size, 256).init(device),
            fc2: LinearConfig::new(256, 2).init(device),
            relu: Relu::new(),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 2> {
        let batch_size = x.dims()[0];

    
        let x = self.conv1.forward(x);
        let x = self.bn1.forward(x);
        let x = self.relu.forward(x);
        let x = self.pool.forward(x);

    
        let x = self.conv2.forward(x);
        let x = self.bn2.forward(x);
        let x = self.relu.forward(x);
        let x = self.pool.forward(x);

        let x = self.conv3.forward(x);
        let x = self.bn3.forward(x);
        let x = self.relu.forward(x);
        let x = self.pool.forward(x);

     
        let flat_size = 128 * (IMG_SIZE / 8) * (IMG_SIZE / 8); // 128 * 16 * 16 = 32768
        let x = x.reshape([batch_size, flat_size]);
        let x = self.fc1.forward(x);
        let x = self.relu.forward(x);
        self.fc2.forward(x)
    }
}
