import sys
import os
sys.path.append("..")

from gan import output
sys.modules["output"] = output

import numpy as np

from gan.doppelganger import DoppelGANger
from gan.load_data import load_data
from gan.network import DoppelGANgerGenerator, Discriminator, AttrDiscriminator
from gan.output import Output, OutputType, Normalization
import tensorflow as tf
from gan.network import DoppelGANgerGenerator, Discriminator,     RNNInitialStateType, AttrDiscriminator
from gan.util import add_gen_flag, normalize_per_sample,     renormalize_per_sample


# In[3]:


def normalize_attribute(data_att, data_att_outputs, data_att_min, data_att_max):
    
    data_att_norm = data_att
    total_dim = 0
    for output in data_att_outputs:
        if output.type_ == OutputType.CONTINUOUS:
            for _ in range(output.dim):
                data_att_norm[:, total_dim] = (data_att_norm[:, total_dim] - data_att_min[total_dim]) / (data_att_max[total_dim] - data_att_min[total_dim])
                if output.normalization == Normalization.MINUSONE_ONE:
                    data_att_norm[:, total_dim] = data_att_norm[:, total_dim] * 2.0 - 1.0

                total_dim += 1
        else:
            total_dim += output.dim


    return data_att_norm

attributes = np.load('../seasia_att_batch2.npy')
train_sample_size = attributes.shape[0]

features = np.load('../seasia_feat_batch2.npy')
features = features.reshape(-1,119,1)

gen_flags = np.ones((train_sample_size,119))


data_feature_outputs = [
	output.Output(type_=OutputType.CONTINUOUS,dim=1,normalization=Normalization.ZERO_ONE,is_gen_flag=False)]
  #hourly solar radiation

data_attribute_outputs = [
	output.Output(type_=OutputType.CONTINUOUS,dim=1,normalization=Normalization.MINUSONE_ONE,is_gen_flag=False),
  #lat
	output.Output(type_=OutputType.CONTINUOUS,dim=1,normalization=Normalization.MINUSONE_ONE,is_gen_flag=False),
  #longi
  output.Output(type_=OutputType.CONTINUOUS,dim=1,normalization=Normalization.ZERO_ONE,is_gen_flag=False),
  #height
  output.Output(type_=OutputType.CONTINUOUS,dim=2,normalization=Normalization.MINUSONE_ONE,is_gen_flag=False),
  #norm vector (xy)
  #output.Output(type_=OutputType.CONTINUOUS,dim=1,normalization=Normalization.ZERO_ONE,is_gen_flag=False),
  #glzr
  output.Output(type_=OutputType.CONTINUOUS,dim=32,normalization=Normalization.MINUSONE_ONE,is_gen_flag=False),
  #latent_im
  output.Output(type_=OutputType.CONTINUOUS,dim=1,normalization=Normalization.ZERO_ONE,is_gen_flag=False),
  #mon
  output.Output(type_=OutputType.CONTINUOUS,dim=1,normalization=Normalization.MINUSONE_ONE,is_gen_flag=False),
  #inc
  output.Output(type_=OutputType.CONTINUOUS,dim=8,normalization=Normalization.ZERO_ONE,is_gen_flag=False)]
  #weather_stat
  #output.Output(type_=OutputType.DISCRETE,dim=656,normalization=None,is_gen_flag=False)]
  #shadow mask


#necessary inputs
data_all = features
data_attribut = attributes
data_gen_flag = gen_flags

sample_len = 17

# normalise data
(data_feature, data_attribute, data_attribute_outputs,
 real_attribute_mask) = normalize_per_sample(
        data_all, data_attribut, data_feature_outputs,
        data_attribute_outputs)

# add generation flag to features
data_feature, data_feature_outputs = add_gen_flag(
    data_feature, data_gen_flag, data_feature_outputs, sample_len)


data_attribute_min = np.amin(data_attribute, axis=0)
data_attribute_max = np.amax(data_attribute, axis=0)
np.save('seasia_att_batch2_min.npy',data_attribute_min)
np.save('seasia_att_batch2_max.npy',data_attribute_max)


data_attribute_normlized = normalize_attribute(data_attribute, data_attribute_outputs, data_attribute_min, data_attribute_max)


generator = DoppelGANgerGenerator(
    feed_back=True,
    noise=True,
    feature_outputs=data_feature_outputs,
    attribute_outputs=data_attribute_outputs,
    real_attribute_mask=real_attribute_mask,
    attribute_num_units =100,
    sample_len=sample_len,
    feature_num_units=100,
    feature_num_layers=2)

discriminator = Discriminator(num_units=100)
attr_discriminator = AttrDiscriminator(num_units=100)

checkpoint_dir = "./results/checkpoint"
sample_dir = "./results/sample"
time_path = "./results/time/time.txt"
epoch = 200
batch_size = 100
g_lr = 0.0001
d_lr = 0.0001 
vis_freq = 1000
vis_num_sample = 1
d_rounds = 3
g_rounds = 1
d_gp_coe = 10.0
attr_d_gp_coe=10.0
attr_d_lr = 0.0001
g_attr_d_coe = 1.0
extra_checkpoint_freq = 1000
num_packing = 1


# config
run_config = tf.ConfigProto()
tf.reset_default_graph() # if you are using spyder 
with tf.Session(config=run_config) as sess:
    gan = DoppelGANger(
        sess=sess, 
        checkpoint_dir=checkpoint_dir,
        sample_dir=sample_dir,
        time_path=time_path,
        epoch=epoch,
        batch_size=batch_size,
        data_feature=data_feature,
        data_attribute=data_attribute_normlized,
        real_attribute_mask=real_attribute_mask,
        data_gen_flag=data_gen_flag,
        sample_len=sample_len,
        data_feature_outputs=data_feature_outputs,
        data_attribute_outputs=data_attribute_outputs,
        vis_freq=vis_freq,
        vis_num_sample=vis_num_sample,
        generator=generator,
        discriminator=discriminator,
        attr_discriminator=attr_discriminator,
        d_gp_coe=d_gp_coe,
        attr_d_gp_coe=attr_d_gp_coe,
        g_attr_d_coe=g_attr_d_coe,
        d_rounds=d_rounds,
        g_rounds=g_rounds,
g_lr=g_lr,
d_lr=d_lr,
attr_d_lr = attr_d_lr,
        num_packing=num_packing,
        extra_checkpoint_freq=extra_checkpoint_freq)

    #building & training
    gan.build()
    #gan.load(checkpoint_dir)
    gan.train()

