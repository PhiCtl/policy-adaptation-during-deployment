import torch
import torch.nn as nn
import utils


OUT_DIM = {2: 39, 4: 35, 6: 31, 8: 27, 10: 23, 11: 21, 12: 19}
OUT_LATENT_DIM = {201: 23 * 5}


def tie_weights(src, trg):
	assert type(src) == type(trg)
	trg.weight = src.weight
	trg.bias = src.bias


class CenterCrop(nn.Module):
	"""Center-crop if observation is not already cropped"""
	def __init__(self, size):
		super().__init__()
		assert size == 84
		self.size = size

	def forward(self, x):
		assert x.ndim == 4, 'input must be a 4D tensor'
		if x.size(2) == self.size and x.size(3) == self.size:
			return x
		elif x.size(-1) == 100:
			return x[:, :, 8:-8, 8:-8]
		else:
			return ValueError('unexepcted input size')


class NormalizeImg(nn.Module):
	"""Normalize observation"""
	def forward(self, x):
		return x/255.


class PixelEncoder(nn.Module):
	"""Convolutional encoder of pixel observations"""
	def __init__(self, obs_shape, feature_dim, num_layers=4, num_filters=32, num_shared_layers=4):
		super().__init__()
		assert len(obs_shape) == 3

		self.feature_dim = feature_dim
		self.num_layers = num_layers
		self.num_shared_layers = num_shared_layers

		self.preprocess = nn.Sequential(
			CenterCrop(size=84), NormalizeImg()
		)

		self.convs = nn.ModuleList(
			[nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
		)
		for i in range(num_layers - 1):
			self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

		out_dim = OUT_DIM[num_layers]
		self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
		self.ln = nn.LayerNorm(self.feature_dim)

	def forward_conv(self, obs, detach=False):
		obs = self.preprocess(obs)
		conv = torch.relu(self.convs[0](obs))

		for i in range(1, self.num_layers):
			conv = torch.relu(self.convs[i](conv))
			if i == self.num_shared_layers-1 and detach:
				conv = conv.detach()

		h = conv.view(conv.size(0), -1)
		return h

	def forward(self, obs, detach=False):
		h = self.forward_conv(obs, detach)
		h_fc = self.fc(h)
		h_norm = self.ln(h_fc)
		out = torch.tanh(h_norm)

		return out

	def copy_conv_weights_from(self, source, n=None):
		"""Tie n first convolutional layers"""
		if n is None:
			n = self.num_layers
		for i in range(n):
			tie_weights(src=source.convs[i], trg=self.convs[i])

	def tie_encoder_from(self, source):
		# Copy convolutional layers
		self.copy_conv_weights_from(source)
		# Copy linear layer
		tie_weights(trg=self.fc, src=source.fc)

	def verify_weights_from(self, source):
		is_equal = True
		n = self.num_layers
		for i in range(n):
			if not utils.verify_weights(src=source.convs[i], trg=self.convs[i]):
				is_equal = False
		if not utils.verify_weights(trg=self.fc, src=source.fc):
			is_equal = False
		return is_equal



class TemporalEncoder(nn.Module):
	"""1D Convolutional encoder for time series feature extraction"""

	def __init__(self, input_dim, output_dim, num_channels, num_filters):
		super().__init__()


		self.conv1 = nn.Conv1d(num_channels, num_filters, 6) # (input - kernel + 1) / stride + 1
		self.pool1 = nn.MaxPool1d(4)
		self.conv2 = nn.Conv1d(num_filters, num_filters, 5, stride=2)

		self.fc = nn.Linear(OUT_LATENT_DIM[input_dim], output_dim)

	def forward(self, dynamics):

		conv = torch.relu(self.conv1(dynamics))
		pooled = self.pool1(conv)
		conv = self.conv2(pooled)

		h = conv.view(conv.size(0), -1)
		output = self.fc(h)
		return output


	def copy_conv_weights_from(self, source):
		tie_weights(src=source.conv1, trg=self.conv1)
		tie_weights(src=source.conv2, trg=self.conv2)
    
	

def make_encoder(
	obs_shape, feature_dim, num_layers, num_filters, num_shared_layers
):
	assert num_layers in OUT_DIM.keys(), 'invalid number of layers'
	if num_shared_layers == -1 or num_shared_layers == None:
		num_shared_layers = num_layers
	assert num_shared_layers <= num_layers and num_shared_layers > 0, \
		f'invalid number of shared layers, received {num_shared_layers} layers'
	return PixelEncoder(
		obs_shape, feature_dim, num_layers, num_filters, num_shared_layers
	)

def make_temp_encoder(
		input_dim, output_dim, num_channels=3, num_filters=5
):
	return TemporalEncoder(input_dim, output_dim, num_channels, num_filters)
