import torch
import json
import pickle 
import os
from torch.autograd import Variable
from image_caption.build_vocab import Vocabulary
from torchvision import transforms 

from image_caption.model import EncoderCNN, DecoderRNN
from PIL import Image

with open("image_caption/pets_captions.pkl", 'rb') as f:
	pets_cap = pickle.load(f)

class ImageCaption:

	def __init__(self):
		# Image preprocessing
		self.transform = transforms.Compose([
											transforms.ToTensor(),
											transforms.Normalize((0.485, 0.456, 0.406),
											(0.229, 0.224, 0.225))])

		# Load vocabulary wrapper
		# self. = Vocabulary()
		with open("/root/server/image_caption/data/vocab.pkl", 'rb') as f:
			self.vocab = pickle.load(f)


		# Build Models
		self.encoder = EncoderCNN(256)
		self.encoder.eval()  # evaluation mode (BN uses moving mean/variance)
		self.decoder = DecoderRNN(256, 512,
			len(self.vocab), 1)


		# Load the trained model parameters
		self.encoder.load_state_dict(torch.load("image_caption/models/encoder-5-3000.pkl"))
		self.decoder.load_state_dict(torch.load("image_caption/models/decoder-5-3000.pkl"))

		# If use gpu
		if torch.cuda.is_available():
			self.encoder.cuda()
			self.decoder.cuda()

	def to_var(self, x, volatile=False):
		if torch.cuda.is_available():
			x = x.cuda()
		return Variable(x, volatile=volatile)

	def load_image(self, image, isbase64=True, transform=None):
		if not isbase64:
			image = Image.open(image)
		image = image.resize([224, 224], Image.LANCZOS)

		if transform is not None:
			image = transform(image).unsqueeze(0)

		return image

	def generate_caption(self, image_path, num_retrieval ,isbase64, used=False):
		if used:
			image_path_ab = image_path.split('/')[-1]
			return pets_cap[image_path_ab]
		image = self.load_image(image_path, isbase64, transform=self.transform)
		image_tensor = self.to_var(image, volatile=True)

		# Generate caption from image
		feature = self.encoder(image_tensor)

		sampled_ids = self.decoder.sample(feature)
		sampled_ids = sampled_ids.cpu().data.numpy()

		# Decode word_ids to words
		sampled_caption = []
		for word_id in sampled_ids:
			word = self.vocab.idx2word[word_id]
			if word == '<end>':
				break
			if word == '<start>':
				continue
			sampled_caption.append(word)

		return ' '.join(sampled_caption)
if __name__ == '__main__':
	caption_model=ImageCaption()
