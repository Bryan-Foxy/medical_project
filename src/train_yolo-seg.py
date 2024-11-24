import os 
import argparse
from ultralytics import YOLO

def training_yolo(model, data, epochs, imgsz, batch, device, save_directory):
	model = YOLO(model)
	model.train(
		data = data,
		epochs = epochs,
		imgsz = imgsz,
		batch = batch,
		device = device,
		project = save_directory,
		name = "yolo_logs"
		)
	print("Training complete ! Results saved as {}".format(save_directory))


def parse_args():
	"""Arguments for our script YOLO"""
	parser = argparse.ArgumentParser(description = "Train YOLO model")
	parser.add_argument('--data', type = str, required = True, help = "yaml file to provide path for datasets")
	parser.add_argument('--model', type = str, default = "yolov8m-seg.pt", help = "Pretrained model YOLO from ultralytics")
	parser.add_argument('--epochs', type = int, default = 50, help = "Number of epochs")
	parser.add_argument('--imgsz', type = int, default = 640, help = "Image size")
	parser.add_argument('--batch', type = int, default = 16, help = "Number of batch")
	parser.add_argument('--device', type = str, default = "cpu", help = "Device to make compuation (cpu | cuda | mps)")
	parser.add_argument('--save_directory', type = str, default = "../saves/", help = "Output directory")

	return parser.parse_args()

def main():
	args = parse_args()
	training_yolo(
		model = args.model,
		data = args.data,
		epochs = args.epochs,
		imgsz = args.imgsz,
		batch = args.batch,
		device = args.device,
		save_directory = args.save_directory
		)

if __name__ == '__main__':
	main()


