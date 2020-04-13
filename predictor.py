from detectron2.config import get_cfg
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.data import MetadataCatalog
	
import torch
import os
from detectron2.utils.visualizer import ColorMode, Visualizer

class BatchPredictor:
	"""
		pred = Predictor(cfg,model_path='output',classes=["checkbox"])
		inputs = cv2.imread("input.jpg")
		outputs = pred([inputs])
	"""
	def __init__(self, cfg_file, model_path, classes, confidence_thresh=0.5):
		self.cpu_device = torch.device("cpu")
		
		self.cfg = get_cfg()
		self.cfg.merge_from_file(cfg_file)
		self.cfg.MODEL.WEIGHTS = os.path.join(model_path, "model_final.pth")
		self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_thresh
		self.model = build_model(self.cfg)
		self.model.eval()
		
		MetadataCatalog.get(self.cfg.DATASETS.TEST[0]).thing_classes = classes
		self.metadata = MetadataCatalog.get(self.cfg.DATASETS.TEST[0])

		checkpointer = DetectionCheckpointer(self.model)
		checkpointer.load(self.cfg.MODEL.WEIGHTS)

		self.transform_gen = T.ResizeShortestEdge(
			[self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MIN_SIZE_TEST], self.cfg.INPUT.MAX_SIZE_TEST
		)

		self.input_format = self.cfg.INPUT.FORMAT
		assert self.input_format in ["RGB", "BGR"], self.input_format

	def __call__(self, images, return_images=False):
		"""
		Args:
			original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

		Returns:
			predictions (dict):
				the output of the model for one image only.
				See :doc:`/tutorials/models` for details about the format.
		"""
		inputs = []
		with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
			# Apply pre-processing to image.
			#prepare a list of dict objects
			for original_image in images:
				if self.input_format == "RGB":
					# whether the model expects BGR inputs or RGB
					original_image = original_image[:, :, ::-1]

				height, width = original_image.shape[:2]
				image = self.transform_gen.get_transform(original_image).apply_image(original_image)
				image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

				inputs.append({"image": image, "height": height, "width": width})
			predictions = self.model(inputs)
			
			output_images = []
			if return_images:
				for image, prediction in zip(images,predictions):
					image = image[:, :, ::-1]
					visualizer = Visualizer(image, MetadataCatalog.get(self.cfg.DATASETS.TEST[0]), instance_mode=ColorMode.IMAGE)
					instances = prediction["instances"].to(self.cpu_device)
					vis_output = visualizer.draw_instance_predictions(predictions=instances)
					output_images.append(vis_output.get_image()[:, :, ::-1])
					
			return predictions, output_images
