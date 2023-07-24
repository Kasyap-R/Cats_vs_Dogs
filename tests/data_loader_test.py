import unittest
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from src.data.preprocessor import ImageDatasetLoader

class TestImageDatasetLoader(unittest.TestCase):

    def setUp(self):
        train_data_path = 'D:/ML_Data/Cats_vs_Dogs/train'
        test_data_path = 'D:/ML_Data/Cats_vs_Dogs/test1'
        batch_size = 128
        validation_split = 0.2
        shuffle_dataset = True
        self.data_loader = ImageDatasetLoader(train_data_path, test_data_path, batch_size, validation_split, shuffle_dataset)
    
    def test_load_train_val(self):
        training_data, validation_data = self.data_loader.load_train_val()
        self.assertGreater(len(training_data), 0)
        self.assertGreater(len(validation_data), 0)

    def test_load_test(self):
        testing_data = self.data_loader.load_test()
        self.assertGreater(len(testing_data), 0)

    def test_visualize_data(self):
        training_data, _ = self.data_loader.load_train_val()
        num_images_to_show = 3
        shown = 0
        
        for images, labels in training_data:
            for i in range(len(images)):
                if shown >= num_images_to_show:
                    break
                plt.imshow(transforms.ToPILImage()(images[i]))
                plt.title("Label: {}".format(labels[i]))
                plt.show()
                shown += 1
            if shown >= num_images_to_show:
                break


if __name__ == "__main__":
    unittest.main()