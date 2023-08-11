import getpass
import json


class STATIC_VALUES:
    base_dir = 'D:\\Other\\Dataset\\Amazon\\' if getpass.getuser() == 'Reza' else 'D:\\Data\\amazon\\'
    local_dir = base_dir + '1.local\\'
    jpg_dir = base_dir + 'train-jpg'
    tif_dir = base_dir + 'train-tif-v2'
    test_tif_dir = base_dir + 'test-tif-v2'
    test_jpg_dir = base_dir + 'test-tif-v2'

    augmented_jpg_dir = base_dir + 'aug-jpg'
    augmented_tif_dir = base_dir + 'aug-tif'

    xception_graph_location = local_dir + 'graphs/xception/'

    weather_labels = ['clear', 'haze', 'partly_cloudy', 'cloudy']

    ground_common_labels = ['cultivation', 'primary', 'water', 'habitation', 'bare_ground', 'agriculture', 'road']

    ground_rare_labels = ['artisinal_mine', 'blow_down', 'selective_logging', 'conventional_mine', 'slash_burn',
                          'blooming']

    labels = weather_labels.copy()
    labels.extend(ground_common_labels)
    labels.extend(ground_rare_labels)
    labels_count = len(labels)

    weather_labels_count = len(weather_labels)
    common_labels_count = len(ground_common_labels)
    rare_labels_count = len(ground_rare_labels)
    ground_labels_count = common_labels_count + rare_labels_count

    batch_size = 5 if getpass.getuser() == 'Reza' else 8
    batches_in_memory = 150 if getpass.getuser() == 'Reza' else 150
    images_per_file = 10000

    image_size = [256, 256]

    train_config = {}
    with open('train-config.json', 'r') as f:
        train_config = json.load(f)
