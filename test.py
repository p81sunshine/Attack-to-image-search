from setup_imagenet_hash import ImageNet, ImageNet_HashModel


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", "--path", type=str)
    args = vars(parser.parse_args())
    data = ImageNet(0,5, args["path"])
    if (len(data.test_data) != 5):
        print("ERROR")
