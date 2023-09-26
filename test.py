from setup_imagenet_hash import ImageNet, ImageNet_HashModel


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", "--path", type=str)
    args = vars(parser.parse_args())
    data = ImageNet(0,5, args["path"])
    print(data.file_list)
    try:
        item = data.file_list[4]
    except IndexError:
        print(args["path"])