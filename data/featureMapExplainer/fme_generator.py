# Created by jing at 26.06.23
from data.featureMapExplainer.patterns import p_color
import config


def main():
    width = 64
    height = 64
    p_color.generate(config.data_path, width, height, train_num=100, test_num=20)
    print("done!")


if __name__ == "__main__":
    main()
