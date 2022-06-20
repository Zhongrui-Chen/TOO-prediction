import argparse
import logging

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('n')
#     args = parser.parse_args()
#     n = args.n if args.n else 'NaN'
#     print(n)

if __name__ == '__main__':
    logging.basicConfig(filename='./foo.log', encoding='utf-8', level=logging.DEBUG)
    logging.info('I told you so')