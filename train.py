import torch
import argparse

def main():
    pass

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="CNN GRU-SeqCap")
    args.add_argument("-c", "--config", default=None, type=str, help = "config file path")
    main()