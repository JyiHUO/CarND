import tensorflow as tf
import torch
def add_summary_value(writer, key, value, step):
    summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
    writer.add_summary(summary, step)

if __name__ == "__main__":
    print (123)