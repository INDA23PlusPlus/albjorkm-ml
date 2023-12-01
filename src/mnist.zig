const std = @import("std");
const DataSet = @import("data.zig");

const pixels_per_image = 784;
const mnist_train_count = 60_000;
const mnist_test_count = 10_000;

var training_images: [mnist_train_count * pixels_per_image]f64 = undefined;
var test_images: [mnist_test_count * pixels_per_image]f64 = undefined;

fn loadMnistData(images_into: []f64, comptime input_idx: []const u8, comptime labels_idx: []const u8, data_set_size: usize) DataSet {
    for (input_idx[16..], 0..) |pixel, i| {
        var f: f64 = @floatFromInt(pixel);
        images_into[i] = f / 255;
    }

    return DataSet{
        .images = images_into,
        .labels = labels_idx[8..],
        .count = data_set_size,
    };
}

pub fn trainingData() DataSet {
    return loadMnistData(&training_images, @embedFile("mnist/train-images.idx3-ubyte"), @embedFile("mnist/train-labels.idx1-ubyte"), mnist_train_count);
}

pub fn testData() DataSet {
    return loadMnistData(&test_images, @embedFile("mnist/t10k-images.idx3-ubyte"), @embedFile("mnist/t10k-labels.idx1-ubyte"), mnist_test_count);
}
