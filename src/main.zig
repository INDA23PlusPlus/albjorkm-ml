const std = @import("std");
const mnist = @import("mnist.zig");
const nn = @import("nn.zig");
const TrainingData = @import("data.zig");

const batch_size = 100;

pub fn main() !void {
    const data = mnist.trainingData();
    const test_data = mnist.testData();

    var last_net = nn.ShallowNeuralNetwork.load();
    const true_accuracy = nn.measure_model_accuracy(&last_net, &test_data);
    std.debug.print("last net's accuracy was: {d}\n", .{true_accuracy});

    //std.debug.print("debug: {any}\n", .{true_net.biases});
    //if (true_accuracy != 0.9) {
    // Work around for unreachable code Zig.
    //    return;
    //}

    //var net = nn.ShallowNeuralNetwork.load();
    var net = nn.ShallowNeuralNetwork.new();
    //std.debug.print("Network: {}", .{net});

    //_ = batches;
    var accuracy: f64 = 0;
    for (0..1000) |step| {
        //std.debug.print("creating batch of size: {d}", .{batch_size});
        const batch = data.batch((step * batch_size) % data.count, batch_size);
        //std.debug.print("batch: {any}", .{batch.train_labels});
        //std.debug.print("images: {d}", .{batch.train_images.len});
        var loss = net.train(0.5, batch);

        if (step % 100 == 0) {
            accuracy = nn.measure_model_accuracy(&net, &test_data);
            std.debug.print("step: {d}/1000, loss: {d}, accuracy: {d}\n", .{ step, loss / batch_size, accuracy });
        } else {
            std.debug.print("step: {d}/1000, loss: {d}\n", .{ step, loss / batch_size });
        }
    }
    accuracy = nn.measure_model_accuracy(&net, &test_data);
    std.debug.print("accuracy after 1000 steps: {d}\n", .{accuracy});

    try net.save();
}
