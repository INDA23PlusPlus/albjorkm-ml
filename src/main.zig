const std = @import("std");
const mnist = @import("mnist.zig");
const nn = @import("nn.zig");

const batch_size = 100;

pub fn main() !void {
    const data = mnist.trainingData();
    const test_data = mnist.testData();

    if (nn.ShallowNeuralNetwork.load()) |last_net_const| {
        var last_net = last_net_const;
        const true_accuracy = nn.measure_model_accuracy(&last_net, &test_data);
        std.debug.print("last net's accuracy was: {d}\n", .{true_accuracy});
    } else |_| {
        std.debug.print("could not load previous model\n", .{});
    }

    var net = nn.ShallowNeuralNetwork.new();

    var accuracy: f64 = 0;
    for (0..1000) |step| {
        const batch = data.batch((step * batch_size) % data.count, batch_size);
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
