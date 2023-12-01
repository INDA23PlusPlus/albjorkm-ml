const std = @import("std");
const RndGen = std.rand.DefaultPrng;
const DataSet = @import("data.zig");

const input_size = 784;
const output_size = 10;

const NetLoadErrors = error{IncompatibleModel};

/// Note that there is only a single layer, hence
/// the neural network is shallow and not deep!
pub const ShallowNeuralNetwork = struct {
    biases: [output_size]f64,
    weights: [output_size][input_size]f64,

    activations: [output_size]f64,

    gradient_biases: [output_size]f64,
    gradient_weights: [output_size][input_size]f64,

    pub fn load() !ShallowNeuralNetwork {
        var result = new();
        const file = try std.fs.cwd().openFile("output.dat", .{});
        const saved_size = output_size * 4 + input_size * output_size * 4;
        var data: [saved_size]u8 = undefined;
        if (try file.readAll(&data) != saved_size) {
            return NetLoadErrors.IncompatibleModel;
        }

        const biases: [output_size]f32 = std.mem.bytesToValue([output_size]f32, data[0 .. output_size * 4]);
        for (biases, 0..) |b, i| {
            result.biases[i] = b;
        }

        const weights: [output_size][input_size]f32 = std.mem.bytesToValue([output_size][input_size]f32, data[output_size * 4 ..]);
        for (weights, 0..) |weights_line, i| {
            for (weights_line, 0..) |w, j| {
                result.weights[i][j] = w;
            }
        }

        return result;
    }

    pub fn save(self: *ShallowNeuralNetwork) !void {
        var file = try std.fs.cwd().createFile("output.dat", .{});
        _ = try file.write(std.mem.sliceAsBytes(&self.biases));
        _ = try file.write(std.mem.sliceAsBytes(&self.weights));
        try file.sync();
        file.close();
    }

    pub fn new() ShallowNeuralNetwork {
        var result = ShallowNeuralNetwork{
            .biases = undefined,
            .weights = undefined,
            .activations = undefined,
            .gradient_biases = undefined,
            .gradient_weights = undefined,
        };

        var generator = RndGen.init(0);
        var ngen = generator.random();

        for (0..output_size) |i| {
            result.biases[i] = ngen.float(f64);
            for (0..input_size) |j| {
                result.weights[i][j] = ngen.float(f64);
            }
        }

        return result;
    }

    pub fn soft_max(self: *ShallowNeuralNetwork) void {
        var max = self.activations[0];
        var sum: f64 = 0;
        for (1..output_size) |i| {
            max = @max(max, self.activations[i]);
        }
        for (0..output_size) |i| {
            self.activations[i] = @exp(self.activations[i] - max);
            sum += self.activations[i];
        }
        for (0..output_size) |i| {
            self.activations[i] /= sum;
        }
    }

    pub fn forward(self: *ShallowNeuralNetwork, previous_layer: []const f64) void {
        for (0..output_size) |i| {
            self.activations[i] = self.biases[i];
            for (0..input_size) |j| {
                self.activations[i] += self.weights[i][j] * previous_layer[j];
            }
        }

        // Makes sure the outputs are not totally crazy.
        self.soft_max();
    }

    pub fn gradient(self: *ShallowNeuralNetwork, previous_layer: []const f64, label: u8) f64 {
        self.forward(previous_layer);
        for (0..output_size) |i| {
            var b_gradient = self.activations[i];
            if (i == label) {
                b_gradient -= 1;
            }

            for (0..input_size) |j| {
                var w_gradient = b_gradient * previous_layer[j];
                self.gradient_weights[i][j] += w_gradient;
            }

            self.gradient_biases[i] += b_gradient;
        }

        return 0 - @log(self.activations[label]);
    }

    pub fn train(self: *ShallowNeuralNetwork, rate: f64, data: DataSet) f64 {
        @memset(self.gradient_biases[0..], 0);
        for (self.gradient_weights[0..]) |*weights| {
            @memset(weights, 0);
        }

        var loss: f64 = 0;
        for (0..data.count) |i| {
            loss += self.gradient(data.images[input_size * i ..], data.labels[i]);
        }

        for (0..output_size) |i| {
            const count: f64 = @floatFromInt(data.count);
            self.biases[i] -= rate * self.gradient_biases[i] / count;
            for (0..input_size) |j| {
                self.weights[i][j] -= rate * self.gradient_weights[i][j] / count;
            }
        }

        return loss;
    }
};

pub fn measure_model_accuracy(net: *ShallowNeuralNetwork, data: *const DataSet) f64 {
    var correct_guesses: f64 = 0;
    for (0..data.count) |i| {
        const per_image = data.images.len / data.count;
        const per_image_label = data.labels.len / data.count;
        const input = data.images[per_image * i ..];
        net.forward(input);
        var max_value = net.activations[0];
        var max_index: usize = 0;
        for (net.activations, 0..) |value, j| {
            if (value > max_value) {
                max_index = j;
                max_value = value;
            }
        }
        const expected = data.labels[per_image_label * i];
        if (max_index == expected) {
            correct_guesses += 1;
        }
    }

    const count: f64 = @floatFromInt(data.count);
    return correct_guesses / count;
}
