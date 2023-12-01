images: []const f64,
labels: []const u8,
count: usize,

const Self = @This();

pub fn batch(self: Self, offset: usize, s: usize) Self {
    if (offset >= self.count) {
        @panic("offset is too big");
    }
    const per_image = self.images.len / self.count;
    const per_image_labels = self.count / self.labels.len;
    const real_count = @min(self.count - offset, s);
    const real_end = offset + real_count;
    //@import("std").debug.print("real_end: {d}\noffset: {d}\nlabels: {d}\n", .{ real_end, offset, self.train_labels.len });
    //@import("std").debug.print("image length: {}, per image: {}\n", .{ self.train_images.len, per_image });
    //@import("std").debug.print("per image: {} offset: {} real end: {} real_count: {} \n", .{ per_image, offset, real_end, real_count });
    //@import("std").debug.print("offset {} \n", .{offset});

    return Self{
        .count = real_count,
        .images = self.images[(offset * per_image)..(real_end * per_image)],
        .labels = self.labels[offset * per_image_labels .. real_end * per_image_labels],
    };
}
